import re
import logging
import os
import tomli
import uuid
import redis
import glob
import yaml
from pathlib import Path
from typing import List

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.callbacks import CallbackManager, BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema import Document

try:
    from langchain.callbacks import StdOutCallbackHandler
except ImportError:
    class StdOutCallbackHandler(BaseCallbackHandler):
        def on_llm_start(self, serialized, prompts, **kwargs):
            print(f"LLM started. Prompts: {prompts}")

        def on_llm_end(self, response, **kwargs):
            print(f"LLM ended. Output: {response}")

        def on_chain_start(self, serialized, inputs, **kwargs):
            print(f"Chain started. Inputs: {inputs}")

        def on_chain_end(self, outputs, **kwargs):
            print(f"Chain ended. Outputs: {outputs}")

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_lc_policy_util_v2")

def load_config(config_path="./config/config.toml"):
    """
    Load configuration from a TOML file, returning a dict with nested sections.
    """
    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        logger.info(f"Loaded TOML config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def validate_redis_connection(redis_url):
    """
    Validates if the Redis server is reachable at the given URL.
    Returns True if successful, False otherwise.
    Logs errors and raises if not reachable.
    """
    try:
        logger.info(f"Attempting to connect to Redis at {redis_url}...")
        redis_client = redis.Redis.from_url(redis_url)
        if redis_client.ping():
            logger.info("Successfully connected to Redis server.")
            return True
        else:
            logger.error("Redis server did not respond to PING.")
            raise ConnectionError("Redis server did not respond to PING.")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise

# -----------------------------------------------------------------------------
# Multi-file Loader and Chunker
# -----------------------------------------------------------------------------
def parse_frontmatter(md_text):
    """
    Extract YAML frontmatter (between ---) and return metadata dict + content.
    """
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", md_text, re.DOTALL)
    if match:
        meta_raw, content = match.groups()
        metadata = yaml.safe_load(meta_raw)
        return metadata, content
    else:
        # No frontmatter found
        return {}, md_text

def load_and_chunk_markdown_docs(folder, chunk_size=1000) -> List[Document]:
    """
    Loads all .md files in the folder, extracts metadata, chunks content
    using MarkdownHeaderTextSplitter (header 1-4), and returns a list of Document objects.
    Each chunk contains the document-level metadata + header section info.
    Preserves tables and code blocks as much as possible.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100,
        separators=["\n\n", "\n", "|", " ", ""],
        keep_separator=True
    )
    docs = []
    for md_file in glob.glob(os.path.join(folder, "*.md")):
        with open(md_file, "r", encoding="utf-8") as f:
            md_text = f.read()
        metadata, content = parse_frontmatter(md_text)
        # Use MarkdownHeaderTextSplitter to keep header structure
        header_chunks = markdown_splitter.split_text(content)
        for doc in header_chunks:
            # Further split if too large (while preserving metadata)
            if len(doc.page_content) > chunk_size:
                sub_docs = text_splitter.split_documents([doc])
                for sub_doc in sub_docs:
                    # Merge doc-level metadata and header metadata
                    sub_doc.metadata = {**metadata, **sub_doc.metadata}
                    docs.append(sub_doc)
            else:
                doc.metadata = {**metadata, **doc.metadata}
                docs.append(doc)
    logger.info(f"Loaded and chunked {len(docs)} chunks from {folder}")
    return docs

# -----------------------------------------------------------------------------
# LangChain Callback Handlers
# -----------------------------------------------------------------------------
class FileLoggerCallbackHandler(BaseCallbackHandler):
    def __init__(self, filename="langchain_events.log", logs_dir="logs"):
        if not os.path.isabs(filename):
            logs_dir = logs_dir or "logs"
            os.makedirs(logs_dir, exist_ok=True)
            filename = os.path.join(logs_dir, filename)
        else:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename

    def _log(self, message):
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._log(f"LLM started. Prompts: {prompts}")

    def on_llm_end(self, response, **kwargs):
        self._log(f"LLM ended. Output: {response}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        self._log(f"Chain started. Inputs: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        self._log(f"Chain ended. Outputs: {outputs}")

def get_callback_manager_from_config(config):
    """
    Create a CallbackManager with both StdOut and File Logging using config,
    only if langchain_logging.active is True.
    """
    lc_logging_config = config.get("langchain_logging", {})
    active = lc_logging_config.get("active", False)
    if not active:
        return None
    logs_dir = lc_logging_config.get("logs_dir", "logs")
    file_log_name = lc_logging_config.get("file_log_name", "langchain_events.log")
    os.makedirs(logs_dir, exist_ok=True)
    handlers = [
        StdOutCallbackHandler(),
        FileLoggerCallbackHandler(file_log_name, logs_dir)
    ]
    return CallbackManager(handlers)

# -----------------------------------------------------------------------------
# Prompt Template Selection
# -----------------------------------------------------------------------------
def get_default_prompt_from_config(config):
    """
    Returns the default PromptTemplate defined in config['prompts'].
    Logs which prompt is selected as default, including its name and the full template.
    """
    prompts = config.get("prompts", [])
    for prompt in prompts:
        if prompt.get("default", False):
            prompt_name = prompt.get("name", "unnamed")
            logger.info(
                f"Using prompt template: {prompt_name} (default)\n"
                f"Prompt name: {prompt_name}\n"
                f"Template (truncated to 100 chars):\n{prompt['template'][:100]}"
            )
            return PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=prompt["template"]
            )
    logger.warning("No default prompt found in config['prompts']. Using LangChain's built-in default.")
    return None

# -----------------------------------------------------------------------------
# RAG System
# -----------------------------------------------------------------------------
class RAGSystem:
    """
    Main RAG (Retrieval-Augmented Generation) system with table-aware processing
    and conversation memory for multi-turn chat.
    Supports persistent chat memory using Redis if configured.
    """

    def __init__(self, config, session_id=None, callback_manager=None):
        """
        Accepts nested config dictionary with sections:
        data, embedding, llm, cache, server, logging, memory.
        Optionally accepts a LangChain callback_manager for instrumentation/logging.
        """
        self.config = config
        self.session_id = session_id or f"inmem-{uuid.uuid4()}"
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.callback_manager = callback_manager
        self.llm = None  # Store LLM instance for later use (summarization, etc.)

    def setup_embeddings(self):
        """
        Initialize embedding model for document vectorization.
        """
        try:
            embedding_model = self.config["embedding"]["embedding_model"]
            ollama_base_url = self.config["llm"]["ollama_base_url"]
            self.embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=ollama_base_url
            )
            logger.info(f"Initialized embeddings: {embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise

    def create_vectorstore(self, documents):
        """
        Create a FAISS vectorstore from documents using the embeddings.
        """
        try:
            search_k = self.config["embedding"]["retriever_k"]
            if not self.embeddings:
                raise ValueError("Embeddings not initialized. Call setup_embeddings() first.")
            self.vectorstore = FAISS.from_documents(documents, embedding=self.embeddings)
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": search_k}
            )
            logger.info(f"Created vectorstore with {len(documents)} document chunks")
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            raise

    def setup_chain(self):
        """
        Creates a ConversationalRetrievalChain with conversation memory.
        If redis_url is provided and valid, uses Redis for persistent memory.
        Otherwise, uses in-memory (non-persistent) buffer.
        The session_id should be provided for per-user/session support.
        Passes callback_manager to the LLM and chain if provided.
        Also injects the default prompt template if specified in config.
        """
        llm_cfg = self.config["llm"]
        memory_cfg = self.config.get("memory", {})
        self.llm = ChatOllama(
            model=llm_cfg["llm_model_name"],
            base_url=llm_cfg["ollama_base_url"],
            temperature=llm_cfg.get("llm_temperature", 0.0),
            num_predict=512,
            callback_manager=self.callback_manager
        )
        memory = None

        redis_url = memory_cfg.get("redis_url")
        session_id = self.session_id

        if redis_url:
            try:
                validate_redis_connection(redis_url)
                chat_history = RedisChatMessageHistory(
                    session_id=session_id,
                    url=redis_url
                )
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    chat_memory=chat_history
                )
                logger.info(f"Using RedisChatMessageHistory for persistent conversation memory for session_id: {session_id}")
            except Exception as e:
                logger.warning(f"Falling back to in-memory buffer due to Redis validation error: {e}")

        if not memory:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            logger.info(f"Using in-memory ConversationBufferMemory (not persistent) for session_id: {session_id}")

        # Use default prompt from config, if present
        prompt = get_default_prompt_from_config(self.config)
        chain_kwargs = {}
        if prompt is not None:
            chain_kwargs["combine_docs_chain_kwargs"] = {"prompt": prompt}

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            callback_manager=self.callback_manager,
            **chain_kwargs
        )
        logger.info(f"Setup ConversationalRetrievalChain with model: {llm_cfg['llm_model_name']} for session_id: {session_id}")

    def query(self, question, metadata_filter=None):
        """
        Send a question to the RAG chain for answer.
        Optionally filter by metadata (e.g. document id/type).
        """
        try:
            logger.info(f"Processing query: {question} (filter: {metadata_filter})")
            # If filter is provided, use retriever directly for filtered results
            if metadata_filter:
                docs = self.retriever.get_relevant_documents(question, filter=metadata_filter)
                context = "\n\n".join([doc.page_content for doc in docs])
                response = self.llm.invoke(f"{context}\n\nQuestion: {question}")
                return {
                    "answer": response,
                    "sources": [doc.metadata for doc in docs]
                }
            else:
                response = self.chain.invoke({"question": question})
                # Optionally add source metadata from retriever results
                if "source_documents" in response:
                    response["sources"] = [doc.metadata for doc in response["source_documents"]]
                return response.get("answer", response)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def get_chat_history(self):
        """
        Returns chat history as a list of dicts: [{"type": ..., "content": ...}, ...]
        """
        chat_memory = self.chain.memory
        history_msgs = chat_memory.chat_memory.messages if hasattr(chat_memory, "chat_memory") else []
        chat_history = []
        for msg in history_msgs:
            # Compatible with both AIMessage/HumanMessage and dicts
            if hasattr(msg, "type"):
                chat_history.append({"type": msg.type, "content": msg.content})
            elif isinstance(msg, dict):
                chat_history.append(msg)
        return chat_history

    def summarize_session(self, chain_type="stuff", last_n=None):
        """
        Summarizes the session's chat history using the specified chain_type and last_n.
        """
        chat_history = self.get_chat_history()
        if last_n:
            chat_history = chat_history[-last_n:]
        chat_text = "\n".join(
            f"{msg['type'].capitalize()}: {msg['content']}" for msg in chat_history
        )
        docs = [Document(page_content=chat_text)]
        if not self.llm:
            raise ValueError("LLM is not initialized in this RAGSystem instance.")
        summarize_chain = load_summarize_chain(self.llm, chain_type=chain_type)
        summary = summarize_chain.run(docs)
        return summary

# -----------------------------------------------------------------------------
# Example: Building the vectorstore from a folder with multi-doc support
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    config = load_config()
    callback_manager = get_callback_manager_from_config(config)
    folder = config.get("data", {}).get("knowledgebase_folder", "./knowledgebase")
    chunk_size = config.get("data", {}).get("chunk_size", 1000)
    # 1. Load and chunk all markdown docs
    all_chunks = load_and_chunk_markdown_docs(folder, chunk_size=chunk_size)
    # 2. Set up RAG system and embeddings
    rag = RAGSystem(config, callback_manager=callback_manager)
    rag.setup_embeddings()
    # 3. Build vectorstore from all chunks
    rag.create_vectorstore(all_chunks)
    # 4. Set up conversational chain
    rag.setup_chain()
    logger.info("RAG system initialized and ready for queries.")