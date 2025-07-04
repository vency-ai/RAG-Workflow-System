import os
import json
import time
import uuid
import shutil
import logging
import datetime
import re
from fastapi import FastAPI, Request, UploadFile, File, Header, Body
from pydantic import BaseModel
from typing import Optional

from rag_lc_pto_policy_util_v2 import (
    DocumentProcessor,
    RAGSystem,
    load_config,
    validate_redis_connection,
    get_callback_manager_from_config
)

from evaluate_rag_cache import evaluate_cache_entries, clear_all_evaluations
from evaluate_rag_cache import EVAL_PROMPTS, EVAL_MODEL, EVAL_OLLAMA_URL
import datetime



#from evaluate_rag_cache import evaluate_cache_entries, clear_all_evaluations
try:
    import tiktoken
    _has_tiktoken = True
except ImportError:
    _has_tiktoken = False

# --- Logging setup ---
config = load_config()
log_config = config.get("logging", {})
LOGS_DIR = log_config.get("logs_dir", "logs")
LOG_FILE = log_config.get("log_file", "rag_pto_langserv_mem.log")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, LOG_FILE), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_pto_langserv_mem")

app = FastAPI(title="Retrieval LangServ App (with Memory & File Cache, Multi-Session)")

# --- In-memory RAG system store (simulates cache) ---
rag_system_store = {}

# --- File Cache Settings ---
CACHE_DIR = config.get("cache", {}).get("cache_dir", os.path.join(os.getcwd(), "cache_data"))
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, config.get("cache", {}).get("cache_file", "response_cache.jsonl"))
BACKUP_CACHE_FILE = os.path.join(CACHE_DIR, "backup_response_cache.jsonl")
CACHE_TTL = config.get("cache", {}).get("file_cache_ttl", 604800)  # Default 1 week

def get_cache_key(question, model):
    norm_question = (question or "").strip().lower()
    norm_model = (model or '').strip().lower()
    return f"{norm_model}::{norm_question}"

def read_cache():
    """Reads all cache entries from the cache file."""
    if not os.path.isfile(CACHE_FILE):
        return []
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_cache(entries):
    """Overwrites the cache file with the given entries."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

def backup_cache_entry(entry):
    """Appends an entry to backup_response_cache.jsonl for historical tracking."""
    with open(BACKUP_CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"Backed up overwritten cache entry for key: {entry.get('key')} (session: {entry.get('session_id')})")

def add_cache_entry(key, response, session_id, input_data, used_model):
    """
    Adds a new cache entry. If an entry with the same key exists, backs it up before overwriting.
    Saves detailed info for evaluation and auditing.
    """
    entries = read_cache()
    now = int(time.time())
    dt = datetime.datetime.utcnow()
    date = dt.strftime("%Y-%m-%d")
    tstr = dt.strftime("%H:%M:%S")
    request_id = str(uuid.uuid4())

    # Backup and remove any previous entry with the same key
    new_entries = []
    for e in entries:
        if e.get("key") == key:
            backup_cache_entry(e)
        else:
            new_entries.append(e)

    cache_entry = {
        "key": key,
        "session_id": session_id,
        "date": date,
        "time": tstr,
        "ts": now,
        "request_id": request_id,
        "input_data": input_data,
        "used_model": used_model,
        "response": response
    }
    new_entries.append(cache_entry)
    write_cache(new_entries)
    logger.info(f"Cached response for key: {key} (session: {session_id}, date: {date}, time: {tstr}, req_id: {request_id})")

def get_cached_entry(key):
    """
    Returns the cached response for the given key if not expired, else None.
    """
    now = int(time.time())
    entries = read_cache()
    for entry in entries:
        if entry.get("key") == key:
            ts = entry.get("ts", 0)
            if now - ts <= CACHE_TTL:
                logger.info(f"Cache hit for key: {key}")
                return entry["response"]
            else:
                logger.info(f"Cache expired for key: {key}")
    return None

def clear_cache():
    """Deletes the main cache file."""
    if os.path.isfile(CACHE_FILE):
        os.remove(CACHE_FILE)
        logger.info("Cache file cleared.")

def estimate_tokens(text, encoding_name="cl100k_base"):
    """Roughly estimates token count, with tiktoken if available."""
    if not text:
        logger.debug("estimate_tokens called with empty text.")
        return 0
    if _has_tiktoken:
        try:
            enc = tiktoken.get_encoding(encoding_name)
            tokens = len(enc.encode(text))
            logger.debug(f"tiktoken estimated {tokens} tokens for text: {repr(text)[:100]}")
            return tokens
        except Exception as e:
            logger.warning(f"tiktoken failed to encode text: {e}")
    tokens = max(1, len(text) // 4)
    logger.debug(f"Fallback estimated {tokens} tokens for text: {repr(text)[:100]}")
    return tokens

def get_chain(session_id: str = None, override_model: str = None):
    """
    Loads or creates a RAGSystem conversational chain for the session/model.
    """
    config = load_config()
    data = config["data"]
    embedding = config["embedding"]
    llm = config["llm"]
    memory = config.get("memory", {})
    
    kb_dir = data.get("kb_dir", "")
    markdown_file = data["markdown_file"]
    markdown_path = os.path.join(kb_dir, markdown_file) if kb_dir else markdown_file
    split_chunk_size = data.get("split_chunk_size", 1000)
    if not os.path.isfile(markdown_path):
        logger.error(f"Markdown file not found: {markdown_path}")
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    # Use provided session_id, or generate a new one for in-memory session
    if not session_id:
        session_id = f"inmem-{uuid.uuid4()}"
        logger.info(f"No session_id provided; using generated in-memory session_id: {session_id}")

    cache_key = f"{session_id}::{override_model or llm.get('llm_model_name')}"
    if cache_key in rag_system_store:
        return rag_system_store[cache_key].chain, override_model or llm.get('llm_model_name'), rag_system_store[cache_key]

    content = DocumentProcessor.load_markdown_with_tables(markdown_path)
    documents = DocumentProcessor.split_markdown(content, chunk_size=split_chunk_size)
    config_for_rag = dict(config)
    if override_model:
        config_for_rag["llm"] = dict(config["llm"])
        config_for_rag["llm"]["llm_model_name"] = override_model
        logger.info(f"Overriding LLM model for session {session_id}: {override_model}")

    callback_manager = get_callback_manager_from_config(config)

    rag = RAGSystem(config_for_rag, session_id=session_id, callback_manager=callback_manager)
    rag.setup_embeddings()
    rag.create_vectorstore(documents)
    rag.setup_chain()
    rag_system_store[cache_key] = rag
    return rag.chain, config_for_rag["llm"]["llm_model_name"], rag

# --- Helper: Evaluate a Single Response with Comments ---
def evaluate_single_response(answer, question, sources):
    """
    Evaluates a single Q&A for fluency, faithfulness, relevance, conciseness.
    Returns an evaluation dict and the LLM responses.
    """
    # Import eval LLM here to avoid circular import issues
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        logger.error("langchain_ollama is required for evaluation. Please install it.")
        return None

    # Prepare sources as text for prompt input
    if isinstance(sources, list):
        sources_text = "\n".join([src.get("text", "") for src in sources if src.get("text")])
    else:
        sources_text = sources or ""

    llm = ChatOllama(model=EVAL_MODEL, base_url=EVAL_OLLAMA_URL)

    # Prepare prompts for each metric using the shared config
    fluency_q = EVAL_PROMPTS["fluency"].format(answer=answer)
    faithfulness_q = EVAL_PROMPTS["faithfulness"].format(answer=answer, sources=sources_text)
    relevance_q = EVAL_PROMPTS["relevance"].format(question=question, answer=answer, sources=sources_text)
    conciseness_q = EVAL_PROMPTS["conciseness"].format(answer=answer)

    # Internal scoring function
    def score(prompt):
        try:
            output = llm.invoke(prompt)
            output_text = output.content if hasattr(output, "content") else str(output)
            match = re.search(r"\b([1-5])\b", output_text)
            return int(match.group(1)) if match else None
        except Exception as e:
            logger.error(f"Error evaluating prompt: {e}")
            return None

    # Call LLM for each metric
    fluency = score(fluency_q)
    faithfulness = score(faithfulness_q)
    relevance = score(relevance_q)
    conciseness = score(conciseness_q)

    evaluation = {
        "fluency": fluency,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "conciseness": conciseness,
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    return evaluation

# --- Add this helper function for single Answer Evaluation ---
def evaluate_single_response1(answer, question, sources):
    """
    Evaluates a single Q&A for fluency, faithfulness, relevance, conciseness.
    Returns an evaluation dict and the LLM responses.
    """
    # Import the eval LLM here to avoid circular import
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        logger.error("langchain_ollama is required for evaluation. Please install it.")
        return None

    # Prepare sources as text if not already
    if isinstance(sources, list):
        sources_text = "\n".join([src.get("text", "") for src in sources if src.get("text")])
    else:
        sources_text = sources or ""

    llm = ChatOllama(model=EVAL_MODEL, base_url=EVAL_OLLAMA_URL)
    # Prepare prompts for each metric
    fluency_q = EVAL_PROMPTS["fluency"].format(answer=answer)
    faithfulness_q = EVAL_PROMPTS["faithfulness"].format(answer=answer, sources=sources_text)
    relevance_q = EVAL_PROMPTS["relevance"].format(question=question, answer=answer, sources=sources_text)
    conciseness_q = EVAL_PROMPTS["conciseness"].format(answer=answer)

    def score(prompt):
        try:
            output = llm.invoke(prompt)
            output_text = output.content if hasattr(output, "content") else str(output)
            match = re.search(r"\b([1-5])\b", output_text)
            return int(match.group(1)) if match else None
        except Exception as e:
            logger.error(f"Error evaluating prompt: {e}")
            return None

    # Run LLM evaluations for each metric
    fluency = score(fluency_q)
    faithfulness = score(faithfulness_q)
    relevance = score(relevance_q)
    conciseness = score(conciseness_q)

    evaluation = {
        "fluency": fluency,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "conciseness": conciseness,
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    return evaluation
####
@app.post("/invoke")
async def invoke(request: Request):
    """
    Main API endpoint to process user questions.
    Supports "eval": true in input for per-Answer Evaluation.
    Compares evaluation scores of cache and fresh results (if both evaluated), 
    and keeps the one with better relevance.
    """
    import re  # Ensure regex is available for evaluation score extraction

    # 1. Session handling
    session_id = request.headers.get('X-Session-Id')
    if not session_id:
        session_id = f"inmem-{uuid.uuid4()}"
        logger.info(f"No session_id header found; generated new in-memory session_id: {session_id}")
    else:
        logger.info(f"Received session_id from header: {session_id}")

    # 2. Parse and validate request body
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Invalid JSON in request: {e}")
        return {"error": "Invalid JSON payload."}

    input_data = body.get("input", {})
    question = input_data.get("question")
    model_override = input_data.get("model")
    use_cache = input_data.get("use_cache", False)
    eval_now = input_data.get("eval", False)

    if not question or not isinstance(question, str):
        logger.error("Missing or invalid 'question' in request input")
        return {"error": "Missing or invalid 'question' in request input."}

    cache_key = get_cache_key(question, model_override or config['llm'].get("llm_model_name"))

    # 3. Attempt to fetch from cache if requested
    cached_entry = get_cached_entry(cache_key) if use_cache else None
    cached_result = cached_entry.get("result") if cached_entry else None
    cached_eval = cached_entry.get("evaluation") if cached_entry else None

    # Helper for answer and evaluation generation
    def generate_answer_and_eval():
        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        chain, used_model, rag_obj = get_chain(session_id, override_model=model_override)
        logger.info(f"Using LLM model '{used_model}' for session '{session_id}'")
        try:
            docs = chain.retriever.get_relevant_documents(question)
        except Exception as e:
            logger.error(f"Retriever failed: {e}")
            return None, None, None
        retrieval_time = time.perf_counter() - retrieval_start
        sources = []
        for doc in docs:
            entry = {
                "text": doc.page_content,
                "metadata": getattr(doc, "metadata", {}),
            }
            if hasattr(doc, "distance"):
                entry["distance"] = doc.distance
            sources.append(entry)
        llm_start = time.perf_counter()
        try:
            response = chain.invoke({"question": question})
        except Exception as e:
            logger.error(f"LLM chain failed: {e}")
            return None, None, None
        llm_time = time.perf_counter() - llm_start
        total_time = time.perf_counter() - total_start
        chat_history = []
        try:
            mem = chain.memory
            if hasattr(mem, "chat_memory") and hasattr(mem.chat_memory, "messages"):
                chat_history = [
                    {"type": m.type, "content": m.content}
                    for m in mem.chat_memory.messages
                ]
            elif hasattr(mem, "buffer") and isinstance(mem.buffer, list):
                chat_history = [
                    {"type": m.type, "content": m.content}
                    for m in mem.buffer
                ]
        except Exception as e:
            logger.warning(f"Unable to extract chat history: {e}")
        token_summary = {}
        note = ""
        question_text = question or ""
        context_texts = [doc["text"] for doc in sources]
        answer_text = response.get("answer", response)
        def count_tokens(text):
            if hasattr(chain, "llm") and hasattr(chain.llm, "get_num_tokens"):
                try:
                    tokens = chain.llm.get_num_tokens(text)
                    logger.debug(f"chain.llm.get_num_tokens() returned {tokens} for text: {repr(text)[:100]}")
                    return tokens
                except Exception as e:
                    logger.warning(f"chain.llm.get_num_tokens failed: {e}")
            tokens = estimate_tokens(text)
            logger.debug(f"Estimated tokens for text: {repr(text)[:100]} = {tokens}")
            return tokens
        try:
            logger.debug(f"Calculating query_tokens for: {repr(question_text)[:100]}")
            token_summary["query_tokens"] = count_tokens(question_text)
            logger.debug(f"Calculating context_tokens for {len(context_texts)} contexts")
            token_summary["context_tokens"] = sum(count_tokens(text) for text in context_texts)
            logger.debug(f"Calculating response_tokens for: {repr(answer_text)[:100]}")
            token_summary["response_tokens"] = count_tokens(answer_text)
            if hasattr(chain, "llm") and hasattr(chain.llm, "get_num_tokens"):
                note = "Token counts use chain.llm.get_num_tokens when available, otherwise estimated using tiktoken or fallback."
            elif _has_tiktoken:
                note = "Token counts are estimated using tiktoken's cl100k_base encoding."
            else:
                note = "Token counts are rough estimates (tiktoken not installed)."
            logger.debug(f"Token summary: {token_summary}")
        except Exception as e:
            note = f"Token count could not be computed: {e}"
            logger.warning(note)
            token_summary = {}
        result = {
            "question": question,
            "chat_history": chat_history,
            "model": used_model,
            "answer": response.get("answer", response),
            "sources": sources,
            "time_summary": {
                "retrieval_time": retrieval_time,
                "llm_time": llm_time,
                "total_time": total_time
            },
            "token_summary": token_summary,
            "note": note
        }
        eval_result = None
        if eval_now:
            eval_result = evaluate_single_response(
                answer=result["answer"],
                question=question,
                sources=sources
            )
        return result, eval_result, used_model

    # 4. If eval is requested: always generate a fresh answer + eval
    if eval_now:
        fresh_result, fresh_eval, used_model = generate_answer_and_eval()
        if fresh_result is None:
            return {"error": "Failed to generate fresh answer."}
        # 5. Compare cache and fresh by relevance (if both evaluated)
        answer_source = None
        decision = None
        result_to_use = None
        eval_to_use = None
        if cached_result and cached_eval and isinstance(cached_eval.get("relevance"), int):
            if fresh_eval and isinstance(fresh_eval.get("relevance"), int):
                if fresh_eval["relevance"] > cached_eval["relevance"]:
                    # Use and cache fresh
                    result_to_use = fresh_result
                    eval_to_use = fresh_eval
                    answer_source = "fresh"
                    decision = f"Fresh answer used because its relevance score ({fresh_eval['relevance']}) > cached ({cached_eval['relevance']})"
                    add_cache_entry(
                        key=cache_key,
                        response={"result": fresh_result, "evaluation": fresh_eval},
                        session_id=session_id,
                        input_data=input_data,
                        used_model=used_model
                    )
                else:
                    # Use cached
                    result_to_use = cached_result
                    eval_to_use = cached_eval
                    answer_source = "cache"
                    decision = f"Cached answer used because its relevance score ({cached_eval['relevance']}) >= fresh ({fresh_eval['relevance']})"
            else:
                # Fresh eval failed, fallback to cached
                result_to_use = cached_result
                eval_to_use = cached_eval
                answer_source = "cache"
                decision = "Cached answer used (fresh evaluation not available)."
        else:
            # No valid cache or cache lacks evaluation: use and cache fresh
            result_to_use = fresh_result
            eval_to_use = fresh_eval
            answer_source = "fresh"
            decision = "No valid cached evaluation found; using fresh answer."
            add_cache_entry(
                key=cache_key,
                response={"result": fresh_result, "evaluation": fresh_eval},
                session_id=session_id,
                input_data=input_data,
                used_model=used_model
            )
        return {
            "result": result_to_use,
            "evaluation": eval_to_use,
            "answer_source": answer_source,
            "decision": decision
        }

    # 6. If eval is NOT requested, use standard cache or fresh answer logic
    if cached_result:
        return {"result": cached_result}

    # 7. Generate a fresh answer (no evaluation)
    fresh_result, _, used_model = generate_answer_and_eval()
    if fresh_result is None:
        return {"error": "Failed to generate fresh answer."}
    add_cache_entry(
        key=cache_key,
        response={"result": fresh_result},
        session_id=session_id,
        input_data=input_data,
        used_model=used_model
    )
    return {"result": fresh_result}
###
@app.post("/invoke2")
async def invoke2(request: Request):
    """
    Main API endpoint to process user questions.
    Now supports "eval": true in the input to evaluate the answer on-the-fly
    and include the evaluation in both the response and the cache.
    """
    session_id = request.headers.get('X-Session-Id')
    if session_id:
        logger.info(f"Received session_id from header: {session_id}")
    else:
        session_id = f"inmem-{uuid.uuid4()}"
        logger.info(f"No session_id header found; generated new in-memory session_id: {session_id}")

    # Parse and validate request body
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Invalid JSON in request: {e}")
        return {"error": "Invalid JSON payload."}

    input_data = body.get("input", {})
    question = input_data.get("question")
    model_override = input_data.get("model")
    use_cache = input_data.get("use_cache", False)
    eval_now = input_data.get("eval", False)  # <-- New flag for on-demand evaluation

    if not question or not isinstance(question, str):
        logger.error("Missing or invalid 'question' in request input")
        return {"error": "Missing or invalid 'question' in request input."}

    cache_key = get_cache_key(question, model_override or config['llm'].get("llm_model_name"))
    cached_entry = None
    result = None
    evaluation = None

    # --- CACHE LOGIC ---
    if use_cache:
        cached_response = get_cached_entry(cache_key)
        if cached_response:
            result = cached_response.get("result", cached_response)
            # If eval requested and cached evaluation exists, return it
            evaluation = cached_response.get("evaluation")
            if eval_now and not evaluation:
                # Run eval (not present in cache), will update cache below
                answer = result.get("answer")
                sources = result.get("sources")
                evaluation = evaluate_single_response(answer, question, sources)
                # Save eval to cache (update the cache entry with evaluation)
                # First, read all entries and update the relevant one
                entries = read_cache()
                for entry in entries:
                    if entry.get("key") == cache_key:
                        entry["evaluation"] = evaluation
                        break
                write_cache(entries)
            # Return with or without eval as appropriate
            response_payload = {"result": result}
            if eval_now:
                response_payload["evaluation"] = evaluation
            return response_payload

    # --- GENERATE FRESH ANSWER ---
    total_start = time.perf_counter()
    retrieval_start = time.perf_counter()

    chain, used_model, rag_obj = get_chain(session_id, override_model=model_override)
    logger.info(f"Using LLM model '{used_model}' for session '{session_id}'")

    # Retrieve context/docs
    try:
        docs = chain.retriever.get_relevant_documents(question)
    except Exception as e:
        logger.error(f"Retriever failed: {e}")
        return {"error": f"Retriever failed: {str(e)}"}

    retrieval_time = time.perf_counter() - retrieval_start

    sources = []
    for doc in docs:
        entry = {
            "text": doc.page_content,
            "metadata": getattr(doc, "metadata", {}),
        }
        if hasattr(doc, "distance"):
            entry["distance"] = doc.distance
        sources.append(entry)

    llm_start = time.perf_counter()
    try:
        response = chain.invoke({"question": question})
    except Exception as e:
        logger.error(f"LLM chain failed: {e}")
        return {"error": f"LLM chain failed: {str(e)}"}
    llm_time = time.perf_counter() - llm_start
    total_time = time.perf_counter() - total_start

    chat_history = []
    try:
        mem = chain.memory
        if hasattr(mem, "chat_memory") and hasattr(mem.chat_memory, "messages"):
            chat_history = [
                {"type": m.type, "content": m.content}
                for m in mem.chat_memory.messages
            ]
        elif hasattr(mem, "buffer") and isinstance(mem.buffer, list):
            chat_history = [
                {"type": m.type, "content": m.content}
                for m in mem.buffer
            ]
    except Exception as e:
        logger.warning(f"Unable to extract chat history: {e}")

    # --- Hybrid Token Counting with Debug Logging and Notes ---
    token_summary = {}
    note = ""
    question_text = question or ""
    context_texts = [doc["text"] for doc in sources]
    answer_text = response.get("answer", response)

    def count_tokens(text):
        if hasattr(chain, "llm") and hasattr(chain.llm, "get_num_tokens"):
            try:
                tokens = chain.llm.get_num_tokens(text)
                logger.debug(f"chain.llm.get_num_tokens() returned {tokens} for text: {repr(text)[:100]}")
                return tokens
            except Exception as e:
                logger.warning(f"chain.llm.get_num_tokens failed: {e}")
        tokens = estimate_tokens(text)
        logger.debug(f"Estimated tokens for text: {repr(text)[:100]} = {tokens}")
        return tokens

    try:
        logger.debug(f"Calculating query_tokens for: {repr(question_text)[:100]}")
        token_summary["query_tokens"] = count_tokens(question_text)
        logger.debug(f"Calculating context_tokens for {len(context_texts)} contexts")
        token_summary["context_tokens"] = sum(count_tokens(text) for text in context_texts)
        logger.debug(f"Calculating response_tokens for: {repr(answer_text)[:100]}")
        token_summary["response_tokens"] = count_tokens(answer_text)

        if hasattr(chain, "llm") and hasattr(chain.llm, "get_num_tokens"):
            note = "Token counts use chain.llm.get_num_tokens when available, otherwise estimated using tiktoken or fallback."
            logger.info("Token summary calculated using chain.llm.get_num_tokens or fallback.")
        elif _has_tiktoken:
            note = "Token counts are estimated using tiktoken's cl100k_base encoding."
            logger.info("Token summary calculated using tiktoken.")
        else:
            note = "Token counts are rough estimates (tiktoken not installed)."
            logger.info("Token summary calculated using character-based estimation.")
        logger.debug(f"Token summary: {token_summary}")
    except Exception as e:
        note = f"Token count could not be computed: {e}"
        logger.warning(note)
        token_summary = {}

    # Construct the response result
    result = {
        "question": question,
        "chat_history": chat_history,
        "model": used_model,
        "answer": response.get("answer", response),
        "sources": sources,
        "time_summary": {
            "retrieval_time": retrieval_time,
            "llm_time": llm_time,
            "total_time": total_time
        },
        "token_summary": token_summary,
        "note": note
    }

    # --- On-demand evaluation if requested ---
    evaluation = None
    if eval_now:
        evaluation = evaluate_single_response(
            answer=result["answer"],
            question=question,
            sources=sources
        )

    # Prepare the response payload for the user
    return_payload = {
        "result": result
    }
    if eval_now:
        return_payload["evaluation"] = evaluation

    # Save to cache, with backup if overwriting
    cache_entry_data = {
        "result": result
    }
    if evaluation:
        cache_entry_data["evaluation"] = evaluation

    add_cache_entry(
        key=cache_key,
        response=cache_entry_data,
        session_id=session_id,
        input_data=input_data,
        used_model=used_model
    )

    logger.info(f"Question: {question}")
    logger.info(f"Answer: {result['answer']}")
    logger.info(f"Model: {used_model}")
    logger.info(f"Retrieval time: {retrieval_time:.3f}s, LLM time: {llm_time:.3f}s, Total: {total_time:.3f}s")
    logger.info(f"Sources count: {len(sources)}")
    logger.info(f"Token summary: {token_summary}, Note: {note}")
    if evaluation:
        logger.info(f"Evaluation: {evaluation}")

    return return_payload

@app.post("/invoke1")
async def invoke1(request: Request):
    """
    Main API endpoint to process user questions.
    Tracks session, question time, input, model used for evaluation and auditing.
    Uses/updates cache and backups as per requirements.
    """
    session_id = request.headers.get('X-Session-Id')
    if session_id:
        logger.info(f"Received session_id from header: {session_id}")
    else:
        session_id = f"inmem-{uuid.uuid4()}"
        logger.info(f"No session_id header found; generated new in-memory session_id: {session_id}")

    # Parse and validate request body
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Invalid JSON in request: {e}")
        return {"error": "Invalid JSON payload."}

    input_data = body.get("input", {})
    question = input_data.get("question")
    model_override = input_data.get("model")
    use_cache = input_data.get("use_cache", False)

    if not question or not isinstance(question, str):
        logger.error("Missing or invalid 'question' in request input")
        return {"error": "Missing or invalid 'question' in request input."}

    cache_key = get_cache_key(question, model_override or config['llm'].get("llm_model_name"))

    # Serve from cache only if requested
    if use_cache:
        cached_response = get_cached_entry(cache_key)
        if cached_response:
            logger.info("Returning cached response.")
            return cached_response  # already contains the 'result' key as per design

    total_start = time.perf_counter()
    retrieval_start = time.perf_counter()

    chain, used_model, rag_obj = get_chain(session_id, override_model=model_override)
    logger.info(f"Using LLM model '{used_model}' for session '{session_id}'")

    # Retrieve context/docs
    try:
        docs = chain.retriever.get_relevant_documents(question)
    except Exception as e:
        logger.error(f"Retriever failed: {e}")
        return {"error": f"Retriever failed: {str(e)}"}

    retrieval_time = time.perf_counter() - retrieval_start

    sources = []
    for doc in docs:
        entry = {
            "text": doc.page_content,
            "metadata": getattr(doc, "metadata", {}),
        }
        if hasattr(doc, "distance"):
            entry["distance"] = doc.distance
        sources.append(entry)

    llm_start = time.perf_counter()
    try:
        response = chain.invoke({"question": question})
    except Exception as e:
        logger.error(f"LLM chain failed: {e}")
        return {"error": f"LLM chain failed: {str(e)}"}
    llm_time = time.perf_counter() - llm_start
    total_time = time.perf_counter() - total_start

    chat_history = []
    try:
        mem = chain.memory
        if hasattr(mem, "chat_memory") and hasattr(mem.chat_memory, "messages"):
            chat_history = [
                {"type": m.type, "content": m.content}
                for m in mem.chat_memory.messages
            ]
        elif hasattr(mem, "buffer") and isinstance(mem.buffer, list):
            chat_history = [
                {"type": m.type, "content": m.content}
                for m in mem.buffer
            ]
    except Exception as e:
        logger.warning(f"Unable to extract chat history: {e}")

    # --- Hybrid Token Counting with Debug Logging and Notes ---
    token_summary = {}
    note = ""
    question_text = question or ""
    context_texts = [doc["text"] for doc in sources]
    answer_text = response.get("answer", response)

    def count_tokens(text):
        if hasattr(chain, "llm") and hasattr(chain.llm, "get_num_tokens"):
            try:
                tokens = chain.llm.get_num_tokens(text)
                logger.debug(f"chain.llm.get_num_tokens() returned {tokens} for text: {repr(text)[:100]}")
                return tokens
            except Exception as e:
                logger.warning(f"chain.llm.get_num_tokens failed: {e}")
        tokens = estimate_tokens(text)
        logger.debug(f"Estimated tokens for text: {repr(text)[:100]} = {tokens}")
        return tokens

    try:
        logger.debug(f"Calculating query_tokens for: {repr(question_text)[:100]}")
        token_summary["query_tokens"] = count_tokens(question_text)
        logger.debug(f"Calculating context_tokens for {len(context_texts)} contexts")
        token_summary["context_tokens"] = sum(count_tokens(text) for text in context_texts)
        logger.debug(f"Calculating response_tokens for: {repr(answer_text)[:100]}")
        token_summary["response_tokens"] = count_tokens(answer_text)

        if hasattr(chain, "llm") and hasattr(chain.llm, "get_num_tokens"):
            note = "Token counts use chain.llm.get_num_tokens when available, otherwise estimated using tiktoken or fallback."
            logger.info("Token summary calculated using chain.llm.get_num_tokens or fallback.")
        elif _has_tiktoken:
            note = "Token counts are estimated using tiktoken's cl100k_base encoding."
            logger.info("Token summary calculated using tiktoken.")
        else:
            note = "Token counts are rough estimates (tiktoken not installed)."
            logger.info("Token summary calculated using character-based estimation.")
        logger.debug(f"Token summary: {token_summary}")
    except Exception as e:
        note = f"Token count could not be computed: {e}"
        logger.warning(note)
        token_summary = {}

    # Construct the response result
    result = {
        "question": question,
        "chat_history": chat_history,
        "model": used_model,
        "answer": response.get("answer", response),
        "sources": sources,
        "time_summary": {
            "retrieval_time": retrieval_time,
            "llm_time": llm_time,
            "total_time": total_time
        },
        "token_summary": token_summary,
        "note": note
    }
    return_payload = {
        "result": result
    }

    # Save to cache, with backup if overwriting
    add_cache_entry(
        key=cache_key,
        response=return_payload,
        session_id=session_id,
        input_data=input_data,
        used_model=used_model
    )

    logger.info(f"Question: {question}")
    logger.info(f"Answer: {result['answer']}")
    logger.info(f"Model: {used_model}")
    logger.info(f"Retrieval time: {retrieval_time:.3f}s, LLM time: {llm_time:.3f}s, Total: {total_time:.3f}s")
    logger.info(f"Sources count: {len(sources)}")
    logger.info(f"Token summary: {token_summary}, Note: {note}")

    return return_payload

@app.post("/update-pto-policy")
async def update_pto_policy(file: UploadFile = File(...)):
    """
    Upload a new PTO policy markdown file, clear previous in-memory vectorstore,
    and reload all sessions for the new policy. Also clears cache.
    """
    config = load_config()
    markdown_path = config["data"]["markdown_file"]
    # Save the new file
    with open(markdown_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    logger.info(f"Uploaded new PTO policy markdown to: {markdown_path}")

    # Clear the in-memory rag_system_store
    rag_system_store.clear()
    logger.info("Cleared in-memory RAG system store; vectorstore will be rebuilt on next request.")

    # Clear file cache on policy update
    clear_cache()

    return {"message": "PTO policy updated successfully. Vectorstore and cache will be rebuilt on next request."}

@app.post("/refresh_cache")
async def refresh_cache():
    """
    Endpoint to clear cache. (Does not clear backup file!)
    """
    clear_cache()
    return {"message": "File cache cleared successfully."}

class SummarizeSessionRequest(BaseModel):
    chain_type: Optional[str] = "stuff"
    last_n: Optional[int] = None

@app.post("/summarize_session")
async def summarize_session(
    request: Request,
    summarize_request: SummarizeSessionRequest = Body(...),
    x_session_id: Optional[str] = Header(None),
):
    """
    Summarizes the chat session for the given session_id.
    """
    session_id = x_session_id or request.headers.get('X-Session-Id')
    if not session_id:
        session_id = f"inmem-{uuid.uuid4()}"
        logger.info(f"No session_id header found; generated new in-memory session_id: {session_id}")

    _, _, rag_obj = get_chain(session_id=session_id)
    summary = rag_obj.summarize_session(
        chain_type=summarize_request.chain_type or "stuff",
        last_n=summarize_request.last_n
    )
    return {"summary": summary}

@app.post("/reload-data")
async def reload_data_endpoint():
    """
    Force reloads the PTO policy data, rebuilding the vectorstore for the default session/model.
    Returns detail logs and chunk summary.
    """
    logger.info("Manual vectorstore/data reload requested via /reload-data.")

    config = load_config()
    markdown_file = config["data"]["markdown_file"]
    split_chunk_size = config["data"].get("split_chunk_size", 1000)
    default_model = config["llm"].get("llm_model_name")
    session_id = "default-session"

    logger.info(f"Loading markdown file: {markdown_file}")
    if not os.path.isfile(markdown_file):
        logger.error(f"Markdown file not found: {markdown_file}")
        return {"error": f"Markdown file not found: {markdown_file}"}

    content = DocumentProcessor.load_markdown_with_tables(markdown_file)
    logger.info(f"Loaded markdown file, {len(content)} characters.")

    logger.info("Splitting markdown into chunks...")
    chunks = DocumentProcessor.split_markdown(content, chunk_size=split_chunk_size)
    num_chunks = len(chunks)
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    avg_chunk_size = sum(chunk_sizes) / num_chunks if num_chunks > 0 else 0
    max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
    min_chunk_size = min(chunk_sizes) if chunk_sizes else 0

    logger.info(f"Number of chunks created: {num_chunks}")
    logger.info(f"Chunk size (min/avg/max): {min_chunk_size}/{avg_chunk_size:.1f}/{max_chunk_size}")

    # Clear in-memory store for default session/model
    cache_key = f"{session_id}::{default_model}"
    if cache_key in rag_system_store:
        del rag_system_store[cache_key]
        logger.info(f"Removed RAG system for cache key: {cache_key}")

    callback_manager = get_callback_manager_from_config(config)

    rag = RAGSystem(config, session_id=session_id, callback_manager=callback_manager)
    rag.setup_embeddings()
    rag.create_vectorstore(chunks)
    rag.setup_chain()
    rag_system_store[cache_key] = rag
    logger.info("Rebuilt vectorstore and RAG system for default session/model.")

    return {
        "message": "PTO policy data reloaded and vectorstore rebuilt.",
        "session_id": session_id,
        "model": default_model,
        "number_of_chunks": num_chunks,
        "min_chunk_size": min_chunk_size,
        "avg_chunk_size": avg_chunk_size,
        "max_chunk_size": max_chunk_size,
        "log": [
            f"Loaded markdown file: {markdown_file}",
            f"Split into {num_chunks} chunks (min/avg/max: {min_chunk_size}/{avg_chunk_size:.1f}/{max_chunk_size})",
            f"Rebuilt vectorstore for session_id: {session_id} and model: {default_model}"
        ]
    }

@app.post("/clear_session_history")
async def clear_session_history(request: Request):
    """
    Clears the chat history for the current session.
    """
    session_id = request.headers.get('X-Session-Id')
    if not session_id:
        return {"error": "Missing X-Session-Id header."}
    model_name = config['llm'].get("llm_model_name")
    cache_key = f"{session_id}::{model_name}"
    rag = rag_system_store.get(cache_key)
    if not rag:
        # Optionally: create a new chain to ensure future turns are clean
        get_chain(session_id)
        rag = rag_system_store.get(cache_key)
    cleared = False
    if hasattr(rag, "chain") and hasattr(rag.chain, "memory") and hasattr(rag.chain.memory, "clear"):
        rag.chain.memory.clear()
        logger.info(f"Cleared chat history for session_id: {session_id}")
        cleared = True
    chain_count = len(rag_system_store)
    if cleared:
        return {
            "message": f"Chat history cleared for session_id: {session_id}",
            "active_chain_sessions": chain_count
        }
    else:
        return {
            "error": "Unable to clear chat history for this session.",
            "active_chain_sessions": chain_count
        }

@app.post("/eval")
async def eval_endpoint(request: Request):
    """
    API endpoint to evaluate all cached Q&A pairs and append evaluation results.
    Skips already evaluated entries. Returns a summary.
    """
    logger.info("Received request to /eval endpoint.")
    result = evaluate_cache_entries()
    if "error" in result:
        logger.error(f"Eval endpoint failed: {result['error']}")
        return {"error": result["error"]}
    return result        


@app.post("/eval_clear")
async def eval_clear_endpoint(request: Request):
    """
    API endpoint to clear all existing evaluations from cache entries.
    """
    logger.info("Received request to /eval_clear endpoint.")
    result = clear_all_evaluations()
    if "error" in result:
        logger.error(f"Eval_clear endpoint failed: {result['error']}")
        return {"error": result["error"]}
    return result    

@app.post("/will_use_cache")
async def will_use_cache(request: Request):
    """
    Endpoint to check if the given /invoke-style request would use the cache.
    Returns whether a valid cache entry exists and will be used, and the reason.
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Invalid JSON in request: {e}")
        return {"error": "Invalid JSON payload."}

    input_data = body.get("input", {})
    question = input_data.get("question")
    model_override = input_data.get("model")
    use_cache = input_data.get("use_cache", False)

    if not question or not isinstance(question, str):
        logger.error("Missing or invalid 'question' in request input")
        return {"will_use_cache": False, "reason": "Missing or invalid 'question' in request input."}

#    if not use_cache:
#        return {"will_use_cache": False, "reason": "use_cache flag not set in input."}

    # Build cache key exactly as /invoke does
    cache_key = get_cache_key(question, model_override or config['llm'].get("llm_model_name"))
    now = int(time.time())
    entries = read_cache()
    for entry in entries:
        if entry.get("key") == cache_key:
            ts = entry.get("ts", 0)
            if now - ts <= CACHE_TTL:
                return {
                    "will_use_cache": True,
                    "reason": "Valid, unexpired cache entry found and will be used.",
                    "cache_key": cache_key,
                    "cached_at": ts,
                }
            else:
                return {
                    "will_use_cache": False,
                    "reason": "Cached entry found but expired.",
                    "cache_key": cache_key,
                    "cached_at": ts,
                }
    return {
        "will_use_cache": False,
        "reason": "No cached entry found for this question/model.",
        "cache_key": cache_key
    }





