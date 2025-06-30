import os
import json
import time
import uuid
import shutil
import logging
import datetime
import tomli

from fastapi import FastAPI, Request, UploadFile, File, Header, Body
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from policy_rag_chain import (
    load_and_chunk_markdown_docs,
    RAGSystem,
    load_config,
    validate_redis_connection,
    get_callback_manager_from_config
)
from evaluate_rag_cache import (
    evaluate_single_response,
    evaluate_cache_entries,
    clear_all_evaluations,
    EVAL_PROMPTS,
    EVAL_MODEL,
    EVAL_OLLAMA_URL
)

try:
    import tiktoken
    _has_tiktoken = True
except ImportError:
    _has_tiktoken = False


def load_toml_config(path="config/config.toml"):
    with open(path, "rb") as f:
        return tomli.load(f)

# --- Logging setup ---
config = load_toml_config()
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

app = FastAPI(title="Retrieval LangServ App (Multi-Doc, Metadata Filtering, Memory, File Cache, Multi-Session)")

# --- In-memory RAG system store ---
rag_system_store = {}

# --- File Cache Settings ---
CACHE_DIR = config.get("cache", {}).get("cache_dir", os.path.join(os.getcwd(), "cache_data"))
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, config.get("cache", {}).get("cache_file", "response_cache.jsonl"))
BACKUP_CACHE_FILE = os.path.join(CACHE_DIR, "backup_response_cache.jsonl")
CACHE_TTL = config.get("cache", {}).get("file_cache_ttl", 604800)  # Default 1 week

def ensure_serializable(obj):
    import datetime
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(x) for x in obj]
    elif hasattr(obj, "__dict__"):
        return ensure_serializable(vars(obj))
    else:
        return str(obj)

def get_cache_key(question, model):
    norm_question = (question or "").strip().lower()
    norm_model = (model or '').strip().lower()
    return f"{norm_model}::{norm_question}"

def read_cache():
    if not os.path.isfile(CACHE_FILE):
        return []
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_cache(entries):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        for entry in entries:
            entry_serializable = ensure_serializable(entry)
            f.write(json.dumps(entry_serializable) + "\n")

def backup_cache_entry(entry):
    with open(BACKUP_CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(ensure_serializable(entry)) + "\n")
    logger.info(f"Backed up overwritten cache entry for key: {entry.get('key')} (session: {entry.get('session_id')})")

def add_cache_entry(key, response, session_id, input_data, used_model):
    entries = read_cache()
    now = int(time.time())
    dt = datetime.datetime.utcnow()
    date = dt.strftime("%Y-%m-%d")
    tstr = dt.strftime("%H:%M:%S")
    request_id = str(uuid.uuid4())

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

def get_cached_entry_full(key):
    now = int(time.time())
    entries = read_cache()
    for entry in entries:
        if entry.get("key") == key:
            ts = entry.get("ts", 0)
            if now - int(ts) <= CACHE_TTL:
            #if now - ts <= CACHE_TTL:
                logger.info(f"Cache hit for key: {key}")
                return entry
            else:
                logger.info(f"Cache expired for key: {key}")
    return None

def clear_cache():
    if os.path.isfile(CACHE_FILE):
        os.remove(CACHE_FILE)
        logger.info("Cache file cleared.")

def estimate_tokens(text, encoding_name="cl100k_base"):
    if not text or not isinstance(text, str):
        logger.debug("estimate_tokens called with empty or non-str text.")
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
    config = load_config()
    data = config["data"]
    kb_dir = data.get("kb_dir", "./knowledgebase")
    split_chunk_size = data.get("split_chunk_size", 1000)

    # Multi-doc: Load ALL markdown docs in the knowledge base folder
    documents = load_and_chunk_markdown_docs(kb_dir, chunk_size=split_chunk_size)

    if not session_id:
        session_id = f"inmem-{uuid.uuid4()}"
        logger.info(f"No session_id provided; using generated in-memory session_id: {session_id}")

    cache_key = f"{session_id}::{override_model or config['llm'].get('llm_model_name')}"
    if cache_key in rag_system_store:
        return rag_system_store[cache_key].chain, override_model or config['llm'].get('llm_model_name'), rag_system_store[cache_key]

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

# ----------------------------- API SCHEMAS -----------------------------

class InvokeInput(BaseModel):
    question: str = Field(..., description="The user's question.")
    model: Optional[str] = Field(None, description="Override LLM model name.")
    use_cache: Optional[bool] = Field(False, description="If true, use cache if available.")
    eval: Optional[bool] = Field(False, description="If true, evaluate with eval model.")
    metadata_filter: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata filter for retrieval, e.g. {\"type\": \"contractor_policy\"}"
    )

class InvokeRequest(BaseModel):
    input: InvokeInput

def serialize_message(m):
    # Handles both langchain.schema.* and other message types
    # You may need to adjust field names if your messages are more complex
    return {
        "type": getattr(m, "type", str(type(m))),
        "content": (
            m.content if isinstance(m.content, (str, int, float, list, dict, type(None)))
            else str(m.content)
        )
    }

# ----------------------------- API ENDPOINTS -----------------------------

@app.post("/invoke")
async def invoke(request: InvokeRequest,
                 x_session_id: Optional[str] = Header(None)
                ):
    input_data = request.input
    question = input_data.question
    model_override = input_data.model
    use_cache = input_data.use_cache
    eval_now = input_data.eval
    metadata_filter = input_data.metadata_filter

    # Use session id from header if provided, else fallback to random
    session_id = x_session_id or f"inmem-{uuid.uuid4()}"
    #session_id = None  # handled per-request in get_chain, or you can add a session header logic

    config = load_toml_config()
    cache_key = get_cache_key(question, model_override or config['llm'].get("llm_model_name"))

    cached_entry_full = None
    if use_cache or eval_now:
        cached_entry_full = get_cached_entry_full(cache_key)
    cached_entry = cached_entry_full["response"] if cached_entry_full else None
    cached_result = cached_entry.get("result") if cached_entry else None
    cached_eval = cached_entry.get("evaluation") if cached_entry else None

    logger.info(f"cache_key: {cache_key}")
    logger.info(f"cached_result: {str(cached_result)[:50]}")
    logger.info(f"cached_eval: {cached_eval}")

    if use_cache and cached_result:
        return {"result": cached_result}

    def generate_answer_and_eval():
        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        chain, used_model, rag_obj = get_chain(session_id=session_id, override_model=model_override)
        logger.info(f"Using LLM model '{used_model}' for session '{session_id}'")
        try:
            if metadata_filter:
                docs = rag_obj.retriever.get_relevant_documents(question, filter=metadata_filter)
            else:
                docs = rag_obj.retriever.get_relevant_documents(question)
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
            if metadata_filter:
                # If filtering, use LLM directly on the context, not the full chain
                context = "\n\n".join([doc.page_content for doc in docs])
                response = rag_obj.llm.invoke(f"{context}\n\nQuestion: {question}")
                answer_text = response
            else:
                response = chain.invoke({"question": question})
                answer_text = response.get("answer", response)
        except Exception as e:
            logger.error(f"LLM chain failed: {e}")
            return None, None, None
        llm_time = time.perf_counter() - llm_start
        total_time = time.perf_counter() - total_start
        chat_history = []
        try:
            mem = chain.memory
            if hasattr(mem, "chat_memory") and hasattr(mem.chat_memory, "messages"):
                chat_history = [serialize_message(m) for m in mem.chat_memory.messages]
            elif hasattr(mem, "buffer") and isinstance(mem.buffer, list):
                chat_history = [serialize_message(m) for m in mem.buffer]
        except Exception as e:
            logger.warning(f"Unable to extract chat history: {e}")
        token_summary = {}
        note = ""
        question_text = question or ""
        context_texts = [doc["text"] for doc in sources]
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
            "answer": answer_text,
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

    if eval_now and cached_result and not cached_eval:
        cached_eval = evaluate_single_response(
            answer=cached_result.get("answer"),
            question=question,
            sources=cached_result.get("sources")
        )
        cached_entry["evaluation"] = cached_eval
        cached_entry_full["response"] = cached_entry
        entries = read_cache()
        for entry in entries:
            if entry.get("key") == cache_key:
                entry["response"] = cached_entry
                break
        write_cache(entries)

    if eval_now:
        fresh_result, fresh_eval, used_model = generate_answer_and_eval()
        if fresh_result is None:
            return {"error": "Failed to generate fresh answer."}
        answer_source = None
        decision = None
        result_to_use = None
        eval_to_use = None

        if cached_result and cached_eval and isinstance(cached_eval.get("relevance"), int):
            if fresh_eval and isinstance(fresh_eval.get("relevance"), int):
                logger.info(f"Comparing relevance: cached={cached_eval['relevance']} fresh={fresh_eval['relevance']}")
                if fresh_eval["relevance"] > cached_eval["relevance"]:
                    result_to_use = fresh_result
                    eval_to_use = fresh_eval
                    answer_source = "fresh"
                    decision = f"Fresh answer used because its relevance score ({fresh_eval['relevance']}) > cached ({cached_eval['relevance']})"
                    add_cache_entry(
                        key=cache_key,
                        response={"result": fresh_result, "evaluation": fresh_eval},
                        session_id=session_id,
                        input_data=input_data.dict(),
                        used_model=used_model
                    )
                else:
                    result_to_use = cached_result
                    eval_to_use = cached_eval
                    answer_source = "cache"
                    decision = f"Cached answer used because its relevance score ({cached_eval['relevance']}) >= fresh ({fresh_eval['relevance']})"
            else:
                result_to_use = cached_result
                eval_to_use = cached_eval
                answer_source = "cache"
                decision = "Cached answer used (fresh evaluation not available)."
        else:
            result_to_use = fresh_result
            eval_to_use = fresh_eval
            answer_source = "fresh"
            decision = "No valid cached evaluation found; using fresh answer."
            add_cache_entry(
                key=cache_key,
                response={"result": fresh_result, "evaluation": fresh_eval},
                session_id=session_id,
                input_data=input_data.dict(),
                used_model=used_model
            )
        logger.info(f"Decision: {decision}")
        return {
            "result": result_to_use,
            "evaluation": eval_to_use,
            "answer_source": answer_source,
            "decision": decision
        }

    fresh_result, _, used_model = generate_answer_and_eval()
    if fresh_result is None:
        return {"error": "Failed to generate fresh answer."}
    add_cache_entry(
        key=cache_key,
        response={"result": fresh_result},
        session_id=session_id,
        input_data=input_data.dict(),
        used_model=used_model
    )
    return {"result": fresh_result}

# ------------------ File Management and Utility Endpoints ------------------

@app.get("/list-docs")
async def list_docs():
    config = load_toml_config()
    kb_dir = config["data"].get("kb_dir", "./knowledgebase")
    files = []
    for fname in os.listdir(kb_dir):
        if fname.endswith(".md"):
            fpath = os.path.join(kb_dir, fname)
            stat = os.stat(fpath)
            files.append({
                "filename": fname,
                "size": stat.st_size,
                "modified": datetime.datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z"
            })
    return {"files": files}

@app.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    config = load_toml_config()
    kb_dir = config["data"].get("kb_dir", "./knowledgebase")
    os.makedirs(kb_dir, exist_ok=True)
    target = os.path.join(kb_dir, file.filename)
    with open(target, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    logger.info(f"Uploaded new document: {target}")
    rag_system_store.clear()
    logger.info("Cleared in-memory RAG system store; vectorstore will be rebuilt on next request.")
    clear_cache()
    return {"message": f"Document {file.filename} uploaded successfully."}

@app.post("/delete-doc")
async def delete_doc(filename: str = Body(..., embed=True)):
    config = load_toml_config()
    kb_dir = config["data"].get("kb_dir", "./knowledgebase")
    target = os.path.join(kb_dir, filename)
    if not os.path.isfile(target) or not filename.endswith(".md"):
        return {"error": f"File {filename} does not exist or is not a markdown file."}
    os.remove(target)
    logger.info(f"Deleted document: {target}")
    rag_system_store.clear()
    logger.info("Cleared in-memory RAG system store; vectorstore will be rebuilt on next request.")
    clear_cache()
    return {"message": f"Document {filename} deleted successfully."}

@app.post("/reload-data")
async def reload_data_endpoint():
    logger.info("Manual vectorstore/data reload requested via /reload-data.")
    config = load_toml_config()
    kb_dir = config["data"].get("kb_dir", "./knowledgebase")
    split_chunk_size = config["data"].get("split_chunk_size", 1000)
    default_model = config["llm"].get("llm_model_name")
    session_id = "default-session"

    logger.info(f"Loading markdown files from folder: {kb_dir}")
    if not os.path.isdir(kb_dir):
        logger.error(f"Knowledge base folder not found: {kb_dir}")
        return {"error": f"Knowledge base folder not found: {kb_dir}"}

    chunks = load_and_chunk_markdown_docs(kb_dir, chunk_size=split_chunk_size)
    num_chunks = len(chunks)
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    avg_chunk_size = sum(chunk_sizes) / num_chunks if num_chunks > 0 else 0
    max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
    min_chunk_size = min(chunk_sizes) if chunk_sizes else 0

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
        "message": "Knowledge base data reloaded and vectorstore rebuilt.",
        "session_id": session_id,
        "model": default_model,
        "number_of_chunks": num_chunks,
        "min_chunk_size": min_chunk_size,
        "avg_chunk_size": avg_chunk_size,
        "max_chunk_size": max_chunk_size,
        "log": [
            f"Loaded markdown files from folder: {kb_dir}",
            f"Split into {num_chunks} chunks (min/avg/max: {min_chunk_size}/{avg_chunk_size:.1f}/{max_chunk_size})",
            f"Rebuilt vectorstore for session_id: {session_id} and model: {default_model}"
        ]
    }

@app.post("/refresh_cache")
async def refresh_cache():
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

@app.post("/clear_session_history")
async def clear_session_history(request: Request):
    config = load_toml_config()
    session_id = request.headers.get('X-Session-Id')
    if not session_id:
        return {"error": "Missing X-Session-Id header."}
    model_name = config['llm'].get("llm_model_name")
    cache_key = f"{session_id}::{model_name}"
    rag = rag_system_store.get(cache_key)
    if not rag:
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

@app.post("/eval_all")
async def eval_all_endpoint(request: Request):
    logger.info("Received request to /eval_all endpoint.")
    result = evaluate_cache_entries()
    if "error" in result:
        logger.error(f"Eval endpoint failed: {result['error']}")
        return {"error": result["error"]}
    return result

@app.post("/eval_clear")
async def eval_clear_endpoint(request: Request):
    logger.info("Received request to /eval_clear endpoint.")
    result = clear_all_evaluations()
    if "error" in result:
        logger.error(f"Eval_clear endpoint failed: {result['error']}")
        return {"error": result["error"]}
    return result

@app.post("/will_use_cache")
async def will_use_cache(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Invalid JSON in request: {e}")
        return {"error": "Invalid JSON payload."}

    input_data = body.get("input", {})
    question = input_data.get("question")
    model_override = input_data.get("model")
    use_cache = input_data.get("use_cache", False)
    config = load_toml_config()
    if not question or not isinstance(question, str):
        logger.error("Missing or invalid 'question' in request input")
        return {"will_use_cache": False, "reason": "Missing or invalid 'question' in request input."}
    cache_key = get_cache_key(question, model_override or config['llm'].get("llm_model_name"))
    now = int(time.time())
    entries = read_cache()
    for entry in entries:
        if entry.get("key") == cache_key:
            ts = entry.get("ts", 0)
            if now - int(ts) <= CACHE_TTL:
            #if now - ts <= CACHE_TTL:
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

class EvaluateRequest(BaseModel):
    question: str = Field(..., description="The question to evaluate against.")
    answer: str = Field(..., description="The answer to be evaluated.")
    sources: List[Dict[str, Any]] = Field(..., description="A list of source passages/documents.")

@app.post("/evaluate")
async def evaluate_endpoint(request: Request, body: EvaluateRequest):
    logger.info("Received /evaluate request.")
    logger.info(f"Payload: question={body.question!r}, answer={body.answer!r}, sources_count={len(body.sources)}")
    if not body.question or not isinstance(body.question, str):
        logger.error("Invalid or missing 'question' in payload.")
        return {"error": "Missing or invalid 'question' in payload."}
    if not body.answer or not isinstance(body.answer, str):
        logger.error("Invalid or missing 'answer' in payload.")
        return {"error": "Missing or invalid 'answer' in payload."}
    if not isinstance(body.sources, list):
        logger.error("Invalid 'sources' in payload. Must be a list.")
        return {"error": "Missing or invalid 'sources' in payload."}
    for idx, src in enumerate(body.sources):
        if not isinstance(src, dict):
            logger.error(f"Source at index {idx} is not a dictionary.")
            return {"error": f"Source at index {idx} is not a dictionary."}
    logger.info("All payload fields validated successfully.")
    try:
        eval_result = evaluate_single_response(
            answer=body.answer,
            question=body.question,
            sources=body.sources
        )
        logger.info(f"Evaluation result: {eval_result}")
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return {"error": f"Evaluation failed: {e}"}
    return {
        "question": body.question,
        "answer": body.answer,
        "sources": body.sources,
        "evaluation": eval_result
    }

import yaml
import re
from fastapi.responses import PlainTextResponse, JSONResponse

def extract_frontmatter(content):
    """
    Extracts YAML frontmatter from markdown content.
    Returns (metadata_dict, remainder_text)
    """
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
    if fm_match:
        fm_text = fm_match.group(1)
        remainder = fm_match.group(2)
        try:
            meta = yaml.safe_load(fm_text)
        except Exception:
            meta = {}
        return meta, remainder
    else:
        return {}, content

@app.get("/get-doc")
async def get_doc(filename: str):
    """
    Download or preview the markdown file by filename.
    """
    config = load_toml_config()
    kb_dir = config["data"].get("kb_dir", "./knowledgebase")
    target = os.path.join(kb_dir, filename)
    if not os.path.isfile(target) or not filename.endswith(".md"):
        return JSONResponse({"error": f"File {filename} not found."}, status_code=404)
    with open(target, "r", encoding="utf-8") as f:
        content = f.read()
    return PlainTextResponse(content, media_type="text/markdown")

@app.get("/doc-metadata")
async def doc_metadata(filename: str):
    """
    Get YAML frontmatter metadata for a given file.
    """
    config = load_toml_config()
    kb_dir = config["data"].get("kb_dir", "./knowledgebase")
    target = os.path.join(kb_dir, filename)
    if not os.path.isfile(target) or not filename.endswith(".md"):
        return JSONResponse({"error": f"File {filename} not found."}, status_code=404)
    with open(target, "r", encoding="utf-8") as f:
        content = f.read()
    metadata, _ = extract_frontmatter(content)
    return {"filename": filename, "metadata": metadata}

@app.get("/list-metadata")
async def list_metadata():
    """
    List all unique frontmatter metadata keys and values across all markdown docs.
    """
    config = load_toml_config()
    kb_dir = config["data"].get("kb_dir", "./knowledgebase")
    metadata_agg = {}
    for fname in os.listdir(kb_dir):
        if fname.endswith(".md"):
            fpath = os.path.join(kb_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            meta, _ = extract_frontmatter(content)
            for k, v in (meta or {}).items():
                if k not in metadata_agg:
                    metadata_agg[k] = set()
                if isinstance(v, list):
                    metadata_agg[k].update(str(x) for x in v)
                else:
                    metadata_agg[k].add(str(v))
    # Convert sets to sorted lists for JSON
    metadata_agg = {k: sorted(list(vs)) for k, vs in metadata_agg.items()}
    return {"metadata_keys": list(metadata_agg.keys()), "metadata_values": metadata_agg}    