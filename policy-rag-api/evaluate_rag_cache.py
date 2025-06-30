import os
import datetime
import logging
import re
import tomli  # Use tomli or tomllib for TOML parsing
import langchain_ollama

from policy_rag_chain import load_config

# --- Utility to load TOML config ---
def load_toml_config(path="config/config.toml"):
    with open(path, "rb") as f:
        return tomli.load(f)

# Load config and evaluation section (from TOML)
config = load_toml_config()
logger = logging.getLogger("evaluate_rag_cache")

# --- Evaluation config section (TOML: use dict notation or .get) ---
eval_config = config.get("evaluation", {})
EVAL_PROMPTS = eval_config.get("prompts", {
    "fluency": (
        "Evaluate the following answer for fluency (grammar, clarity, and readability). "
        "Rate from 1 (very poor) to 5 (excellent). "
        "Reply with only a single integer (1-5), no explanation or extra text.\n"
        "Answer:\n{answer}\n"
        "Score:"
    ),
    "faithfulness": (
        "Evaluate the following answer for faithfulness to the provided sources. "
        "Rate from 1 (completely unfaithful or hallucinated) to 5 (fully faithful and supported by the sources). "
        "Reply with only a single integer (1-5), no explanation or extra text.\n"
        "Answer:\n{answer}\n"
        "Sources:\n{sources}\n"
        "Score:"
    ),
    "relevance": (
        "Evaluate the following answer for relevance to the given question and sources. "
        "Rate from 1 (irrelevant) to 5 (highly relevant and on-topic). "
        "Reply with only a single integer (1-5), no explanation or extra text.\n"
        "Question:\n{question}\n"
        "Answer:\n{answer}\n"
        "Sources:\n{sources}\n"
        "Score:"
    ),
    "conciseness": (
        "Evaluate the following answer for conciseness (succinctness, no unnecessary information). "
        "Rate from 1 (verbose or contains irrelevant details) to 5 (very concise and to the point). "
        "Reply with only a single integer (1-5), no explanation or extra text.\n"
        "Answer:\n{answer}\n"
        "Score:"
    )
})
if "prompts" in eval_config:
    # TOML: evaluation.prompts is a subtable
    EVAL_PROMPTS = eval_config["prompts"]

EVAL_MODEL = eval_config.get("model", config.get("llm", {}).get("llm_model_name", "phi4"))
EVAL_OLLAMA_URL = eval_config.get("ollama_base_url", config.get("llm", {}).get("ollama_base_url", "http://localhost:11434"))
CACHE_DIR = eval_config.get("cache_dir", config.get("cache", {}).get("cache_dir", "cache_data"))
CACHE_FILE = os.path.join(CACHE_DIR, eval_config.get("cache_file", config.get("cache", {}).get("cache_file", "response_cache.jsonl")))

def is_evaluated(entry):
    """Check if entry has a complete evaluation section."""
    ev = entry.get("evaluation")
    if not ev:
        return False
    fields = ["fluency", "faithfulness", "relevance", "conciseness"]
    return all(ev.get(f) is not None for f in fields)

def get_llm():
    """
    Returns a ChatOllama instance for evaluation.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        logger.error("langchain_ollama is required for evaluation. Please install it.")
        raise
    try:
        llm = ChatOllama(model=EVAL_MODEL, base_url=EVAL_OLLAMA_URL)
        logger.info(f"Initialized ChatOllama for model '{EVAL_MODEL}' at '{EVAL_OLLAMA_URL}'")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize ChatOllama: {e}")
        raise

def score_with_llm(llm, prompt):
    """Send the prompt to Ollama and extract an integer 1-5 from the response."""
    try:
        output = llm.invoke(prompt)
        if hasattr(output, "content"):
            output_text = output.content
        else:
            output_text = str(output)
        logger.info(f"LLM reply: {repr(output_text)}")
        match = re.search(r"\b([1-5])\b", output_text)
        if match:
            return int(match.group(1))
        else:
            logger.warning("LLM output did not contain an integer score 1-5.")
            return None
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        return None

def read_cache():
    """Reads all cache entries from the cache file."""
    if not os.path.isfile(CACHE_FILE):
        return []
    import json
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_cache(entries):
    """Overwrites the cache file with the given entries."""
    import json
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

def evaluate_cache_entries():
    """
    Evaluates all cached Q&A pairs, appending evaluation results.
    Skips already evaluated entries.
    Returns a summary dict with answer included in each detail.
    """
    logger.info("Starting evaluation of cache entries.")
    try:
        entries = read_cache()
    except Exception as e:
        logger.error(f"Failed to read cache: {e}")
        return {"error": f"Failed to read cache: {str(e)}"}

    if not entries:
        logger.warning("No cached entries to evaluate.")
        return {"message": "No cached entries to evaluate."}

    llm = None
    try:
        llm = get_llm()
    except Exception as e:
        return {"error": f"Failed to initialize LLM: {str(e)}"}

    updated_entries = []
    total = len(entries)
    evaluated = 0
    skipped = 0
    errors = 0
    detail_results = []

    for entry in entries:
        try:
            resp = entry.get("response", {}).get("result", {})
            answer = resp.get("answer", "")
            question = resp.get("question", "")
            sources = "\n".join([src.get("text", "") for src in resp.get("sources", []) if src.get("text")])

            if is_evaluated(entry):
                logger.info(f"Entry already evaluated (key: {entry.get('key')}). Skipping.")
                skipped += 1
                updated_entries.append(entry)
                ev = entry.get("evaluation", {})
                detail_results.append({
                    "key": entry.get('key'),
                    "answer": answer,
                    "fluency": ev.get("fluency"),
                    "faithfulness": ev.get("faithfulness"),
                    "relevance": ev.get("relevance"),
                    "conciseness": ev.get("conciseness"),
                })
                continue

            if not (answer and question):
                logger.warning(f"Missing answer or question in entry (key: {entry.get('key')}). Skipping.")
                skipped += 1
                updated_entries.append(entry)
                continue

            fluency_q = EVAL_PROMPTS["fluency"].format(answer=answer)
            faithfulness_q = EVAL_PROMPTS["faithfulness"].format(answer=answer, sources=sources)
            relevance_q = EVAL_PROMPTS["relevance"].format(question=question, answer=answer, sources=sources)
            conciseness_q = EVAL_PROMPTS["conciseness"].format(answer=answer)

            fluency = score_with_llm(llm, fluency_q)
            faithfulness = score_with_llm(llm, faithfulness_q)
            relevance = score_with_llm(llm, relevance_q)
            conciseness = score_with_llm(llm, conciseness_q)

            if any(val is None for val in [fluency, faithfulness, relevance, conciseness]):
                logger.warning(
                    f"Some evaluation scores are None for entry (key: {entry.get('key')})."
                )
                errors += 1

            entry["evaluation"] = {
                "fluency": fluency,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "conciseness": conciseness,
                "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            updated_entries.append(entry)
            evaluated += 1

            detail_results.append({
                "key": entry.get('key'),
                "answer": answer,
                "fluency": fluency,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "conciseness": conciseness,
            })
            logger.info(
                f"Evaluated entry key={entry.get('key')}: "
                f"fluency={fluency}, faithfulness={faithfulness}, relevance={relevance}, conciseness={conciseness}"
            )

        except Exception as e:
            logger.error(f"Error evaluating entry key={entry.get('key')}: {e}")
            updated_entries.append(entry)
            errors += 1

    try:
        write_cache(updated_entries)
        logger.info(f"Evaluation complete. Updated {evaluated} entries, skipped {skipped}, errors: {errors}")
    except Exception as e:
        logger.error(f"Failed to write updated cache: {e}")
        return {"error": f"Failed to write updated cache: {str(e)}"}

    return {
        "message": "Evaluation completed.",
        "total_entries": total,
        "evaluated": evaluated,
        "skipped": skipped,
        "errors": errors,
        "details": detail_results[:25],  # show up to 25 results for preview
    }


def clear_all_evaluations():
    """
    Removes the 'evaluation' field from all cache entries in-place.
    Returns a summary dict.
    """
    logger.info("Clearing all evaluation fields from cache entries.")
    try:
        entries = read_cache()
    except Exception as e:
        logger.error(f"Failed to read cache: {e}")
        return {"error": f"Failed to read cache: {str(e)}"}

    if not entries:
        logger.info("No cache entries found to clear.")
        return {"message": "No cache entries found."}

    changed = 0
    errors = 0
    updated_entries = []

    for entry in entries:
        try:
            if "evaluation" in entry:
                del entry["evaluation"]
                changed += 1
            updated_entries.append(entry)
        except Exception as e:
            logger.error(f"Failed to clear evaluation for entry key={entry.get('key')}: {e}")
            errors += 1
            updated_entries.append(entry)

    try:
        write_cache(updated_entries)
        logger.info(f"Cleared evaluations from {changed} entries. Errors: {errors}")
    except Exception as e:
        logger.error(f"Failed to write updated cache: {e}")
        return {"error": f"Failed to write updated cache: {str(e)}"}

    return {
        "message": "Evaluations cleared.",
        "total_entries": len(entries),
        "cleared": changed,
        "errors": errors,
    }


def evaluate_single_response(answer, question, sources):
    """
    Evaluates a single Q&A for fluency, faithfulness, relevance, conciseness.
    Returns an evaluation dict and the LLM responses.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        logger.error("langchain_ollama is required for evaluation. Please install it.")
        return None

    if isinstance(sources, list):
        sources_text = "\n".join([src.get("text", "") for src in sources if src.get("text")])
    else:
        sources_text = sources or ""

    llm = ChatOllama(model=EVAL_MODEL, base_url=EVAL_OLLAMA_URL)
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