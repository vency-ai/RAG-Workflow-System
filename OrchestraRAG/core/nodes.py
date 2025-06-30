# core/nodes.py
import requests
import logging
import time
from typing import List, Dict, Optional, Literal

from core.state import RAGLoopState
from core.utils import log_and_time_node, emojify
from core.config_manager import (
    BASE_URL, MAX_RAG_RETRIES, MIN_RELEVANCE_SCORE,
    CLEAR_HISTORY_TIMEOUT, CHECK_CACHE_TIMEOUT, RAG_INVOKE_TIMEOUT,
    SUMMARIZE_TIMEOUT, EVALUATE_SINGLE_TIMEOUT, EVALUATE_MULTI_TIMEOUT
)

logger = logging.getLogger(__name__)

# --- Node Definitions ---

@log_and_time_node("check_question_count")
def check_question_count(state: RAGLoopState) -> RAGLoopState:
    """Determines if the current session involves a single question or multiple questions."""
    state.is_single_question = len(state.questions) == 1
    logger.debug(f"[CHECK] Single question? {state.is_single_question}")
    return state

@log_and_time_node("clear_session_history")
def clear_session_history(state: RAGLoopState) -> RAGLoopState:
    """Calls the RAG service to clear the session history, important for multi-turn conversations."""
    headers = {
        "Content-Type": "application/json",
        "X-Session-Id": state.session_id
    }
    try:
        logger.debug(f"[INIT] Clearing session history for: {state.session_id}")
        res = requests.post(f"{BASE_URL}/clear_session_history", headers=headers, timeout=CLEAR_HISTORY_TIMEOUT)
        if res.status_code == 200:
            data = res.json()
            logger.debug(f"[CLEAN] {data.get('message')} | Active: {data.get('active_chain_sessions')}")
            state.active_chain_sessions = data.get("active_chain_sessions")
        else:
            logger.warning(f"[WARN] Failed to clear history: {res.status_code} - {res.text}")
    except requests.exceptions.Timeout:
        logger.error(f"[ERROR] /clear_session_history timed out for session {state.session_id}")
    except Exception as e:
        logger.exception("[ERROR] Exception during clear_session_history")
    return state

@log_and_time_node("check_cache")
def check_cache(state: RAGLoopState) -> RAGLoopState:
    """Checks the RAG service cache for the current question if session-level caching is enabled."""
    if not state.current_question:
        logger.warning(f"[WARN] check_cache: current_question is None, cannot check cache.")
        state.use_cache_for_current_question = False
        return state

    if not state.use_cache_session_level:
        logger.debug(f"[CACHE-CHECK] Session-level caching disabled. Skipping /will_use_cache API call.")
        state.use_cache_for_current_question = False
        return state

    payload = {
        "input": {
            "question": state.current_question,
            "model": "llama3.2", # This could be made configurable or dynamic
            "eval": True # Assuming eval flag is consistent for cache check
        }
    }
    headers = {
        "Content-Type": "application/json",
        "X-Session-Id": state.session_id
    }
    try:
        logger.debug(f"[API-REQ] Checking cache for question: {state.current_question}")
        res = requests.post(f"{BASE_URL}/will_use_cache", json=payload, headers=headers, timeout=CHECK_CACHE_TIMEOUT)
        data = res.json()

        state.use_cache_for_current_question = data.get("will_use_cache", False)
        logger.debug(f"[CACHE-CHECK] use_cache_for_current_question={state.use_cache_for_current_question}")

    except requests.exceptions.Timeout:
        logger.error(f"[ERROR] /will_use_cache timed out for question {state.current_question}")
        state.use_cache_for_current_question = False
    except Exception as e:
        logger.exception("[ERROR] Exception during /will_use_cache check")
        state.use_cache_for_current_question = False
    return state

@log_and_time_node("fetch_next_question")
def fetch_next_question(state: RAGLoopState) -> RAGLoopState:
    """Fetches the next question from the list for processing."""
    if state.index < len(state.questions):
        # Reset retry counter when a new question is fetched, ensuring each new question starts fresh.
        state.retries_for_current_question = 0
        logger.debug(f"[RETRY] Resetting retry counter for new question at index {state.index}.")
        state.current_question = state.questions[state.index]
        logger.debug(f"[Q] Question [{state.index}]: {state.current_question}")
    else:
        logger.warning(f"[WARN] fetch_next_question: No more questions to fetch but node was called.")
        state.current_question = None
    return state

@log_and_time_node("call_rag_api")
def call_rag_api(state: RAGLoopState) -> RAGLoopState:
    """Calls the RAG service's /invoke API to get an answer and sources for the current question."""
    if not state.current_question:
        logger.error("[ERROR] call_rag_api: current_question is None. Cannot proceed.")
        error_message = "Error: No current question to process."
        state.results.append(error_message)
        state.structured_results.append({
            "question": "N/A",
            "answer": error_message,
            "sources": [],
            "model_used": "unknown"
        })
        return state

    payload = {
        "input": {
            "question": state.current_question
        }
    }

    # If this is a retry attempt, force use_cache to False to ensure a fresh lookup.
    if state.retries_for_current_question > 0:
        payload["input"]["use_cache"] = False
        logger.info(emojify(f"Retry attempt {state.retries_for_current_question}: Forcing cache OFF for RAG API.", "??"))
    elif state.use_cache_for_current_question:
        # If cache was checked and allowed at this stage, enable it in the payload.
        payload["input"]["use_cache"] = True
        
    headers = {
        "Content-Type": "application/json",
        "X-Session-Id": state.session_id
    }

    try:
        logger.debug(f"[API-REQ] Sending question to /invoke: {payload['input']['question']} (payload_use_cache: {payload['input'].get('use_cache', False)})")
        res = requests.post(f"{BASE_URL}/invoke", json=payload, headers=headers, timeout=RAG_INVOKE_TIMEOUT)
        
        answer = "No answer found"
        sources = []
        model_used = "unknown"
        
        if res.status_code == 200:
            try:
                rag_response = res.json().get("result", {})
                answer = rag_response.get("answer", "No answer found")
                sources = rag_response.get("sources", [])
                model_used = rag_response.get("model", "unknown")

                logger.debug(f"[API-RSP] Answer: {answer[:50]}... | Sources: {len(sources)} found")
            except Exception as e:
                answer = "Error parsing response from RAG API"
                logger.error(f"Error parsing RAG API response: {e}")
        else:
            answer = f"Error {res.status_code}: {res.text}"
            logger.warning(f"[WARN] RAG API call failed: {answer}")

        state.results.append(answer)
        state.current_sources = sources
        
        # Update or append structured results based on whether it's a new question or a retry.
        if state.is_single_question and state.structured_results and state.structured_results[-1].get("question") == state.current_question:
            state.structured_results[-1].update({
                "answer": answer,
                "sources": sources,
                "model_used": model_used
            })
        else:
            state.structured_results.append({
                "question": state.current_question,
                "answer": answer,
                "sources": sources,
                "model_used": model_used
            })

    except requests.exceptions.Timeout:
        logger.error(f"[ERROR] RAG API call timed out for question {state.current_question}")
        timeout_msg = f"RAG API call timed out: {state.current_question}"
        state.results.append(timeout_msg)
        state.current_sources = []
        if state.is_single_question and state.structured_results and state.structured_results[-1].get("question") == state.current_question:
            state.structured_results[-1].update({
                "answer": timeout_msg,
                "sources": [],
                "model_used": "unknown"
            })
        else:
            state.structured_results.append({
                "question": state.current_question,
                "answer": timeout_msg,
                "sources": [],
                "model_used": "unknown"
            })
    except Exception as e:
        logger.exception("[ERROR] Exception during RAG API call")
        exception_msg = f"Exception during RAG API call: {str(e)}"
        state.results.append(exception_msg)
        state.current_sources = []
        if state.is_single_question and state.structured_results and state.structured_results[-1].get("question") == state.current_question:
            state.structured_results[-1].update({
                "answer": exception_msg,
                "sources": [],
                "model_used": "unknown"
            })
        else:
            state.structured_results.append({
                "question": state.current_question,
                "answer": exception_msg,
                "sources": [],
                "model_used": "unknown"
            })

    return state

@log_and_time_node("summarize_session")
def summarize_session(state: RAGLoopState) -> RAGLoopState:
    """Calls the RAG service's /summarize_session API to get a summary of the conversation."""
    payload = {"chain_type": "stuff", "last_n": 6} # `last_n` could be made configurable too.
    headers = {
        "Content-Type": "application/json",
        "X-Session-Id": state.session_id
    }
    try:
        logger.debug("[API-REQ] Calling /summarize_session")
        res = requests.post(f"{BASE_URL}/summarize_session", json=payload, headers=headers, timeout=SUMMARIZE_TIMEOUT)
        if res.status_code == 200:
            summary = res.json().get("summary", "")
            logger.debug(f"[SUMMARY] {summary}")
            state.summary = summary
        else:
            logger.warning(f"[WARN] Summary failed: {res.status_code} - {res.text}")
            state.summary = f"Error summarizing: {res.text}"
    except requests.exceptions.Timeout:
        logger.error(f"[ERROR] /summarize_session timed out for session {state.session_id}")
        state.summary = f"Summary API call timed out."
    except Exception as e:
        logger.exception("[ERROR] Exception during summarization")
        state.summary = f"Summary exception: {str(e)}"
    return state

@log_and_time_node("advance_to_next_question")
def advance_to_next_question(state: RAGLoopState) -> RAGLoopState:
    """Increments the question index to move to the next question in a multi-question sequence."""
    state.index += 1
    logger.debug(f"[INDEX] Advanced to index: {state.index}")
    return state

@log_and_time_node("eval_answer_single")
def eval_answer_single(state: RAGLoopState) -> RAGLoopState:
    """
    Evaluates the answer for a single question using the RAG service's /evaluate API.
    Handles retry logic based on relevance score and `MAX_RAG_RETRIES`.
    """
    if not state.run_evaluation:
        logger.info(emojify("Skipping evaluation as 'eval' parameter was false.", "??"))
        state.done = True # Mark as done since no eval means no retry path
        state.index += 1 # Advance index if evaluation is skipped
        return state

    if not state.structured_results:
        logger.warning("[WARN] eval_answer_single: No structured results found for evaluation.")
        state.done = True # Mark as done if nothing to evaluate
        state.index += 1 # Advance index if no results
        return state

    last_structured_result = state.structured_results[-1]
    
    payload = {
        "question": last_structured_result.get("question"),
        "answer": last_structured_result.get("answer"),
        "sources": last_structured_result.get("sources", [])
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-Session-Id": state.session_id # Keep session ID if your /evaluate uses it for logging
    }

    relevance_score = 0 # Initialize relevance_score to a value that would trigger a retry by default if evaluation fails
    
    try:
        logger.debug("[API-REQ] Calling /evaluate for single question")
        res = requests.post(f"{BASE_URL}/evaluate", json=payload, headers=headers, timeout=EVALUATE_SINGLE_TIMEOUT)
        
        if res.status_code == 200:
            evaluation_response = res.json()
            evaluation_results = evaluation_response.get("evaluation", {})
            relevance_score = evaluation_results.get("relevance", 0) # Get relevance score from the response
            logger.debug(f"[API-RSP] Evaluation results: {evaluation_results} | Relevance: {relevance_score}")
            state.current_evaluation_results = evaluation_results # Store evaluation results in state
            
            state.structured_results[-1]["evaluation"] = evaluation_results

        else:
            logger.warning(f"[WARN] /evaluate failed: {res.status_code} - {res.text}")
            state.current_evaluation_results = {"error": f"Evaluation API failed: {res.text}"}
    except requests.exceptions.Timeout:
        logger.error(f"[ERROR] /evaluate timed out for question {state.current_question}")
        state.current_evaluation_results = {"error": "Evaluation API timed out"}
    except Exception as e:
        logger.exception("[ERROR] Exception during /evaluate call")
        state.current_evaluation_results = {"error": f"Evaluation API exception: {str(e)}"}
    
    # Decide whether to retry or finish based on MIN_RELEVANCE_SCORE and `MAX_RAG_RETRIES`.
    if relevance_score < MIN_RELEVANCE_SCORE and state.retries_for_current_question < MAX_RAG_RETRIES:
        state.retries_for_current_question += 1
        logger.info(emojify(f"Relevance {relevance_score} < {MIN_RELEVANCE_SCORE}. Retrying RAG for question. Retry count: {state.retries_for_current_question}", "??"))
        state.done = False # Keep done as False to indicate a retry is needed.
    else:
        state.done = True # Mark as done if relevance is good or retries are exhausted.
        state.index += 1 # Increment index only when truly done with this question.
        if relevance_score >= MIN_RELEVANCE_SCORE:
            logger.info(emojify(f"Relevance {relevance_score} >= {MIN_RELEVANCE_SCORE}. Evaluation complete for question.", "?"))
        else:
            logger.warning(emojify(f"Relevance {relevance_score} < {MIN_RELEVANCE_SCORE}, but max retries ({MAX_RAG_RETRIES}) reached. Finishing for this question.", "??"))

    return state

@log_and_time_node("eval_answer_multi")
def eval_answer_multi(state: RAGLoopState) -> RAGLoopState:
    """Triggers a batch evaluation for multiple questions using the /eval_all API."""
    if not state.run_evaluation:
        logger.info(emojify("Skipping evaluation as 'eval' parameter was false.", "??"))
        state.done = True # Still mark as done to finish the pipeline
        return state
    try:
        logger.debug("[API-REQ] Calling /eval_all")
        res = requests.post(f"{BASE_URL}/eval_all", headers={"X-Session-Id": state.session_id}, timeout=EVALUATE_MULTI_TIMEOUT)
        logger.debug(f"[API-RSP] Eval status: {res.status_code}")
    except requests.exceptions.Timeout:
        logger.error(f"[ERROR] /eval_all timed out for session {state.session_id}")
    except Exception as e:
        logger.exception("[ERROR] Evaluation call failed")
    state.done = True
    return state

# --- Conditional Routing Functions ---
def route_after_check(state: RAGLoopState) -> Literal["multi", "single"]:
    """Routes based on whether there's a single question or multiple questions."""
    return "multi" if len(state.questions) > 1 else "single"

def route_after_cache_check(state: RAGLoopState) -> Literal["use_cached", "no_cache"]:
    """Routes based on whether caching is enabled and if a cache hit occurred."""
    if not state.use_cache_session_level:
        return "no_cache"
    return "use_cached" if state.use_cache_for_current_question else "no_cache"

def route_after_rag_multi(state: RAGLoopState) -> Literal["more", "summarize"]:
    """Routes in the multi-question flow: either process more questions or summarize the session."""
    # This routing now occurs AFTER `advance_to_next_question` has incremented the index.
    return "more" if state.index < len(state.questions) else "summarize"

def route_after_single_eval(state: RAGLoopState) -> Literal["retry_rag", "end_pipeline"]:
    """
    Determines if a single question needs retrying (due to low relevance) or if the pipeline
    for this question should end (good relevance or max retries reached).
    """
    # The `eval_answer_single` node sets `state.done = True` if the question is fully processed
    # (either good relevance or retries exhausted). If `state.done` is True, we end the pipeline for this question.
    if state.done:
        return "end_pipeline"
    else: # `state.done` is False, meaning `eval_answer_single` has flagged a need for retry.
        return "retry_rag"
