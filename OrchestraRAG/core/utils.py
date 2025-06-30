# core/utils.py
import time
import logging
from typing import Callable, Any
from core.state import RAGLoopState
from core.config_manager import USE_EMOJIS

logger = logging.getLogger(__name__)

def emojify(msg: str, emoji: str) -> str:
    """Prepends an emoji to a message if USE_EMOJIS is true."""
    return f"{emoji} {msg}" if USE_EMOJIS else msg

def log_and_time_node(name: str) -> Callable[[Callable[[RAGLoopState], RAGLoopState]], Callable[[RAGLoopState], RAGLoopState]]:
    """
    A decorator that logs entry/exit and measures the execution time of a LangGraph node.
    It also includes a warning if the 'questions' list is modified or reassigned,
    which can indicate unintended side effects.
    """
    def wrapper(fn: Callable[[RAGLoopState], RAGLoopState]) -> Callable[[RAGLoopState], RAGLoopState]:
        def inner(state: RAGLoopState) -> RAGLoopState:
            logger.debug(f"[ENTER] Node: {name}")
            start = time.time()

            # Capture initial state of questions for modification detection
            original_questions_id = id(state.questions)
            original_questions_content_copy = list(state.questions) # Create a shallow copy for content comparison

            try:
                new_state = fn(state) # Execute the actual node function
            except Exception as e:
                logger.exception(f"[ERROR] Exception in node '{name}': {e}")
                raise # Re-raise the exception to propagate it up the graph

            end = time.time()
            duration = round(end - start, 4)

            # --- Questions Modification Detection ---
            if id(new_state.questions) != original_questions_id:
                logger.warning(
                    f"!!! WARNING: Node '{name}' REASSIGNED 'questions' attribute! "
                    f"Original ID: {original_questions_id}, New ID: {id(new_state.questions)}"
                )
            elif new_state.questions != original_questions_content_copy:
                logger.warning(
                    f"!!! WARNING: Node '{name}' MODIFIED CONTENT of 'questions' list IN-PLACE! "
                    f"Original: {original_questions_content_copy}, New: {new_state.questions}"
                )
            # ----------------------------------------

            # Store the duration in the state's node_timings dictionary
            new_state.node_timings[name] = duration
            logger.debug(f"[EXIT] Node: {name} | Duration: {duration}s")
            return new_state
        return inner
    return wrapper
