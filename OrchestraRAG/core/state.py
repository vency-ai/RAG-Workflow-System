# core/state.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Annotated
from langgraph.channels.last_value import LastValue

# Define the Workflow State Type using BaseModel and Annotated for channels
class RAGLoopState(BaseModel):
    """
    Represents the state of the RAG (Retrieval Augmented Generation) workflow.
    Each field is Annotated with LastValue to ensure that only the last value
    written to the channel is retained.
    """
    questions: Annotated[List[str], LastValue]
    is_single_question: Annotated[Optional[bool], LastValue] = None
    current_question: Annotated[Optional[str], LastValue] = None
    current_sources: Annotated[Optional[List[Dict]], LastValue] = None # Assuming sources are a list of dicts from RAG API
    structured_results: Annotated[List[Dict], LastValue] = Field(default_factory=list) # e.g., [{"question": "...", "answer": "...", "sources": [...]}]
    run_evaluation: Annotated[Optional[bool], LastValue] = None
    # New field to store evaluation results for the current question
    # This will typically hold the 'evaluation' dictionary from the /evaluate response
    current_evaluation_results: Annotated[Optional[Dict], LastValue] = None
    retries_for_current_question: Annotated[int, LastValue] = 0 # Retry counter for the current question

    index: Annotated[int, LastValue] = 0 # Current index of the question being processed
    results: Annotated[List[str], LastValue] = Field(default_factory=list) # Simple answer list, for backward compatibility or basic output
    done: Annotated[bool, LastValue] = False # Flag to indicate if the current question's processing is complete
    summary: Annotated[Optional[str], LastValue] = None # Summary of the session, for multi-question flow
    node_timings: Annotated[Dict[str, float], LastValue] = Field(default_factory=dict) # To store execution time of each node
    session_id: Annotated[str, LastValue] # Unique session ID for tracking with the RAG service
    use_cache_session_level: Annotated[Optional[bool], LastValue] = None # Reflects the input from /run-sequence payload
    use_cache_for_current_question: Annotated[Optional[bool], LastValue] = None # Reflects if the *specific question* found a cache hit during check_cache
    active_chain_sessions: Annotated[Optional[int], LastValue] = None # Number of active chain sessions in the RAG service

