# api/main.py
from fastapi import FastAPI, Header
from fastapi.responses import PlainTextResponse
import logging
import time

from api.models import QuestionList
from core.state import RAGLoopState
from core.graph_builder import compiled_graph, graph_dot
from core.utils import emojify

# --- Logging Configuration ---
# Configure logging for the FastAPI application.
# Note: BasicConfig only sets up the root logger, you might want more granular control
# in a larger application.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- FastAPI App Instance ---
app = FastAPI(
    title="RAG Graph API Server",
    description="A FastAPI service for orchestrating RAG workflows using LangGraph, with retry and evaluation capabilities.",
    version="1.0.0"
)

# --- FastAPI Endpoint to Run the Sequence ---
@app.post("/run-sequence", summary="Run a RAG sequence for one or more questions")
async def run_sequence(
    payload: QuestionList,
    session_id: str = Header(..., alias="X-Session-Id", description="Unique session ID for conversation tracking and caching.")
):
    """
    Executes the RAG workflow for a list of questions.
    Uses the X-Session-Id header to maintain session context with the RAG service.
    
    The workflow includes:
    - Determining if it's a single or multi-question session.
    - Clearing session history for multi-question sessions.
    - Fetching questions, calling the RAG service, and optionally checking/using cache.
    - Retrying RAG calls for single questions if evaluation indicates low relevance.
    - Summarizing the session for multi-question flows.
    - Triggering evaluation (single or batch) based on input.
    """
    if not payload.questions:
        logger.warning(emojify("Empty question list received", "??"))
        return {"error": "No questions provided."}

    # Initialize the state for the LangGraph workflow with data from the API request.
    state_initial = RAGLoopState(
        questions=payload.questions,
        session_id=session_id,
        use_cache_session_level=payload.use_cache,
        run_evaluation=payload.eval
    )

    logger.info(emojify(f"Starting pipeline with {len(payload.questions)} questions for session {session_id}", "??"))

    pipeline_start = time.time()
    try:
        # Invoke the compiled LangGraph pipeline with the initial state.
        # `compiled_graph` is imported from core.graph_builder.
        final_state_data = compiled_graph.invoke(state_initial)
    except Exception as e:
        logger.exception(f"[CRITICAL] LangGraph pipeline failed for session {session_id}")
        return PlainTextResponse(f"Internal Server Error: {str(e)}", status_code=500)

    pipeline_end = time.time()
    total_time = round(pipeline_end - pipeline_start, 4)
    
    # Extract relevant results from the final state of the pipeline.
    # Note: `final_state_data` is a dict-like object from LangGraph.
    answers = final_state_data.get("results", [])
    structured_answers = final_state_data.get("structured_results", [])
    summary = final_state_data.get("summary")
    node_timings = final_state_data.get("node_timings", {})
    active_chain_sessions = final_state_data.get("active_chain_sessions")
    evaluation_results = final_state_data.get("current_evaluation_results") # For single-question flow

    logger.info(emojify(f"Pipeline completed in {total_time}s for session {session_id}", "?"))

    # Return the comprehensive results of the pipeline execution.
    return {
        "answers": answers,
        "structured_answers": structured_answers,
        "summary": summary,
        "node_timings": node_timings,
        "pipeline_time": total_time,
        "active_chain_sessions": active_chain_sessions,
        "evaluation_results": evaluation_results
    }


# --- GET Endpoint to Visualize the Graph ---
@app.get("/graphviz", response_class=PlainTextResponse, summary="Get Graphviz DOT source for LangGraph workflow")
async def get_graphviz():
    """
    Returns the DOT source of the LangGraph, which can be used to visualize the workflow.
    Paste this output into a Graphviz viewer (e.g., graphviz.anvard.org) to see the flow.
    """
    return graph_dot.source
