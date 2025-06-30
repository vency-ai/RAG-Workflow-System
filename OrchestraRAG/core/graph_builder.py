# core/graph_builder.py
from langgraph.graph import StateGraph, END, START
from graphviz import Digraph # For visualization

from core.state import RAGLoopState
from core.nodes import (
    check_question_count, clear_session_history, fetch_next_question,
    check_cache, call_rag_api, summarize_session, advance_to_next_question,
    eval_answer_single, eval_answer_multi,
    route_after_check, route_after_cache_check, route_after_rag_multi, route_after_single_eval
)
from core.config_manager import MIN_RELEVANCE_SCORE # For graphviz label


def build_rag_graph():
    """
    Constructs and compiles the LangGraph StateGraph for the RAG workflow.
    Defines all nodes, edges, and conditional routing.
    """
    builder = StateGraph(RAGLoopState)

    # Add common nodes first that are used by both single and multi-question flows.
    builder.add_node("check_question_count", check_question_count)
    builder.add_node("clear_session_history", clear_session_history)

    # Nodes specifically for the Single-Question Path
    builder.add_node("fetch_next_question_single", fetch_next_question)
    builder.add_node("check_cache_single", check_cache)
    builder.add_node("call_rag_api_single", call_rag_api)
    builder.add_node("eval_answer_single", eval_answer_single)

    # Nodes specifically for the Multi-Question Path
    builder.add_node("fetch_next_question_multi", fetch_next_question)
    builder.add_node("call_rag_api_multi", call_rag_api)
    builder.add_node("advance_to_next_question", advance_to_next_question)
    builder.add_node("summarize_session_multi", summarize_session)
    builder.add_node("eval_answer_multi", eval_answer_multi)

    # Define the flow (edges)
    builder.add_edge(START, "check_question_count")

    # Route based on single/multi question count at the very beginning of the graph.
    builder.add_conditional_edges(
        "check_question_count",
        route_after_check,
        {"multi": "clear_session_history", "single": "fetch_next_question_single"}
    )

    # --- Multi-Question Flow Edges ---
    builder.add_edge("clear_session_history", "fetch_next_question_multi")
    builder.add_edge("fetch_next_question_multi", "call_rag_api_multi")
    # Insert `advance_to_next_question` before the routing decision in the multi-question loop.
    builder.add_edge("call_rag_api_multi", "advance_to_next_question")
    builder.add_conditional_edges(
        "advance_to_next_question", # The router now comes after the index is incremented.
        route_after_rag_multi,
        {"more": "fetch_next_question_multi", "summarize": "summarize_session_multi"}
    )
    builder.add_edge("summarize_session_multi", "eval_answer_multi")
    builder.add_edge("eval_answer_multi", END) # The multi-question flow ends after batch evaluation.

    # --- Single-Question Flow Edges ---
    builder.add_edge("fetch_next_question_single", "check_cache_single")
    builder.add_conditional_edges(
        "check_cache_single",
        route_after_cache_check,
        {"use_cached": "call_rag_api_single", "no_cache": "call_rag_api_single"}
    )

    builder.add_edge("call_rag_api_single", "eval_answer_single")
    # Conditional edge after `eval_answer_single` to handle retry logic.
    builder.add_conditional_edges(
        "eval_answer_single",
        route_after_single_eval,
        {"retry_rag": "call_rag_api_single", "end_pipeline": END}
    )

    # Compile the graph into a runnable LangGraph application.
    graph = builder.compile()
    return graph, builder # Return both compiled graph and builder for visualization


def generate_graph_visualization(builder_instance: StateGraph) -> Digraph:
    """
    Generates a Graphviz Digraph object representing the LangGraph workflow.
    """
    dot = Digraph(comment='LangGraph Flow with Distinct Paths')

    # Node declarations corresponding to the graph builder nodes for visualization.
    dot.node('START', 'START')
    dot.node('CQC', 'check_question_count')
    dot.node('CSH', 'clear_session_history')

    # Single-question path nodes (suffix _S for clarity in the graph).
    dot.node('FNQ_S', 'fetch_next_question_single')
    dot.node('CHK_S', 'check_cache_single')
    dot.node('CRA_S', 'call_rag_api_single')
    dot.node('EVAL_S', 'eval_answer_single')

    # Multi-question path nodes (suffix _M for clarity in the graph).
    dot.node('FNQ_M', 'fetch_next_question_multi')
    dot.node('CRA_M', 'call_rag_api_multi')
    dot.node('ATNQ_M', 'advance_to_next_question')
    dot.node('SUM_M', 'summarize_session_multi')
    dot.node('EVAL_M', 'eval_answer_multi')

    dot.node('END', 'END')

    # Edges for the graph visualization, mirroring the LangGraph builder.
    dot.edge('START', 'CQC')
    dot.edge('CQC', 'CSH', label='multi')
    dot.edge('CQC', 'FNQ_S', label='single')

    # Multi-Question Path Edges
    dot.edge('CSH', 'FNQ_M')
    dot.edge('FNQ_M', 'CRA_M')
    dot.edge('CRA_M', 'ATNQ_M')
    dot.edge('ATNQ_M', 'FNQ_M', label='more')
    dot.edge('ATNQ_M', 'SUM_M', label='summarize')
    dot.edge('SUM_M', 'EVAL_M', label='then eval')
    dot.edge('EVAL_M', 'END', label='done')

    # Single-Question Path Edges
    dot.edge('FNQ_S', 'CHK_S')
    dot.edge('CHK_S', 'CRA_S', label='cache decision leads to RAG API')
    dot.edge('CRA_S', 'EVAL_S', label='eval after RAG call')
    dot.edge('EVAL_S', 'CRA_S', label=f'relevance < {MIN_RELEVANCE_SCORE}, retry')
    dot.edge('EVAL_S', 'END', label=f'relevance >= {MIN_RELEVANCE_SCORE} or retries exhausted')

    return dot

# Compile the graph and generate visualization when this module is imported
compiled_graph, graph_builder_instance = build_rag_graph()
graph_dot = generate_graph_visualization(graph_builder_instance)

# Save the DOT file and render it to a PNG image.
graph_dot.save('langgraph_flow_with_retries.dot')
graph_dot.render('langgraph_flow_with_retries', format='png', cleanup=True)
