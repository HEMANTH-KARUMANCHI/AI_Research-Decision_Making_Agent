import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import END
from agent.graph import build_graph
from agent.router import should_continue
from config.settings import MAX_ITERATIONS

def test_graph_compilation():
    """Verify that the LangGraph compiles without configuration or cycle errors."""
    graph = build_graph()
    assert graph is not None

def test_router_with_tool_call():
    """Verify that should_continue routes to 'tools' when an AIMessage contains tool_calls."""
    mock_ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}, "id": "call_1"}]
    )
    state = {
        "messages": [HumanMessage(content="What is 2+2?"), mock_ai_message],
        "iterations": 1
    }
    decision = should_continue(state)
    assert decision == "tools"

def test_router_without_tool_call():
    """Verify that should_continue routes to END when the agent has formulated a final response."""
    mock_ai_message = AIMessage(content="2 + 2 is 4.")
    state = {
        "messages": [HumanMessage(content="What is 2+2?"), mock_ai_message],
        "iterations": 1
    }
    decision = should_continue(state)
    assert decision == END

def test_router_max_iterations_reached():
    """Verify that should_continue forcibly terminates to END when MAX_ITERATIONS is reached."""
    mock_ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}, "id": "call_1"}]
    )
    state = {
        "messages": [mock_ai_message],
        "iterations": MAX_ITERATIONS  # Hit limit
    }
    decision = should_continue(state)
    assert decision == END
