from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Central state passed across all nodes in the LangGraph workflow."""
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
