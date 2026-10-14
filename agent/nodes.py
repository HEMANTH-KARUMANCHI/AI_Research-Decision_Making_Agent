import time
from langchain_core.messages import SystemMessage
from config.settings import MAX_ITERATIONS, get_llm
from agent.state import AgentState
from tools import web_search, calculator, text_analysis

ALL_TOOLS = [web_search, calculator, text_analysis]

def invoke_with_retry(llm_runnable, messages, max_retries=3):
    """Invokes LLM with retry logic for temporary capacity/503 spikes."""
    for attempt in range(1, max_retries + 1):
        try:
            return llm_runnable.invoke(messages)
        except Exception as e:
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < max_retries:
                wait_time = attempt * 2
                time.sleep(wait_time)
            else:
                raise e

def agent_node(state: AgentState) -> dict:
    """Agent node that reads state, reasons over history, and selects tools or produces the final answer."""
    current_iterations = state.get("iterations", 0) + 1
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    messages = state["messages"]

    system_guidelines = (
        "You are an AI Research & Decision-Making Agent.\n"
        "Guidelines:\n"
        "1. Understand the user's objective and decide if tools are needed.\n"
        "2. Use 'web_search' for real-time information, product specs, and pricing.\n"
        "3. Use 'calculator' for exact numerical and arithmetic calculations.\n"
        "4. Use 'text_analysis' for text statistics, word counts, and keyword analysis.\n"
        "5. If a tool returns an error, explain the limitation clearly.\n"
        "6. If you have gathered enough information, synthesize a structured recommendation and final answer."
    )

    if current_iterations >= MAX_ITERATIONS:
        override_msg = SystemMessage(
            content="IMPORTANT: You have reached the maximum allowed tool call limit. Do NOT call any more tools. Summarize your findings based on the existing conversation history."
        )
        response = invoke_with_retry(llm, messages + [override_msg])
    else:
        response = invoke_with_retry(llm_with_tools, messages + [SystemMessage(content=system_guidelines)])

    return {
        "messages": [response],
        "iterations": current_iterations
    }
