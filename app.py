import streamlit as st
import sys
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from agent import build_graph

# Configure page
st.set_page_config(
    page_title="AI Research & Decision-Making Agent",
    page_icon="🧠",
    layout="wide"
)

# Custom header
st.title("🧠 AI Research & Decision-Making Agent")
st.caption("Powered by LangGraph, Dynamic Tool Calling & Structured Decision-Making")

# Sidebar with system info
with st.sidebar:
    st.header("⚙️ Agent Architecture")
    st.markdown("""
    **Core Capabilities:**
    - 🔍 **Web Search Tool**: Real-time facts, tech specs & current information.
    - 🧮 **Calculator Tool**: Deterministic arithmetic and financial evaluations.
    - 📊 **Text Analysis Tool**: Linguistic stats, word density, and keywords.
    
    **Workflow Graph:**
    ```text
    [START] 
       ↓
    [Agent Node] ──► (Router)
       ↑               │
    [Tools Node] ◄─────┤ (Needs Tools)
                       ↓ (Finished)
                     [END]
    ```
    """)
    st.divider()
    st.markdown("💡 **Tip:** The agent autonomously determines which tools to call, executes them iteratively, and stops when enough evidence is collected.")

# User prompt input
st.subheader("Enter your research or decision-making task:")

example_prompts = [
    "Compare NVIDIA H100 and AMD MI300X GPUs and calculate the cost difference for 8 units.",
    "If I invest ₹20,000 per month for 5 years, how much total principal will I contribute?",
    "Analyze this job description: 'Seeking a Senior AI Engineer skilled in Python, LangGraph, LangChain, Docker, Kubernetes, and AWS.'",
    "What are the major architecture differences between PostgreSQL and MongoDB?"
]

selected_example = st.selectbox(
    "Or select a sample prompt to test:",
    options=["-- Choose an example or type below --"] + example_prompts
)

default_query = selected_example if selected_example != "-- Choose an example or type below --" else ""
user_task = st.text_area(
    "Research Task / Prompt:",
    value=default_query,
    placeholder="e.g., Compare two cloud providers and calculate which is cheaper for 10 servers...",
    height=100
)

col1, col2 = st.columns([1, 5])
with col1:
    run_button = st.button("🚀 Run Agent", type="primary", use_container_width=True)

if run_button and user_task.strip():
    st.divider()
    
    # Activity trace container
    activity_container = st.container()
    
    with activity_container:
        st.subheader("📋 Agent Activity Trace")
        trace_placeholder = st.empty()
        
    with st.spinner("Agent is reasoning and executing tools..."):
        try:
            # Build the compiled LangGraph workflow
            graph = build_graph()
            
            initial_state = {
                "messages": [
                    SystemMessage(
                        content="You are an AI Research & Decision-Making Agent. Decide which tools are needed, execute them, and synthesize a clear final recommendation."
                    ),
                    HumanMessage(content=user_task.strip())
                ],
                "iterations": 0
            }

            trace_steps = []
            trace_steps.append("🏁 **Task Initiated:** Graph started with user prompt.")
            trace_placeholder.markdown("\n\n".join(trace_steps))

            final_answer_text = ""
            
            # Stream graph execution step-by-step
            for step_event in graph.stream(initial_state):
                for node_name, state_update in step_event.items():
                    if node_name == "agent":
                        new_messages = state_update.get("messages", [])
                        current_iter = state_update.get("iterations", 1)
                        if new_messages:
                            latest_msg = new_messages[-1]
                            if isinstance(latest_msg, AIMessage) and latest_msg.tool_calls:
                                for tc in latest_msg.tool_calls:
                                    tool_name = tc.get("name", "tool")
                                    args = tc.get("args", {})
                                    trace_steps.append(
                                        f"🤔 **Agent Decision (Iter {current_iter}):** Needs `{tool_name}` with input `{args}`"
                                    )
                            else:
                                trace_steps.append(f"✍️ **Agent Decision (Iter {current_iter}):** Sufficient information collected. Synthesizing final answer.")
                                final_answer_text = latest_msg.content
                    elif node_name == "tools":
                        tool_messages = state_update.get("messages", [])
                        for tm in tool_messages:
                            t_name = getattr(tm, "name", "tool")
                            content_preview = str(tm.content)[:250] + ("..." if len(str(tm.content)) > 250 else "")
                            trace_steps.append(f"✅ **Tool Executed (`{t_name}`):** Result: _{content_preview}_")

                    trace_placeholder.markdown("\n\n".join(trace_steps))

            trace_steps.append("🏁 **Workflow Finished:** Graph reached `END`.")
            trace_placeholder.markdown("\n\n".join(trace_steps))

            # Render Final Output
            st.divider()
            st.subheader("🎯 Final Answer & Recommendation")
            
            # Normalize content format
            if isinstance(final_answer_text, list):
                final_answer_text = "\n".join(
                    [item.get("text", "") for item in final_answer_text if isinstance(item, dict) and "text" in item]
                )
            
            st.markdown(final_answer_text)

        except Exception as e:
            st.error(f"❌ Execution encountered an error: {str(e)}")
elif run_button:
    st.warning("Please enter a research task or question first.")
