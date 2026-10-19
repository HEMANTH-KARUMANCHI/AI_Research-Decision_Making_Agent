import streamlit as st
import time
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from agent import build_graph
from config.settings import MAX_ITERATIONS

# 1. Page Configuration
st.set_page_config(
    page_title="AI Research & Decision-Making Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Custom Styling
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E88E5;
    }
    
    /* Header Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-langgraph { background-color: #E3F2FD; color: #1565C0; }
    .badge-tools { background-color: #E8F5E9; color: #2E7D32; }
    .badge-agent { background-color: #F3E5F5; color: #7B1FA2; }

    /* Custom Container Boxes */
    .result-card {
        background-color: #FAFAFA;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
    }

    /* Streamlit Button enhancement */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration & Architecture Inspector
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=110)
    st.title("Control Panel")
    
    st.markdown("### ⚙️ System Settings")
    st.info(f"**Max Iteration Limit:** `{MAX_ITERATIONS}` steps\n\n**Loop Type:** Stateful ReAct Loop (LangGraph)")
    
    st.divider()
    st.markdown("### 🧰 Available Tools")
    st.markdown("""
    * 🔍 **Web Search (`web_search`)**  
      Live internet data retrieval via DuckDuckGo.
    * 🧮 **Calculator (`calculator`)**  
      Deterministic arithmetic & financial precision.
    * 📊 **Text Analysis (`text_analysis`)**  
      Linguistic stats, reading time & key skills.
    """)
    
    st.divider()
    st.markdown("### 🔄 Graph Execution Flow")
    st.code("""
[START]
   ↓
[Agent Node] ◄────┐
   │ (Router)      │
   ├── Needs Tool ─┘
   │
   └── Done / Limit
         ↓
       [END]
    """, language="text")

# 4. Main Hero Section
st.title("🧠 AI Research & Decision-Making Agent")
st.markdown("""
<div>
    <span class="badge badge-langgraph">⚡ LangGraph v0.2</span>
    <span class="badge badge-agent">🤖 Autonomous ReAct Loop</span>
    <span class="badge badge-tools">🛠️ 3 Native Tools</span>
</div>
""", unsafe_allow_html=True)
st.caption("Autonomously plans actions, retrieves web research, calculates numbers, and synthesizes decisions.")

st.divider()

# 5. Quick Prompt Chips & Interactive Input
st.markdown("##### 💡 Select a Quick-Test Scenario or Type Your Own:")

col_q1, col_q2, col_q3, col_q4 = st.columns(4)

# Session state initialization for user prompt
if "input_query" not in st.session_state:
    st.session_state.input_query = ""

with col_q1:
    if st.button("⚖️ Compare GPUs & Cost", use_container_width=True):
        st.session_state.input_query = (
            "Search for the current pricing of NVIDIA H100 and AMD MI300X GPUs. "
            "Then calculate the cost difference between buying 8 NVIDIA H100s vs 8 AMD MI300Xs."
        )

with col_q2:
    if st.button("💰 Monthly SIP Math", use_container_width=True):
        st.session_state.input_query = (
            "If an investor saves ₹25,000 every month for 7 years, what is their total invested principal? "
            "Also compute what 10% annual bonus on that principal would equal."
        )

with col_q3:
    if st.button("📑 Job Skill Analysis", use_container_width=True):
        st.session_state.input_query = (
            "Analyze this job description for key metrics and skills: 'We are hiring a Lead AI Engineer "
            "skilled in Python, LangGraph, LangChain, PyTorch, Docker, Kubernetes, AWS, and RAG architectures.'"
        )

with col_q4:
    if st.button("🗄️ Postgres vs MongoDB", use_container_width=True):
        st.session_state.input_query = (
            "What are the major architectural differences between PostgreSQL and MongoDB, and when should an engineering team pick each?"
        )

user_task = st.text_area(
    "Enter your research task or decision challenge:",
    value=st.session_state.input_query,
    placeholder="e.g., Research AWS vs GCP GPU compute costs, calculate savings for a 12-month reserved instance, and give a recommendation...",
    height=110
)

col_run, col_clear, _ = st.columns([1.5, 1, 4])
with col_run:
    run_button = st.button("🚀 Run Agent", type="primary", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.input_query = ""
        st.rerun()

# 6. Graph Execution & Live Observability
if run_button and user_task.strip():
    start_time = time.time()
    
    st.divider()
    
    # Modern st.status container for live agent activity trace
    with st.status("🤖 Agent Reasoning & Tool Execution in Progress...", expanded=True) as status_box:
        try:
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

            final_answer_text = ""
            total_tools_called = 0
            iterations_completed = 0
            
            # Stream the execution step-by-step
            for step_event in graph.stream(initial_state):
                for node_name, state_update in step_event.items():
                    if node_name == "agent":
                        new_messages = state_update.get("messages", [])
                        current_iter = state_update.get("iterations", 1)
                        iterations_completed = current_iter
                        
                        if new_messages:
                            latest_msg = new_messages[-1]
                            if isinstance(latest_msg, AIMessage) and latest_msg.tool_calls:
                                for tc in latest_msg.tool_calls:
                                    total_tools_called += 1
                                    tool_name = tc.get("name", "tool")
                                    args = tc.get("args", {})
                                    st.write(f"🤔 **Iteration {current_iter} (Agent Decision):** Needs tool `{tool_name}` with parameters:")
                                    st.json(args, expanded=False)
                            else:
                                st.write(f"✍️ **Iteration {current_iter} (Agent Decision):** Gathered sufficient context. Synthesizing final decision.")
                                final_answer_text = latest_msg.content

                    elif node_name == "tools":
                        tool_messages = state_update.get("messages", [])
                        for tm in tool_messages:
                            t_name = getattr(tm, "name", "tool")
                            content_str = str(tm.content)
                            with st.expander(f"📥 Tool Output: `{t_name}`", expanded=False):
                                st.write(content_str)

            elapsed_time = round(time.time() - start_time, 2)
            
            # Collapse the trace cleanly once complete
            status_box.update(
                label=f"✅ Reasoning Complete in {iterations_completed} Iteration(s) ({elapsed_time}s)",
                state="complete",
                expanded=False
            )

            # 7. Metrics Summary Dashboard
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Iterations", f"{iterations_completed} / {MAX_ITERATIONS}")
            with m2:
                st.metric("Tools Executed", f"{total_tools_called}")
            with m3:
                st.metric("Execution Latency", f"{elapsed_time}s")
            with m4:
                st.metric("Graph Status", "Completed (END)")

            # 8. Render Structured Recommendation & Final Answer
            st.markdown("### 🎯 Final Recommendation & Insights")
            
            # Normalize list content if returned from model
            if isinstance(final_answer_text, list):
                final_answer_text = "\n".join(
                    [item.get("text", "") for item in final_answer_text if isinstance(item, dict) and "text" in item]
                )
            
            st.markdown(final_answer_text)

            # 9. Download Report Feature
            st.download_button(
                label="📥 Download Research Report (.md)",
                data=f"# AI Research & Decision Report\n\n**Task:** {user_task.strip()}\n\n---\n\n{final_answer_text}",
                file_name="ai_research_decision_report.md",
                mime="text/markdown"
            )

        except Exception as e:
            status_box.update(label="❌ Agent Execution Error", state="error", expanded=True)
            st.error(f"Error encountered: {str(e)}")

elif run_button:
    st.warning("⚠️ Please provide a research question or click one of the quick scenarios above.")
