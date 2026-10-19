import streamlit as st
import time
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from agent import build_graph
from config.settings import (
    MAX_ITERATIONS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_GEMINI_MODEL,
    OPENAI_API_KEY,
    GOOGLE_API_KEY
)

# 1. Page Configuration
st.set_page_config(
    page_title="AI Research & Decision-Making Agent | Hemanth Karumanchi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced Portfolio-Inspired Styling (Inter, Deep Slate, Neon Blue Accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Background and container refinement */
    .stApp {
        background-color: #020617;
        color: #f8fafc;
    }

    /* Top Portfolio-style Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }

    .hero-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
    }

    .brand-group {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .brand-monogram {
        width: 50px;
        height: 50px;
        border-radius: 12px;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.25rem;
        color: #ffffff;
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 4px;
    }

    /* Status indicator pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.3); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* Badges */
    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid transparent;
    }

    .badge-blue { background: rgba(37, 99, 235, 0.12); color: #60a5fa; border-color: rgba(59, 130, 246, 0.25); }
    .badge-slate { background: rgba(51, 65, 85, 0.3); color: #cbd5e1; border-color: rgba(71, 85, 105, 0.4); }
    .badge-purple { background: rgba(147, 51, 234, 0.12); color: #c084fc; border-color: rgba(168, 85, 247, 0.25); }

    /* Custom Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #60a5fa !important;
        font-weight: 800 !important;
        font-size: 1.7rem !important;
    }

    /* Buttons Modernization */
    div.stButton > button {
        background-color: #0f172a;
        color: #f1f5f9;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 0.6rem 1.1rem;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div.stButton > button:hover {
        border-color: #3b82f6;
        background-color: #1e293b;
        color: #93c5fd;
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.25);
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.55);
        transform: translateY(-2px);
    }

    /* Input Text Area */
    div[data-testid="stTextArea"] textarea {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }

    /* Final Result Container */
    .result-container {
        background: linear-gradient(180deg, #0f172a 0%, #0b1329 100%);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 26px;
        margin-top: 18px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    /* Footer styling */
    .footer-bar {
        border-top: 1px solid #1e293b;
        padding: 24px 0 12px 0;
        margin-top: 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        color: #64748b;
        font-size: 0.85rem;
    }
    .footer-bar a {
        color: #3b82f6;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.15s ease;
    }
    .footer-bar a:hover {
        color: #60a5fa;
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# 3. Cache the Compiled Graph for Maximum Efficiency
@st.cache_resource(show_spinner=False)
def get_compiled_graph():
    """Compiles and caches the LangGraph state machine to eliminate cold-start compile latency."""
    return build_graph()

# 4. Determine Active Model & Provider
active_provider = "Google Gemini" if GOOGLE_API_KEY and not GOOGLE_API_KEY.startswith("your_") else "OpenAI"
active_model = DEFAULT_GEMINI_MODEL if active_provider == "Google Gemini" else DEFAULT_OPENAI_MODEL

# 5. Sidebar Navigation & Architecture Inspector
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <div style="width: 36px; height: 36px; border-radius: 8px; background: linear-gradient(135deg, #2563eb, #1d4ed8); display: flex; align-items: center; justify-content: center; font-weight: 700; color: white;">KH</div>
        <div>
            <div style="font-weight: 700; font-size: 1rem; color: #f8fafc;">Control Panel</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">Autonomous Agent System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Engine Telemetry")
    st.markdown(f"""
    * **Provider:** `{active_provider}`
    * **Model:** `{active_model}`
    * **Max Loop Limit:** `{MAX_ITERATIONS}` steps
    * **Orchestrator:** `LangGraph v0.2`
    """)

    st.divider()
    st.markdown("### 🧰 Native Tool Suite")
    st.markdown("""
    * 🔍 **Web Search (`web_search`)**  
      Live web intelligence via DuckDuckGo (`ddgs`).
    * 🧮 **Calculator (`calculator`)**  
      Deterministic arithmetic & financial precision.
    * 📊 **Text Analysis (`text_analysis`)**  
      Linguistic stats, reading time & key skills.
    """)

    st.divider()
    st.markdown("### 🔄 ReAct Execution Cycle")
    st.code("""
[User Task]
    ↓
[agent] ◄────────┐
   │ (Router)    │
   ├── Needs Tool ─┘
   │
   └── Done / Limit
         ↓
       [END]
    """, language="text")

    st.divider()
    st.markdown("### 🌐 Portfolio & Links")
    st.markdown("""
    * 👨‍💻 [Hemanth's Portfolio](https://hemanth-karumanchi.github.io/)
    * 📦 [GitHub Repository](https://github.com/HEMANTH-KARUMANCHI/AI_Research-Decision_Making_Agent)
    """)

# 6. Hero Section (Matching Portfolio Design)
st.markdown(f"""
<div class="hero-container">
    <div class="hero-header">
        <div class="brand-group">
            <div class="brand-monogram">KH</div>
            <div>
                <h1 class="hero-title">AI Research & Decision-Making Agent</h1>
                <div class="hero-subtitle">
                    Autonomous ReAct Agent • Dynamic Tool Orchestration • Deterministic Reasoning
                </div>
            </div>
        </div>
        <div class="status-pill">
            <span class="pulse-dot"></span>
            Operational • {active_provider}
        </div>
    </div>
    <div class="badge-bar">
        <span class="badge badge-blue">⚡ LangGraph v0.2 State Machine</span>
        <span class="badge badge-purple">🤖 Autonomous Multi-Iteration Loop</span>
        <span class="badge badge-slate">🛠️ 3 Native Tools (Search, Math, NLP)</span>
        <span class="badge badge-slate">🛡️ Max-Iteration Safeguard ({MAX_ITERATIONS})</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 7. Quick-Prompt Scenario Grid
st.markdown("##### 💡 Select a Quick-Test Scenario or Type Your Own:")

if "input_query" not in st.session_state:
    st.session_state.input_query = ""

col_q1, col_q2, col_q3, col_q4 = st.columns(4)

with col_q1:
    if st.button("⚖️ Hardware: GPU Pricing & Math", use_container_width=True):
        st.session_state.input_query = (
            "Search for current pricing of NVIDIA H100 and AMD MI300X GPUs. "
            "Then calculate the cost difference between buying 8 NVIDIA H100s vs 8 AMD MI300Xs."
        )

with col_q2:
    if st.button("💰 Finance: SIP Savings & Bonus", use_container_width=True):
        st.session_state.input_query = (
            "If an investor saves ₹25,000 every month for 7 years, what is their total invested principal? "
            "Also compute what 10% annual bonus on that principal would equal."
        )

with col_q3:
    if st.button("📑 NLP: Lead AI Job Description", use_container_width=True):
        st.session_state.input_query = (
            "Analyze this job description for key metrics and skills: 'We are hiring a Lead AI Engineer "
            "skilled in Python, LangGraph, LangChain, PyTorch, Docker, Kubernetes, AWS, and RAG architectures.'"
        )

with col_q4:
    if st.button("🗄️ Architecture: Postgres vs Mongo", use_container_width=True):
        st.session_state.input_query = (
            "What are the major architectural differences between PostgreSQL and MongoDB, and when should an engineering team pick each?"
        )

# 8. Interactive Task Input
user_task = st.text_area(
    "Enter your research task or decision challenge:",
    value=st.session_state.input_query,
    placeholder="e.g., Research AWS vs GCP GPU compute costs, calculate savings for a 12-month reserved instance, and give a recommendation...",
    height=110
)

col_run, col_clear, _ = st.columns([1.5, 0.8, 4])
with col_run:
    run_button = st.button("🚀 Execute Autonomous Agent", type="primary", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.input_query = ""
        st.rerun()

# 9. Graph Execution & Observability Pipeline
if run_button and user_task.strip():
    start_time = time.time()
    
    st.divider()
    
    # Modern st.status container for live agent telemetry
    with st.status("⚡ Agent Reasoning & Telemetry in Progress...", expanded=True) as status_box:
        try:
            # Retrieve cached compiled graph
            graph = get_compiled_graph()
            
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
                                    st.write(f"🤔 **Iteration {current_iter} (Agent Decision):** Invoking tool `{tool_name}` with parameters:")
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
            
            # Update telemetry status box to complete
            status_box.update(
                label=f"✅ Execution Complete in {iterations_completed} Iteration(s) ({elapsed_time}s)",
                state="complete",
                expanded=False
            )

            # 10. Real-Time Telemetry Dashboard
            st.markdown("#### 📊 Execution Telemetry")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Iterations", f"{iterations_completed} / {MAX_ITERATIONS}")
            with m2:
                st.metric("Tools Executed", f"{total_tools_called}")
            with m3:
                st.metric("Latency", f"{elapsed_time}s")
            with m4:
                st.metric("Graph State", "Terminated (END)")

            # 11. Render Structured Recommendation Card
            st.markdown("#### 🎯 Synthesized Decision & Report")
            
            # Normalize list content if returned from model
            if isinstance(final_answer_text, list):
                final_answer_text = "\n".join(
                    [item.get("text", "") for item in final_answer_text if isinstance(item, dict) and "text" in item]
                )
            
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown(final_answer_text)
            st.markdown('</div>', unsafe_allow_html=True)

            # 12. Download Report Action
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
    st.warning("⚠️ Please provide a research question or select one of the quick scenarios above.")

# 13. Sleek Portfolio Footer
st.markdown("""
<div class="footer-bar">
    <div>
        Engineered by <strong><a href="https://hemanth-karumanchi.github.io/" target="_blank">Hemanth Karumanchi</a></strong> • AI & Data Systems
    </div>
    <div>
        Powered by <a href="https://github.com/langchain-ai/langgraph" target="_blank">LangGraph</a> & <a href="https://streamlit.io/" target="_blank">Streamlit</a>
    </div>
</div>
""", unsafe_allow_html=True)
