# 🧠 AI Research & Decision-Making Agent — LangGraph

An end-to-end autonomous research and decision-making AI agent built with **LangGraph**, **LangChain**, and **Streamlit**. 

Unlike simple RAG (Retrieval-Augmented Generation) applications that merely retrieve and summarize documents, this project demonstrates an **Agentic AI system**: an autonomous reasoning control loop that dynamically analyzes user queries, selects appropriate tools, executes them sequentially, handles errors, respects iteration limits, and synthesizes grounded decisions.

---

## 🏗️ 1. Project Overview & Architecture

```text
                                  USER QUERY
                                       │
                                       ▼
                              STREAMLIT WEB UI
                                       │
                                       ▼
                             ┌──────────────────┐
                             │    AGENT NODE    │ (Inspects state history & reasons)
                             └─────────┬────────┘
                                       │
                              should_continue?
                               /              \
             [Requires Tool]  /                \  [Information Sufficient OR
                             ▼                  \  Iterations >= MAX]
                   ┌───────────────────┐         ▼
                   │    TOOLS NODE     │       [END] ──► Final Synthesis to User
                   │  (Prebuilt Tool)  │
                   └─────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
     ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
     │ web_search  │  │ calculator  │  │text_analysis │
     │ (DuckDuckGo)│  │ (Safe Math) │  │  (NLP/Stats) │
     └──────┬──────┘  └──────┬──────┘  └──────┬───────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
                             ▼
                        ToolMessage
                             │
                             └────────► (Loops back to Agent Node)
```

---

## 🚀 2. Key Technologies

* **Python 3.10+**: Core runtime
* **LangGraph**: Stateful graph orchestration, dynamic conditional routing, and cyclic agent loops
* **LangChain Core & Community**: Message schemas (`HumanMessage`, `AIMessage`, `ToolMessage`) and tool bindings
* **Google Gemini / OpenAI**: Foundation model backends (with auto-failover and exponential retry)
* **DuckDuckGo Search**: Real-time web information retrieval
* **Streamlit**: Interactive user interface with real-time agent execution trace
* **Pytest**: Unit and integration test suite

---

## 🛠️ 3. Primary Agent Tools

The agent is equipped with three tools designed for specific reasoning domains:

1. **`web_search`** (`tools/web_search.py`):
   * Retrieves live, real-time facts, hardware specifications, and market pricing from DuckDuckGo.
2. **`calculator`** (`tools/calculator.py`):
   * Evaluates exact mathematical and financial formulas (`20000 * 12 * 5`, exponential growth, ratios).
   * Sandboxed Python execution with protection against arbitrary code execution and division-by-zero crashes.
3. **`text_analysis`** (`tools/text_analysis.py`):
   * Calculates textual metrics: total word count, character count, sentence count, estimated reading time, and keyword frequency.

---

## 🔄 4. How the LangGraph Control Loop Works

1. **State Initialization (`AgentState`)**:
   * A shared state dictionary holding `messages: Annotated[list[BaseMessage], add_messages]` and an `iterations: int` counter.
   * `add_messages` reducer ensures messages append rather than overwrite history.
2. **Agent Reasoning Node (`agent_node`)**:
   * Evaluates message history and decides whether additional external data is needed.
   * Emits either an `AIMessage` with `tool_calls` or a final synthesis.
3. **Conditional Routing (`should_continue`)**:
   * Checks if `iterations >= MAX_ITERATIONS` (default: 5) to block runaway loops.
   * Routes to the `tools` node if tool calls are present; otherwise terminates to `END`.
4. **Tool Execution Node (`tools`)**:
   * Runs the requested tool in Python, captures the output, and appends a `ToolMessage` with the matching `tool_call_id`.
5. **Observation & Re-Evaluation Loop**:
   * Flow routes back to `agent_node` so the LLM can evaluate the tool result and decide the next action.

---

## 📂 5. Project Directory Structure

```text
AI Research & Decision-Making Agent/
│
├── app.py                     # Streamlit web application & trace visualizer
│
├── agent/                     # LangGraph agent orchestration
│   ├── __init__.py
│   ├── graph.py               # Graph builder and compilation
│   ├── state.py               # AgentState schema (messages + iterations)
│   ├── nodes.py               # Agent reasoning node & LLM invocation
│   └── router.py              # Conditional routing logic & loop termination
│
├── tools/                     # Modular tool implementations
│   ├── __init__.py
│   ├── web_search.py          # Real-time web search tool
│   ├── calculator.py          # Safe Python arithmetic tool
│   └── text_analysis.py       # Text statistics and keyword extraction tool
│
├── config/                    # Configuration and settings
│   ├── __init__.py
│   └── settings.py            # Centralized settings & LLM provider factory
│
├── tests/                     # Automated test suite
│   ├── test_agent.py          # Graph compilation & routing tests
│   ├── test_calculator.py     # Calculator arithmetic, error & security tests
│   └── test_tools.py          # Text analysis & search integration tests
│
├── phase1_basic_llm.py        # Phase 1 verification script
├── phase2_minimal_graph.py    # Phase 2 minimal graph verification
├── phase3_calculator_tool.py  # Phase 3 manual tool call verification
├── phase4_conditional_routing.py # Phase 4 dynamic routing verification
├── phase5_multiple_tools.py   # Phase 5 multi-tool verification
├── phase6_agent_loop.py       # Phase 6 reason-act loop verification
├── phase7_error_handling.py   # Phase 7 error & boundary verification
│
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Python package dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore file
└── README.md                  # Project documentation
```

---

## ⚡ 6. Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Git

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone <your-repo-url>
cd "AI Research & Decision-Making Agent"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your API credentials (either Google Gemini or OpenAI):
```env
# Option A: Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Option B: OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Adjust maximum loop iterations (default: 5)
MAX_ITERATIONS=5
```

---

## 🖥️ 7. Running the Application

### Launch the Streamlit Web Interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Run Interactive Console Scripts:
Each developmental phase has a standalone verification script:
```bash
python phase6_agent_loop.py
python phase7_error_handling.py
```

---

## 🧪 8. Running Automated Tests

Run the full pytest suite:
```bash
python -m pytest -v
```

All 11 test cases verify:
* Graph compilation and cycle validation
* Routing to `tools` when tool calls are requested
* Routing to `END` on final response synthesis
* Hard enforcement of `MAX_ITERATIONS`
* Calculator arithmetic, operators, division-by-zero, and sanitization
* Text analysis metric extraction and empty input handling

---

## 💡 9. Example Queries to Try

1. **Multi-Tool Research & Financial Comparison:**
   > *"Compare NVIDIA H100 and AMD MI300X specifications and calculate the price difference for an 8-GPU cluster."*
2. **Deterministic Arithmetic:**
   > *"If I invest ₹25,000 every month for 7 years, what is my total principal contribution?"*
3. **Document Analysis:**
   > *"Analyze this job description and identify the key skills: 'Looking for a Senior AI Engineer proficient in Python, LangGraph, Docker, Kubernetes, and AWS.' "*
4. **General Reasoning (Zero Tool Calls):**
   > *"What is the difference between synchronous and asynchronous programming in Python?"*
5. **Graceful Error Handling:**
   > *"Calculate 100 / 0 using the calculator tool."*

---

## ⚠️ 10. Limitations & Future Improvements

### Current Limitations:
* **Stateless UI Sessions**: Conversation history resets per question (single-turn agent graph).
* **Single Web Search Engine**: Relies on DuckDuckGo search without fallbacks to paid APIs (SerpAPI/Tavily).

### Future Roadmap:
* **Checkpointer / Long-Term Memory**: Integrate `MemorySaver` or SQLite checkpointer to support multi-turn conversational chat.
* **MCP (Model Context Protocol)**: Expose agent tools via standardized MCP servers.
* **Human-in-the-Loop**: Implement approval checkpoints before high-stakes tool execution.
