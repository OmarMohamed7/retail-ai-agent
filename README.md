# Retail Analytics Copilot (DSPy + LangGraph)

AI agent that answers retail analytics questions by combining:
● RAG over local docs (docs/)
● SQL over a local SQLite DB (Northwind)

## 📋 Prerequisites

- Python 3.12 or higher
- pip (Python package installer)

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd retail-ai-agent
   ```

2. **Create and activate virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

1. **Activate your virtual environment**

   ```bash
   source venv/bin/activate
   ```

2. **Run the application**

   ```bash
   python app.py
   ```

## 📁 Project Structure

```
retail-ai-agent/
├── agent/                    # Core agent modules
│   ├── __init__.py
│   ├── config.py             # Configuration management (Pydantic)
│   ├── dspy_signatures.py     # DSPy signatures (Router, GenerateSQL, SynthesizeAnswer)
│   ├── graph_hybrid.py       # LangGraph workflow definition
│   ├── rag/                  # RAG components
│   │   ├── __init__.py
│   │   └── retrieval.py      # Document retriever (TF-IDF)
│   └── tools/                 # Database tools
│       ├── __init__.py
│       └── sqlite_tool.py    # SQLite database interface
├── data/                      # Database files
│   └── northwind.db          # Northwind SQLite database
├── docs/                      # Documentation files for RAG
│   ├── catalog.md
│   ├── kpi_definitions.md
│   ├── marketing_calender.md
│   └── product_policy.md
├── docker/                    # Docker configuration
│   └── docker-compose.yml     # Ollama service setup
├── run_agent_hybrid.py        # Main entry point
├── sample_questions_hybrid_eval.jsonl  # Sample questions
├── outputs_hybrid.jsonl       # Output results
├── requirements.txt           # Python dependencies
├── .env                       # Environment configuration
└── README.md                  # This file
```

## 🔄 LangGraph Workflow

The agent uses a **LangGraph StateGraph** with the following nodes and flow:

```
                    START
                      │
                      ▼
                  [Router] ───────────────┐
                      │                  │
         ┌────────────┼────────────┐     │
         │            │            │     │
         ▼            ▼            ▼     │
      [RAG]        [SQL]       [Hybrid]   │
         │            │            │     │
         │            │            │     │
         └────────────┼────────────┘     │
                      │                  │
                      ▼                  │
              [Planner/ExtractConstraints]│
                      │                  │
                      ▼                  │
                  [NL2SQL]               │
                      │                  │
                      ▼                  │
                 [Executor]              │
                      │                  │
         ┌────────────┴────────────┐     │
         │                         │     │
    [Success]                  [Error] │
         │                         │     │
         │                         ▼     │
         │                    [Repair] ──┘
         │                         │
         │                         │
         └────────────┬────────────┘
                      │
                      ▼
              [Synthesizer]
                      │
                      ▼
                     END
```

### Graph Nodes:

1. **Router** - Determines route: `rag`, `sql`, or `hybrid`
2. **Retriever** - Retrieves relevant document chunks using TF-IDF
3. **Planner** - Extracts constraints (dates, filters) from documents
4. **NL2SQL** - Generates SQL query from natural language
5. **Executor** - Executes SQL query against SQLite database
6. **Repair** - Handles SQL errors and retries (max 2 attempts)
7. **Synthesizer** - Combines SQL results and document chunks into final answer

### State Management:

The graph uses `AgentState` (TypedDict) to pass data between nodes:
- Input: `question`, `format_hint`, `question_id`
- Routing: `route`, `reasoning`
- RAG: `retrievied_chunks`
- SQL: `db_schema`, `extracted_constraints`, `query`, `query_result`
- Output: `answer`, `citations`
- Control: `error`, `repair_attempts`, `trace`

## 🔧 Configuration

Create a `.env` file in the root directory with your configuration:

```env

```
