# 🤖 Multi-Agent Collaboration Development Team

A fully automated software development team powered by **LangGraph** and **LangChain**. Through a Supervisor–Worker architecture, the system performs requirements analysis, task breakdown, asynchronous parallel frontend/backend development, and strict automated code reviews — delivering a complete software project from a single natural-language prompt.

---

## ✨ Features

- **Two-Phase Execution** — Planning phase designs the architecture; Execution phase implements it in parallel
- **True Async Parallel Development** — Frontend and backend experts work simultaneously via LangGraph's `Send` API
- **Strict Automated Code Review** — Every submission goes through a Reviewer agent that never compromises
- **Circuit Breaker** — Prevents infinite retry loops with a configurable `MAX_RETRIES` limit (default: 10)
- **Structured Outputs** — Pydantic models enforce strict data contracts between all agents
- **Pluggable Expert Architecture** — Easily extend with Database Expert, DevOps Expert, etc.
- **Workspace Sandboxing** — All file operations and shell commands are restricted to a dedicated workspace directory

---

## 🏗️ Architecture

### System Workflow

```
┌─────────────────── Phase 1: Planning ────────────────────┐
│                                                          │
│   User Requirement                                       │
│        │                                                 │
│        ▼                                                 │
│   ┌──────────┐     ┌──────────────┐                      │
│   │ Planner  │────▶│ Plan Reviewer │──── pass ──────┐    │
│   └──────────┘     └──────────────┘                 │    │
│        ▲                  │                         │    │
│        └──── fail ────────┘                         │    │
│              (with feedback)                        │    │
└─────────────────────────────────────────────────────│────┘
                                                      │
                                                      ▼
┌─────────────────── Phase 2: Execution ───────────────────┐
│                                                          │
│   ┌──────────┐                                           │
│   │  Leader   │──── dispatches tasks in parallel ──┐     │
│   └──────────┘                                     │     │
│        ▲                                           │     │
│        │                              ┌────────────┴──┐  │
│        │                              │               │  │
│   all done                    ┌───────▼──┐   ┌───────▼──┐│
│        │                      │ Frontend  │   │ Backend  ││
│        │                      │  Expert   │   │  Expert  ││
│   ┌────┴────────┐             └─────┬─────┘   └─────┬────┘│
│   │Task Reviewer│◀──────────────────┴───────────────┘    │
│   └─────────────┘                                        │
│     │         │                                          │
│    pass      fail ──▶ back to Expert (with feedback)     │
│     │                                                    │
│     ▼                                                    │
│   Leader (next round)                                    │
│     ...                                                  │
│     ▼                                                    │
│    END                                                   │
└──────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Role | Key Behavior |
|---|---|---|
| **Planner** | System architecture & task breakdown | Produces fine-grained frontend/backend task lists |
| **Leader** | Task dispatching & progress tracking | Reads state, dispatches one task per domain in parallel. Never writes code. |
| **Frontend Expert** | Frontend implementation | Receives a sub-task, uses tools (read/write/bash) via ReAct loop |
| **Backend Expert** | Backend implementation | Same as Frontend Expert, but for backend domain |
| **Reviewer** | Quality gate & code review | Strict pass/fail. Handles both plan review (Phase 1) and code review (Phase 2) |

---

## 📁 Project Structure

```
agent_team/
├── main.py                          # CLI entry point
├── pyproject.toml                   # Dependencies & project metadata
├── .env.example                     # Environment variable template
│
├── agent_team/                      # Core package
│   ├── schemas/
│   │   ├── state.py                 # GraphState (TypedDict) with reducers
│   │   └── models.py               # Pydantic models (PlannerOutput, LeaderDecision, etc.)
│   │
│   ├── agents/
│   │   ├── planner.py               # Planner node
│   │   ├── leader.py                # Leader node (task dispatcher)
│   │   ├── experts.py               # Frontend/Backend expert nodes (ReAct loop)
│   │   └── reviewer.py              # Reviewer node (plan + task review, circuit breaker)
│   │
│   ├── tools/
│   │   ├── file_tools.py            # read_file, write_file (with path-traversal protection)
│   │   └── bash_tool.py             # bash tool (subprocess with timeout)
│   │
│   └── graph/
│       ├── config.py                # Configuration from .env
│       └── builder.py               # StateGraph construction & routing
│
└── tests/
    ├── conftest.py                  # Shared fixtures (mock LLM outputs)
    ├── test_schemas.py              # Pydantic model validation tests
    ├── test_graph.py                # Graph compilation & routing tests
    └── test_circuit_breaker.py      # Circuit breaker mechanism tests
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python ≥ 3.11
- An OpenAI-compatible API key

### 2. Install

```bash
git clone <repo-url>
cd agent_team
pip install -e ".[dev]"
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` with your API credentials:

```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
TEMPERATURE=0.2
MAX_RETRIES=10
WORKSPACE_PATH=./workspace
```

> **Note:** Any OpenAI-compatible API endpoint works (e.g. Azure OpenAI, local LLM servers). Just set `OPENAI_BASE_URL` and `MODEL_NAME` accordingly.

### 4. Run

```bash
# Via command-line flag
python main.py --requirement "Build a REST API with user authentication and a React dashboard"

# Via stdin
python main.py
# Then type your requirement and press Ctrl+Z (Windows) or Ctrl+D (Unix) to submit
```

### 5. Run Tests

```bash
pytest tests/ -v
```

Tests use mock LLM responses — no API key required.

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `""` | Your API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint URL |
| `MODEL_NAME` | `gpt-4o` | LLM model to use |
| `TEMPERATURE` | `0.2` | LLM sampling temperature |
| `MAX_RETRIES` | `10` | Circuit breaker threshold per task |
| `WORKSPACE_PATH` | `./workspace` | Directory where generated code is written |

---

## 🛡️ Safety & Guardrails

### Circuit Breaker
If any single task is rejected by the Reviewer **10 times** (configurable), it is automatically marked as `failed` and the workflow moves on. This applies to both:
- **Phase 1**: Planner output rejection
- **Phase 2**: Expert code submission rejection

### Path Traversal Protection
All `read_file` and `write_file` operations resolve paths relative to `WORKSPACE_PATH` and reject any path that resolves outside it (e.g. `../../etc/passwd`).

### Subprocess Timeout
The `bash` tool enforces a **60-second timeout** on all commands and restricts execution to the workspace directory.

### Hallucination Guard
The Leader validates all LLM-dispatched tasks against the actual task list — only tasks with `status="pending"` are accepted. Completed, failed, or nonexistent task IDs are silently filtered.

### Parallel Data Safety
- `code_base` uses a **dict-merge reducer** so parallel experts' file writes combine instead of overwriting
- `expert_submissions` uses an **operator.add reducer** so parallel results concatenate correctly

---

## 🔌 Extending with New Experts

The expert architecture is designed as a factory pattern. To add a new expert (e.g. Database Expert):

1. **In `experts.py`:**
   ```python
   database_expert_node = _build_expert_node("database")
   ```

2. **In `models.py`:** Add `"database"` to the domain regex patterns:
   ```python
   pattern=r"^(frontend|backend|database)$"
   ```

3. **In `builder.py`:** Register the node and add edges:
   ```python
   graph.add_node("database_expert", database_expert_node)
   graph.add_edge("database_expert", "task_reviewer")
   ```

4. **Update routing functions** to handle the new expert name.

---

## 📄 License

MIT
