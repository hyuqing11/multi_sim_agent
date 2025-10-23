# Multi-DFT Agent Platform

## Overview

This repository hosts a research prototype for a multi-agent assistant that helps
materials scientists plan and execute density functional theory (DFT)
workflows. The system combines LangGraph-based agents, a FastAPI backend, and a
Streamlit chat UI. Available agents can research recent literature, design and
validate VASP inputs, plan end-to-end pipelines, and submit jobs to HPC
clusters through Machine Control Protocol (MCP) tools.

## Key Features

- **LangGraph agent service** with typed state graphs, SQLite checkpointing, and
  streaming updates via Server-Sent Events.
- **Agent catalog** covering general chat, VASP structure/input generation,
  literature search, workflow planning, HPC job submission, and a high-level
  orchestrator that chains them together.
- **Streamlit chat client** that supports token streaming, tool-call displays,
  feedback collection, and deep links for sharing conversations.
- **Extensible tool layer** with support for Tavily-powered search, MCP HPC job
  execution, and external data sources (Materials Project, ASE, pymatgen, etc.).
- **Containerized deployment** through `docker-compose` plus shell scripts for
  local development.

## Repository Layout

```
backend/                FastAPI service, LangGraph agents, shared models
frontend/               Streamlit UI and static assets
scripts/                Helper scripts for install, run, and stop workflows
docker-compose.yaml     Two-service deployment (backend + frontend)
pyproject.toml          Poetry-style project configuration managed by uv
WORKSPACE/              User-facing workspace mounted into agents (created on install)
logs/, data/            Runtime logs and scratch directories (created on install)
```

## Getting Started

### Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) package manager (installed automatically by
  `scripts/install.sh` if missing)
- API keys for at least one supported LLM provider (OpenAI, Anthropic, Groq,
  Hugging Face, Ollama, or the built-in fake model for testing)
- Optional: Tavily API key for enriched web search, HPC credentials, and other
  provider-specific secrets

### 1. Install Dependencies

```bash
scripts/install.sh
```

The script:

- Installs `uv` if needed
- Creates a `.env` file (copying `env.example` when available)
- Creates runtime folders (`WORKSPACE`, `logs`, `data/inputs`)
- Syncs all dependencies defined in `pyproject.toml` into `.venv`

### 2. Configure Environment Variables

Edit `.env` and provide the secrets required for the agents you plan to run. At
a minimum you should set one LLM API key so the backend can select a default
model. Common variables include:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
HF_API_KEY=...
OLLAMA_MODEL=llama3
AUTH_SECRET=optional-shared-secret

# Optional agent integrations
TAVILY_API_KEY=...
MP_API_KEY=...              # Materials Project
ASTA_KEY=...                # External data sources
HOST=0.0.0.0
PORT=8083
```

If `AUTH_SECRET` is supplied, the backend enforces bearer-token authentication
for all endpoints and the Streamlit client will include the header automatically.

### 3. Run the Services

Use the helper scripts to start both backend and frontend:

```bash
scripts/run.sh
```

This launches:

- FastAPI agent service on `http://localhost:8083`
- Streamlit chat UI on `http://localhost:8501`

Stop them with:

```bash
scripts/stop.sh
```

You can also run the components manually:

```bash
uv run uvicorn backend.api.main:app --host 0.0.0.0 --port 8083 --reload
uv run streamlit run frontend/app.py
```

Or deploy with Docker:

```bash
docker compose up --build
```

The compose file mounts `WORKSPACE/` into both containers so generated inputs
and artifacts are easy to inspect.

## Using the Assistant

Open the Streamlit UI and select an agent and model from the Settings popover.
The chat interface supports streaming tokens, displays tool activity (including
sub-agent hand-offs), and records feedback on the final response.

Thread IDs are persisted in the URL, allowing you to share or resume a session.
The frontend lazily fetches history from the backend so conversation state is
restored after a reload.

## Agent Catalogue

- `chatbot`: General-purpose DFT assistant that can delegate to other agents.
- `vasp_structure_agent`: Generates candidate crystal structures and validates
  them using ASE/pymatgen tooling.
- `vasp_input_agent`: Builds VASP (and some LAMMPS) input decks given target
  structures and calculation settings.
- `vasp_pipeline_agent`: Chains structure generation with input preparation to
  produce ready-to-run workflows.
- `hpc_agent`: Connects to MCP servers (see `backend/agents/library/hpc_agent`)
  for cluster automation, job submission, and monitoring. Integrates Tavily
  search for policy lookups.
- `literature_agent`: Uses the CROW stack to survey recent literature and
  extract relevant findings for a user query.
- `materials_planner`: Translates literature insights into concrete DFT
  experiment plans.
- `materials_orchestrator`: High-level coordinator that blends research,
  planning, and execution agents.

Each agent is defined as a LangGraph state graph and lazily constructed via the
factory functions in `backend/agents/agent_manager.py`.

## Developer Guide

- Set `PYTHONPATH` to the repository root when running scripts directly (the
  helper scripts already do this).
- Use `uv` for common tasks:

  ```bash
  uv run ruff check .        # lint
  uv run pytest              # tests (if added)
  ```

- The backend keeps conversation state in `checkpoints.db` via
  `AsyncSqliteSaver`. Delete the file if you want a clean slate.
- Streamlit stores per-session data in `st.session_state`; a rerun preserves the
  current thread and cached agent metadata.

## Troubleshooting

- **Missing API key:** The backend raises `ValueError: At least one LLM API key
  must be provided.` Ensure you set one of the supported keys in `.env`.
- **AUTH failure:** If you configured `AUTH_SECRET`, include the same token in
  any external client. The bundled Streamlit UI reads it from `.env`.
- **Streaming errors:** The frontend now closes async streams on rerun; if you
  still see network issues, check `logs/service.log` and `logs/app.log`.
- **Docker permissions:** The compose file mounts `WORKSPACE` as a volume. Make
  sure the host user has write access when agents need to persist files.

## Contributing

- Format patches with readable commit messages.
- Keep new dependencies in `pyproject.toml`; run `uv sync` to refresh
  `uv.lock`.
- Prefer `rg` for repository searches and async-friendly patterns in agents.

Feel free to open issues or PRs for additional agents, new tools, or improved
workflow orchestration.

