# CodeLens

Ingest any public GitHub repository, generate documentation, query the codebase in natural language and open pull requests, all from a single UI.

Built with Python, FastAPI, LangChain, Chroma, Groq (Llama 3.1 8B + Llama 3.3 70B) and Streamlit.

---

## What it does

| Tab | Capability |
|-----|-----------|
| **Ingest** | Clone a repo, chunk with tree-sitter AST parsing, embed locally, store in Chroma |
| **Generate Docs** | Produce API Reference, README or Beginner Guide. Download as Markdown or PDF |
| **RAG Code Assistant** | RAG-powered chat grounded in the actual source code with query expansion and deduplication |
| **Create PR** | Commit generated docs or code changes to a branch and open a pull request |

---

## Stack

- **Backend:** FastAPI, LangChain, Chroma, sentence-transformers, PyGithub, GitPython
- **LLM:** Groq free tier: Llama 3.1 8B (summarisation) + Llama 3.3 70B (assembly, Q&A, PR descriptions)
- **Frontend:** Streamlit
- **Infra:** Docker Compose, deployable to Railway or Render

Zero paid LLM spend. Everything runs on Groq's free tier.

---

## Project structure

```
├── backend/
│   ├── app/
│   │   ├── core/            config.py, logging.py
│   │   ├── models/          schemas.py
│   │   ├── services/        ingestion, chunker, vector_store, doc_generator,
│   │   │                    rag, llm_client, router, github_agent, oauth, cleanup
│   │   └── api/             ingestion.py, docs.py, github.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── streamlit_app.py
│   ├── api_client.py
│   ├── Dockerfile
│   └── requirements.txt
├── deploy/
│   ├── railway.toml
│   └── render.yaml
├── docker-compose.yml
└── README.md
```

---

## Quick start

### Prerequisites

- Python 3.11+
- git
- [Groq API key](https://console.groq.com) (free)
- GitHub OAuth App (for the PR tab, see below)

### 1. Clone and configure

```bash
git clone https://github.com/your-username/codelens
cd codelens
cp backend/.env.example backend/.env
```

Open `backend/.env` and fill in `GROQ_API_KEY` and your GitHub OAuth credentials.

### 2. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

First run downloads the embedding model (~80MB, one-time). Check it's up at `http://localhost:8000/health`.

### 3. Frontend

```bash
cd frontend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

### 4. Docker

```bash
cp backend/.env.example backend/.env
docker compose up --build
```

Backend at `http://localhost:8000`, frontend at `http://localhost:8501`.

---

## GitHub OAuth setup

1. Go to [github.com/settings/developers](https://github.com/settings/developers) and create a new OAuth App
2. Set **Authorization callback URL** to `http://localhost:8000/api/github/callback`
3. Copy the **Client ID** and generate a **Client Secret**
4. Add both to `backend/.env`:

```env
GITHUB_CLIENT_ID=your_client_id
GITHUB_CLIENT_SECRET=your_client_secret
```

Users connect their own GitHub account. Pull requests open under their username, not yours.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | From console.groq.com |
| `GITHUB_CLIENT_ID` | For PR tab | GitHub OAuth App client ID |
| `GITHUB_CLIENT_SECRET` | For PR tab | GitHub OAuth App client secret |
| `GITHUB_REDIRECT_URI` | For deploy | Defaults to `http://localhost:8000/api/github/callback` |
| `GITHUB_FRONTEND_URL` | For deploy | Defaults to `http://localhost:8501` |
| `MAX_STORED_JOBS` | No | Jobs to retain (default: 10) |

---

## Architecture notes

**Ingestion** runs as a FastAPI BackgroundTask. The frontend polls `GET /api/jobs/{id}` for status. Clones are deleted after embedding and old Chroma collections are pruned automatically via `MAX_STORED_JOBS`.

**Chunking** uses tree-sitter AST parsing for Python, JavaScript and TypeScript, extracting whole functions and classes with exact line ranges. Fixed-size windows with overlap as fallback for other file types.

**LLM routing** sends high-volume summarisation to `llama-3.1-8b-instant` (fast tier) and assembly, Q&A and PR descriptions to `llama-3.3-70b-versatile` (reasoning tier).

**RAG Q&A** uses query expansion before retrieval and deduplicates results by chunk ID. History is capped at 3 turns to stay within token limits.

**PR agent** detects ownership, auto-forks if needed, polls until the fork SHA is available, creates a branch, commits files and drafts a PR description with the LLM.