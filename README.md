# Agentic RAG IT Support Copilot

An evidence-aware IT support assistant deployed on DigitalOcean. The FastAPI backend runs a LangGraph workflow that checks the company knowledge base first, evaluates the retrieved evidence, and uses Tavily web search only when internal evidence is insufficient. A browser UI displays the response, its source citations, and the agent's decision trace.

This project is designed for a hosted deployment and can also be run locally for development. It is intentionally lightweight, easy to configure, and suitable for experimenting with grounded answers, retrieval quality, and evidence-based fallback behavior.

The repository also contains [Agentic_RAG.ipynb](./Agentic_RAG.ipynb), the instructor-style notebook demo from which the workflow was developed.

## Quick start

The application is deployed on DigitalOcean, but the repository can also be run locally for development and testing.

1. Create and activate a Python environment.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` (or use `.conda_env/.env`) and populate the required keys.
4. Index the sample knowledge base with `python ingest_sample_kb.py`.
5. Start the app with `python run.py`.
6. Open `http://127.0.0.1:8080` in the browser for local development.

The app exposes the chat UI, health endpoint, and FastAPI docs at `/docs`.

## Features

- Chat interface with example IT support questions.
- LangGraph routing for direct messages and IT questions.
- Pinecone retrieval with local Hugging Face embeddings.
- Evidence grading, Tavily web fallback, and a bounded query-rewrite retry.
- Answers with source labels, citations, and a workflow trace.
- Authenticated PDF, TXT, Markdown, and DOCX ingestion.
- SQLite audit records for questions, selected source, and workflow trace.
- Health endpoint and FastAPI interactive API documentation.

## Workflow

```text
Question
   |
   v
Route: direct conversation or IT support
   |-- direct --------------------------> Direct answer
   |
   v
Retrieve from Pinecone company KB
   |
   v
Grade private evidence
   |-- good ----------------------------> Grounded private-KB answer
   |
   v
Search Tavily and grade web evidence
   |-- good ----------------------------> Grounded web answer with citations
   |-- weak --> Rewrite query and retry (bounded)
                    |
                    v
             Insufficient-evidence response
```

## Technology

| Component | Technology |
| --- | --- |
| Web application and API | FastAPI, Uvicorn, Jinja2 |
| User interface | HTML, CSS, JavaScript |
| Agent orchestration | LangGraph |
| Chat model | OpenRouter through `ChatOpenAI` |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| Vector database | Pinecone |
| Web search fallback | Tavily |
| Audit storage | SQLite |

Hugging Face embeddings run locally after the model is downloaded; this avoids a paid embedding API. OpenRouter, Tavily, and Pinecone are separate services and may have usage limits or charges depending on your account and plan.

## Project structure

```text
.
├── app/
│   ├── api/routes.py          # Health, chat, and authenticated ingestion endpoints
│   ├── core/                  # Settings and logging
│   ├── rag/                   # LangGraph workflow, state, and Pinecone integration
│   ├── services/              # Document loading/chunking and SQLite audit
│   └── main.py                # FastAPI application and UI route
├── data/
│   └── sample_kb/             # Sample IT handbook and service-desk runbook
├── static/                    # Browser CSS and JavaScript
├── templates/index.html       # Chat and document-ingestion interface
├── uploads/                   # Local uploaded documents (not committed)
├── Agentic_RAG.ipynb          # Notebook demonstration
├── ingest_sample_kb.py        # Index the sample knowledge base
├── run.py                     # Start the local web application
└── requirements.txt
```

## Requirements and setup

- Python 3.12 recommended.
- Conda, or another Python environment manager.
- API keys for OpenRouter and Pinecone; a Tavily key is needed when the graph uses web fallback.

From the repository root, create and activate an environment and install dependencies:

```powershell
conda create -n agentic-rag python=3.12 -y
conda activate agentic-rag
python -m pip install -r requirements.txt
```

Configure credentials in a `.env` file at the repository root or in `.conda_env/.env`. The application reads either location. Start from `.env.example` if useful:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
PINECONE_API_KEY=your_pinecone_api_key
TAVILY_API_KEY=your_tavily_api_key
ADMIN_API_KEY=replace_with_a_long_random_secret
```

`ADMIN_API_KEY` is required for document uploads through the UI. Keep `.env` private; never commit API keys or upload real company documents to Git.

## Index sample knowledge

The sample handbook and service-desk runbook are included under `data/sample_kb/`. Index them into Pinecone with:

```powershell
python ingest_sample_kb.py
```

The configured defaults are Pinecone index `fde-it-support-rag` and namespace `company-it-kb`. Settings may be overridden using environment variables, including `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`, `EMBEDDING_MODEL`, `TOP_K`, and `MAX_RETRIES`. The index dimension must match the embedding model; the application reports a mismatch rather than deleting an existing index. The first local embedding run may download the Hugging Face model.

## Run the application locally

```powershell
python run.py
```

The app starts on `http://127.0.0.1:8080` because the local runner uses `uvicorn` with port `8080`. The API health endpoint is `http://127.0.0.1:8080/api/health`, and interactive API documentation is available at `http://127.0.0.1:8080/docs`.

If you prefer the more common port 8000, change the port in `run.py` or launch the app directly with `uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`.

## Deployment

This repository is set up for deployment in a containerized environment and is currently hosted on DigitalOcean. The Dockerfile is configured for a managed web service, with the application listening on `PORT` and binding to `0.0.0.0` inside the container.

## Run with Docker

Build the image from the repository root, where the root-level `Dockerfile` and `.dockerignore` are located:

```powershell
docker build -t agentic-rag .
docker run --rm -p 8080:8080 --env-file .conda_env/.env agentic-rag
```

Alternatively, pass the path to your root `.env` file. Provide the required OpenRouter, Pinecone, and Tavily credentials, and set `ADMIN_API_KEY` to enable uploads. The container listens on port `8080` and is exposed locally at `http://127.0.0.1:8080`. Container-local audit records and uploads are not persistent when the container is removed unless you configure Docker volumes.

The Docker build installs CPU-only PyTorch for the local Hugging Face embedding model to avoid CUDA packages. If the platform builder still runs out of memory, increase the build instance memory; the embedding model also uses memory at runtime.

To ask a question from the API:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/api/chat `
  -ContentType "application/json" `
  -Body '{"question":"How do I connect to the company VPN?"}'
```

The response includes `answer`, `source_used`, `trace`, `citations`, and `rewritten_query`.

## Add a company document

Use **Add company document** in the UI, enter the configured admin key, and choose a PDF, TXT, Markdown, or DOCX file. The `/api/ingest` endpoint also accepts the `X-Admin-Key` header. Uploads are written to the local `uploads/` directory and their chunks are added to Pinecone. Upload access is disabled if `ADMIN_API_KEY` is not configured.

## Hosted data and security

- `data/audit.db` stores chat questions, selected answer source, and workflow traces. Treat it as sensitive runtime data whether the app is running locally or in the cloud.
- Uploaded documents under `uploads/` are runtime data. Review and remove them according to your organization's retention requirements.
- Questions and retrieved context may be sent to OpenRouter; fallback searches are sent to Tavily; embeddings are computed locally and vectors are stored in Pinecone. Do not use confidential data unless these services are approved for it.
- The sample handbook and runbook are demo content, not real company policy. Replace them with documents you are authorized to use.
- This project is suitable for a managed deployment such as DigitalOcean, but it still requires review of identity/access controls, upload limits and validation, secrets management, logging, privacy, and operational monitoring before production use at scale.
