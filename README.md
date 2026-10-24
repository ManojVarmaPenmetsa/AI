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
- Persistent conversations with browser refresh and backend restart recovery.
- Authenticated PDF, TXT, Markdown, and DOCX ingestion with page-aware OCR and opt-in visual descriptions.
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
| Conversation ownership and LangGraph checkpoints | SQLite |
| Audit storage | Separate SQLite database |

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

Conversations are stored in `CONVERSATION_DB_PATH` (default `data/conversations.db`). The LangGraph checkpointer and the conversation ownership registry share this database connection, while audit records remain in `AUDIT_DB_PATH`. For a hosted single-instance deployment, point this setting at a persistent mounted disk so data survives replacement of the application container. SQLite is suitable for local development and a single app instance; deployments with multiple app replicas or a shared network filesystem should move the checkpointer and ownership registry to a managed PostgreSQL backend before scaling. This app currently configures the SQLite saver.

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

The response includes `answer`, `source_used`, `trace`, `citations`, `rewritten_query`, and `conversation_id`. Pass that ID in subsequent `/api/chat` requests to continue the thread. `POST /api/conversations` starts a new thread; `GET /api/conversations/{conversation_id}` resumes its visible message history. The API sets an opaque HttpOnly browser cookie and binds each ID to that owner token. This is browser-level isolation; deployments that need account-level identity should connect it to the application's authentication system.

## Add a company document

Use **Add company document** in the UI, enter the configured admin key, and choose a PDF, TXT, Markdown, or DOCX file. The `/api/ingest` endpoint also accepts the `X-Admin-Key` header. Upload access is disabled if `ADMIN_API_KEY` is not configured.

PDF ingestion first extracts selectable page text and detected tables. Pages with little selectable text are rendered and sent to the configured OCR provider. Relevant pages with large embedded images or many vector drawings are rendered as complete pages for visual analysis, subject to `PDF_VISION_MAX_PAGES`. The example configuration selects `nvidia/nemotron-3.5-super-vl:free`, an image-input model listed by OpenRouter. The vision client reuses `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL` unless `VISION_API_KEY` or `VISION_BASE_URL` overrides them. Keep the base URL as an API root (for OpenRouter, `https://openrouter.ai/api/v1`); the OpenAI-compatible client adds `/chat/completions` once, and a configured URL that already ends in that path is normalized first. The ordinary text model is unchanged and is never assumed to accept images. When there are visual pages but vision is unavailable, the upload response reports `vision_status` as `not_configured` or `partial`; successfully described pages report `completed`. Text and OCR continue to be indexed where possible. Provider errors, empty responses, and page-limit skips appear as warnings.

OCR defaults to local Tesseract (`OCR_PROVIDER=tesseract`) and does not use a paid OCR service. On Windows, install a current build using `winget install --id UB-Mannheim.TesseractOCR -e` or the installer linked from [Tesseract's download guidance](https://tesseract-ocr.github.io/tessdoc/Downloads.html), then open a new terminal and verify it with `tesseract --version`. Set `TESSERACT_CMD` to the full `tesseract.exe` path if the executable is not on `PATH`; the Docker image installs the English Tesseract binary. Set `OCR_PROVIDER=none` to disable OCR. `VISION_TIMEOUT_SECONDS`, `OCR_TIMEOUT_SECONDS`, render DPI, page text threshold, visual thresholds, maximum rendered pixel count (`PDF_VISION_MAX_PIXELS`), maximum image payload bytes (`PDF_VISION_MAX_IMAGE_BYTES`), and maximum number of vision pages are configurable. Oversized pages/images are skipped with an upload warning while other extracted content continues through indexing.

To try PDF ingestion locally, set `OPENROUTER_API_KEY` and `ADMIN_API_KEY` in the active env file (`.conda_env/.env` takes precedence over the root `.env`), install/verify Tesseract for scanned pages, start the app, and upload a small PDF from **Add company document**. Check the returned page, OCR, and vision status fields and then ask a question grounded in its visual content to verify retrieval. Run the mocked, no-paid-API unit suite with `python -m unittest discover -s tests -v` (or `.\.conda_env\python.exe -m unittest discover -s tests -v`). OpenRouter free-model limits vary by account and may change; check the current key's allowance through OpenRouter's key API or dashboard. A free route may be rate-limited or temporarily unavailable; such pages are marked partial and their ordinary text/OCR remain indexed. Select another model that explicitly lists image input in the [OpenRouter model browser](https://openrouter.ai/models?input_modalities=image) by changing `VISION_MODEL`; do not change `OPENROUTER_MODEL` unless you intend to change chat behavior.

Ingestion audit records store document identifiers, version, status, vision status, and safe warning diagnostics. They do not store document text, image payloads, API keys, or model responses.

The document registry is stored in `DOCUMENT_REGISTRY_DB_PATH` (default `data/document_registry.db`), separately from conversations and audit records. It assigns durable document identities, tracks ingestion generations, and records which fully verified generation is active. Uploads are parsed and chunked, upserted, fetched back by expected chunk IDs, and only then activated in a SQLite transaction. Retrieval filters to active generations and continues to include older vectors that have no generation metadata until a replacement for their document is activated. Failed or incomplete generations remain inactive. Cleanup after activation is restricted to the selected `document_id`; stale chunk IDs from a retried generation are recorded for cleanup recovery.

The first upload for a filename adopts the previous filename hash as a migration bridge, then records it in the registry. Existing vectors are not reset or globally migrated. Reupload that filename to replace the registered document. To create a second logical document with the same filename, send the multipart form field `new_document=true`; subsequent updates must include that returned `document_id`. If a filename already has multiple registered documents, upload returns HTTP 409 with their IDs instead of guessing. The UI's normal upload remains unchanged for unambiguous filenames. `DELETE /api/ingest/{document_id}` using `X-Admin-Key` removes that document's vectors, records a registry tombstone so delayed legacy vectors stay hidden, and removes its versioned local uploads.

The registry and upload files must be on persistent storage and backed up together. SQLite locking is intended for a single-host app. Before running multiple app replicas, migrate the registry to a shared transactional database with cross-process advisory locking; the active pointer and Pinecone writes cannot participate in one distributed transaction. A process crash after activation but before cleanup is safe for retrieval: the registry continues to select the new generation and the next successful replacement retries document-scoped cleanup. Legacy vectors without `document_id` remain available for compatibility and require an explicit inventory/migration before they can be safely associated with a document.

Every new chunk stores `document_id`, content-hash `document_version`, `generation_id`, `chunk_id`, source filename, page, content type, ingestion timestamp, provenance, and existing multimodal metadata. Chunk IDs are deterministic within a generation and include the generation ID, so changes in OCR or vision output produce distinct chunks. Private KB citations include page numbers so generated descriptions can be checked against the original PDF.

## Hosted data and security

- `data/audit.db` stores chat questions, selected answer source, and workflow traces. Treat it as sensitive runtime data whether the app is running locally or in the cloud.
- `data/conversations.db` stores LangGraph execution checkpoints and a separate conversation ownership table. The graph uses LangGraph's message state for turns and restricts checkpoint deserialization to its safe built-in type allowlist. Do not put credentials or unnecessary sensitive data in chat messages.
- Uploaded documents under `uploads/` are runtime data. Review and remove them according to your organization's retention requirements.
- Questions and retrieved context may be sent to OpenRouter; fallback searches are sent to Tavily; embeddings are computed locally and vectors are stored in Pinecone. Do not use confidential data unless these services are approved for it.
- The sample handbook and runbook are demo content, not real company policy. Replace them with documents you are authorized to use.
- This project is suitable for a managed deployment such as DigitalOcean, but it still requires review of identity/access controls, upload limits and validation, secrets management, logging, privacy, and operational monitoring before production use at scale.
