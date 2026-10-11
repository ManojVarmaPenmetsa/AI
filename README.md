# Agentic RAG

A Jupyter notebook prototype of an Agentic Retrieval-Augmented Generation (RAG) workflow. It uses LangGraph to decide whether to answer from a vector-store knowledge base, fall back to web search, retry with a rewritten query, or respond directly to a simple message.

> **Project scope:** The working Agentic RAG workflow is currently a notebook prototype. The repository also includes an app-oriented directory scaffold, but its FastAPI service, web interface, document-upload flow, and audit database are placeholders—not implemented features.

## What it demonstrates

- Routes greetings and simple messages separately from knowledge-seeking questions.
- Loads and chunks source material, embeds it locally, and indexes it in Pinecone.
- Retrieves relevant chunks and grades whether the private knowledge base is sufficient.
- Uses Tavily web search when the knowledge-base evidence is weak, then grades the web evidence.
- Rewrites a weak query and retries, with a retry limit to avoid an unbounded loop.
- Generates an answer from the selected source and records whether it came from the knowledge base, web search, or direct response.

## Workflow

```text
Question
   |
   v
Route question
   |-- greeting/simple message ----------> Direct answer
   |
   v
Retrieve from Pinecone knowledge base
   |
   v
Grade knowledge-base evidence
   |-- good -----------------------------> Answer from knowledge base
   |
   v
Tavily web search
   |
   v
Grade web evidence
   |-- good -----------------------------> Answer from web results
   |-- weak --> Rewrite query and retry (bounded)
                    |
                    v
              Insufficient-evidence response
```

## Technology

| Component | Technology |
| --- | --- |
| Agent workflow | LangGraph |
| Chat model | OpenRouter through `ChatOpenAI` |
| Current model ID | `nvidia/nemotron-3.5-lightning:free` |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Pinecone |
| Web search fallback | Tavily |
| Implementation | Python notebook |

## Repository contents

```text
.
├── Agentic_RAG.ipynb
├── create_project.py
├── requirements.txt
├── app/                # Application scaffold; modules are placeholders
├── data/sample_kb/
├── static/
├── templates/
├── tests/
├── uploads/
└── README.md
```

### Create the application scaffold

Run the scaffold script from the repository root:

```powershell
python create_project.py
```

It creates missing application folders and placeholder files without overwriting existing files. It also creates `.env.example` with the keys used by the notebook; it deliberately does not create `.env`. Copy `.env.example` to `.env` and add your own credentials. The generated application files are starting points and do not yet provide a runnable web app.

## Prerequisites

- Python 3.12 (the notebook has been used with Python 3.12).
- Conda or another Python environment manager.
- API keys for OpenRouter, Pinecone, and Tavily.
- VS Code with the Jupyter extension, or another Jupyter notebook interface.

## Setup

Create and activate an environment, then install the project dependencies:

```powershell
conda create -n agentic-rag python=3.12 -y
conda activate agentic-rag
python -m pip install -r requirements.txt
```

In VS Code, open `Agentic_RAG.ipynb` and select the `agentic-rag` environment as the notebook kernel. If the environment does not appear as a kernel, install and register `ipykernel`:

```powershell
python -m pip install ipykernel
python -m ipykernel install --user --name agentic-rag --display-name "Python (agentic-rag)"
```

Run the notebook cells from top to bottom. The embedding cell may download the Hugging Face model on its first run.

## Configure API keys

Copy `.env.example` to `.env` in the repository root, then add your keys:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
PINECONE_API_KEY=your_pinecone_api_key
TAVILY_API_KEY=your_tavily_api_key
```

The notebook checks common project and parent-directory locations for `.env`. If a key is not found, it prompts for it without echoing the value. Do not commit `.env` or paste API keys into notebook cells.

## Knowledge-base setup

The notebook currently loads the LangGraph Agentic RAG documentation page:

```text
https://docs.langchain.com/oss/python/langgraph/agentic-rag
```

It splits the loaded page into chunks, creates 384-dimensional embeddings, and indexes them in Pinecone. The notebook uses the index `industry-agentic-rag-kb` in the `langgraph-agentic-rag` namespace and creates the index in AWS `us-east-1` if it does not already exist. Pinecone usage may incur charges under your account's plan.

To use other material, update the notebook's source-loading section to load content you are authorized to use. Review data handling and access requirements before sending private or sensitive documents to any external service.

## Demo queries

The notebook includes examples for:

1. Answering an Agentic RAG question from the knowledge base.
2. Falling back to Tavily web search when the knowledge base is insufficient.
3. Answering a greeting directly without retrieval.
4. Asking about current or external information.

The exact route depends on the retrieved evidence and model grading, so a query titled as a knowledge-base demo may still take the web fallback path.

## Notes

- The graph makes multiple sequential model and service requests; response time depends on OpenRouter model availability, Pinecone, Tavily, network conditions, and whether the web fallback is used.
- The configured OpenRouter model is a free route and may be rate-limited or time out. A 504 from a provider is a failed request, not a successful answer.
- The notebook uses a public documentation page as its current example corpus. The phrase “private KB” describes the retrieval path, not the confidentiality of the sample source.
- The app files created by `create_project.py` are placeholders. Installing dependencies does not make the web app or API functional.

## Possible next steps

- Extract the notebook workflow into reusable Python modules.
- Add tests for routing, evidence grading, retries, and source tracking.
- Build a FastAPI service and a user interface around the graph.
- Add an authenticated document-ingestion workflow and operational logging.
- Document deployment, monitoring, and data-retention controls before production use.
