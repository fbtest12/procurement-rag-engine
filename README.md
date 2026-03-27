# Procurement RAG Engine

An AI-powered Retrieval-Augmented Generation system for government procurement document Q&A. Built to demonstrate production-grade RAG architecture for processing RFPs, bid documents, compliance regulations, and procurement policy manuals.

## What It Does

Government procurement teams deal with thousands of pages of RFPs, bid packages, compliance rules, and policy manuals. This system ingests those documents and lets users ask natural language questions like:

- "What are the submission requirements for the IT services RFP?"
- "What DBE participation goals apply to federally funded projects?"
- "What insurance coverage is required for the bridge rehabilitation bid?"

The engine retrieves the most relevant document sections and generates grounded, cited answers — no hallucination, no guessing.

## Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│   FastAPI     │────▶│  RAG Engine   │────▶│  LLM Provider    │
│   Service     │     │  (orchestrator)│     │  (OpenAI/Claude) │
└──────┬───────┘     └───────┬───────┘     └──────────────────┘
       │                     │
       │              ┌──────▼───────┐
       │              │ Vector Store  │
       │              │  (ChromaDB)   │
       │              └──────────────┘
       │
┌──────▼───────────────────────────┐
│       Ingestion Pipeline          │
│  Loader → Chunker → Embeddings   │
└──────────────────────────────────┘
```

### Key Design Decisions

**Provider-agnostic LLM layer** — The `LLMProvider` abstract base class means swapping from OpenAI to Anthropic (or Azure, Bedrock, a local model) is a config change, not a code change. The factory pattern keeps concrete providers out of business logic.

**Multiple chunking strategies** — Procurement documents have strong structural cues (numbered sections, capitalized headers). The `semantic_sections` strategy exploits this structure, while `recursive` and `fixed_size` serve as reliable fallbacks. The strategy is configurable per-ingestion, which matters when you're processing both structured RFPs and freeform policy memos.

**Evaluation pipeline built in** — RAG systems degrade silently. The evaluator scores retrieval relevance, answer faithfulness, and source accuracy across procurement-specific test cases. This runs in CI to catch regressions when you change chunking parameters or swap models.

**Domain-aware query preprocessing** — Government procurement is full of acronyms (RFP, DBE, MBE, SOW, IFB). The engine expands these during retrieval to improve recall without requiring users to know the full terms.

## Project Structure

```
procurement-rag-engine/
├── src/
│   ├── llm/                    # Provider-agnostic LLM abstraction
│   │   ├── base.py             # Abstract LLMProvider + LLMResponse
│   │   ├── openai_provider.py  # OpenAI GPT implementation
│   │   ├── anthropic_provider.py # Anthropic Claude implementation
│   │   └── factory.py          # Provider factory
│   ├── ingestion/              # Document processing pipeline
│   │   ├── loader.py           # Multi-format document loader (PDF, DOCX, TXT)
│   │   ├── chunker.py          # Chunking strategies (fixed, recursive, semantic)
│   │   └── pipeline.py         # End-to-end ingestion orchestrator
│   ├── vectorstore/            # Vector storage abstraction
│   │   ├── store.py            # Abstract VectorStore interface
│   │   └── chroma_store.py     # ChromaDB implementation
│   ├── rag/                    # Core RAG engine
│   │   └── engine.py           # Retrieve → Generate pipeline with instrumentation
│   ├── evaluation/             # Quality assurance
│   │   └── evaluator.py        # Multi-dimensional RAG scoring
│   └── api/                    # REST API layer
│       ├── app.py              # FastAPI application with full CRUD
│       └── config.py           # Typed settings via pydantic-settings
├── data/
│   └── sample_docs/            # Sample procurement documents for demo
├── tests/                      # Unit tests
├── scripts/
│   ├── ingest_sample_docs.py   # Batch ingestion script
│   └── run_eval.py             # Evaluation runner
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/procurement-rag-engine.git
cd procurement-rag-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — default is OpenRouter (free, no credit card needed)
# Get your free API key at https://openrouter.ai/keys
# Set OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 3. Ingest sample documents

```bash
python -m scripts.ingest_sample_docs
```

### 4. Start the API

```bash
python main.py
# Or with Docker:
docker compose up --build
```

### 5. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the DBE participation requirements?"}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System status and vector store stats |
| `POST` | `/query` | Ask a question about ingested documents |
| `POST` | `/ingest/file` | Ingest a single document |
| `POST` | `/ingest/directory` | Batch ingest a directory |
| `GET` | `/documents` | List all ingested sources |
| `DELETE` | `/documents/{source}` | Remove a document's chunks |

## Running Tests

```bash
pytest
```

## Running Evaluation

```bash
# Requires documents to be ingested first
python -m scripts.run_eval --output eval_results.json
```

The evaluator scores across four dimensions: retrieval relevance, answer faithfulness, answer relevance, and source accuracy, with a weighted composite score and per-category breakdowns.

## Configuration

All settings are environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openrouter` | LLM backend (`openrouter`, `openai`, or `anthropic`) |
| `LLM_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | Model identifier |
| `OPENROUTER_API_KEY` | | Free key from https://openrouter.ai/keys |
| `CHUNKING_STRATEGY` | `recursive` | `fixed_size`, `recursive`, or `semantic_sections` |
| `CHUNK_SIZE` | `1000` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `DEFAULT_TOP_K` | `5` | Chunks retrieved per query |
| `DEFAULT_SCORE_THRESHOLD` | `0.3` | Minimum cosine similarity |

## What I'd Add Next

These are deliberate scope boundaries for an MVP, not oversights:

- **Hybrid search** — Combine dense vector retrieval with BM25 sparse retrieval for better recall on exact terms (contract numbers, dollar amounts)
- **Re-ranking** — Add a cross-encoder re-ranker (e.g., Cohere Rerank or a local cross-encoder) between retrieval and generation for precision
- **Streaming responses** — SSE streaming for the `/query` endpoint so users see answers as they generate
- **Document versioning** — Track document versions so updated RFPs don't leave stale chunks in the store
- **Agentic workflows** — Multi-step reasoning for complex queries like "Compare the insurance requirements across all active RFPs"
- **Observability** — OpenTelemetry integration for tracing the full retrieve-generate pipeline in production
- **Authentication** — API key or OAuth2 middleware for multi-tenant deployment

## Tech Stack

- **Python 3.11** — Core language
- **FastAPI** — Async REST API framework
- **ChromaDB** — Embedded vector database (swappable via abstract interface)
- **OpenAI / Anthropic** — LLM providers (provider-agnostic architecture)
- **PyMuPDF** — PDF text extraction
- **sentence-transformers** — Document embeddings
- **Docker** — Containerized deployment
- **pytest** — Testing framework
