# Enterprise Research Assistant

A citation-first, enterprise-grade research assistant with a browser dashboard and a documented API.

## Web dashboard

Run `uvicorn app.main:app --reload`, then open `http://localhost:8000/`.
The dashboard lets you ingest documents, search the corpus, ask for cited answers (direct RAG or
the controlled agent), and see the newest Ragas evaluation summary. It uses the same `/api/v1`
endpoints as programmatic clients, avoiding a separate implementation of the RAG pipeline.

## Step 2: ingestion

The `POST /api/v1/documents/ingest` endpoint accepts PDF, TXT, and Markdown files.
It extracts page-aware text, splits it with LlamaIndex, creates local Ollama embeddings, and upserts
both chunks and their citation metadata into the persistent Chroma collection.

Before using the endpoint, start Ollama and download the configured model:

```text
ollama pull nomic-embed-text
```

Example request:

```text
curl -X POST http://localhost:8000/api/v1/documents/ingest -F "file=@paper.pdf"
```

The response contains a deterministic document ID and SHA-256 digest. Uploading the same file again
uses the same chunk IDs, so Chroma upserts rather than creating duplicate records.

To ingest every supported document in the `data` folder, with Ollama running:

```text
python -m app.ingestion.batch data
```

## Evaluation data

Place your ground-truth JSONL file at `data/evaluations/ground_truth.jsonl`.
See `data/evaluations/README.md` for the required fields. Evaluation is implemented in the Ragas step.

Run the Ragas evaluation after dependencies are installed and the API providers are configured:

```text
python -m app.evaluation.runner
```

The command runs the application on each ground-truth question and saves a timestamped,
machine-readable report in `data/evaluations/results/`. It scores faithfulness, factual correctness,
context precision, and context recall using Ragas collections metrics. The resolved dependency pin
includes a small compatibility bridge for an upstream Ragas/LangChain VertexAI import regression;
it is not used for DeepSeek evaluation calls.

Evaluation reports are cached by the exact SHA-256 content hash of
`data/evaluations/ground_truth.json`. Re-running the command with an unchanged dataset reuses its
completed report and makes no provider calls. Editing the ground-truth file creates a new hash, so
the next run automatically performs a fresh evaluation and saves a new report.

## Agent endpoint

`POST /api/v1/ask` is the direct, lowest-latency cited RAG endpoint. The controlled agent endpoint,
`POST /api/v1/agent/ask`, exposes only one read-only tool: `research_documents`. That tool calls the
LlamaIndex RAG tool and returns its cited answer unchanged. It has no web, shell, file-write, or database-write tools.

### Plan-and-execute agent

`POST /api/v1/agent/plan` adds a plan-and-execute research agent on top of the same citation-first
pipeline. It decomposes the user's question into up to `AGENT_PLAN_MAX_SUBQUESTIONS` sub-questions
via Gemini, answers each sub-question with the existing `ResearchAnswerService`, deduplicates
citations across sub-answers, and merges everything into one cited answer using DeepSeek. The
response extends `AnswerResponse` with `sub_questions` and `sub_answers`, so the plan is transparent
to the caller. Existing endpoints are unchanged.

```text
curl -X POST http://localhost:8000/api/v1/agent/plan \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare stroke classification accuracy across the indexed papers."}'
```

## Local setup

1. Copy `.env.example` to `.env` and set environment-specific values.
2. Create a Python 3.11–3.13 virtual environment.
3. Install: `pip install -e ".[dev]"`
4. Run: `uvicorn app.main:app --reload`
5. Open `http://localhost:8000/` for the dashboard, or `http://localhost:8000/docs` for API docs.

## Checks

Run `pytest` and `ruff check .` before committing.

## Dependency policy

Dependencies use compatible version ranges in `pyproject.toml`; CI will later produce a locked, reviewed dependency set. Production secrets belong in a secret manager, never source control.
