# Banana Financial Intelligence Service

A production-grade Retrieval-Augmented Generation (RAG) system for grounded financial query answering. Deployed on Kubernetes via Minikube, the service integrates a multi-node LangGraph pipeline, three MCP data sources, FinBERT-based sentiment analysis, and a confidence-gated reflection mechanism.

> **April 1, 2026 evaluation:** 0% hallucination rate on live API, 88% of adversarial queries correctly blocked, 73.5% FinBERT sentiment accuracy on 49 labeled samples.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Services](#services)
- [Pipeline Design](#pipeline-design)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Deployment](#deployment)
- [Known Issues](#known-issues)

---

## Architecture Overview

The system is built around three layers:

```
<img width="887" height="459" alt="Screenshot 2026-04-02 at 1 39 52 PM" src="https://github.com/user-attachments/assets/1e96f2ae-dd71-4c01-853b-e8cd64792bbb" />

┌─────────────────────────────────────────────────────┐
│  Outer Peel — MCP Data Sources                      │
│  banana-market (Alpha Vantage)                      │
│  banana-sec    (SEC EDGAR)                          │
│  banana-social (Twitter/X sentiment)                │
├─────────────────────────────────────────────────────┤
│  Pulp — LangGraph Agentic Pipeline                  │
│  intent → fetch → validate → store → retrieve       │
│  → ticker_guard → evaluate → answer                 │
│  → analyst (FinBERT) → reflection → scribe          │
├─────────────────────────────────────────────────────┤
│  Core — Vector Store + Model Backbone               │
│  ChromaDB (persistent)  +  all-MiniLM-L6-v2         │
└─────────────────────────────────────────────────────┘
```

Four microservices run as independent Kubernetes pods:

| Service | Image | Port | Role |
|---|---|---|---|
| `banana-api` | `banana-api:v2` | 8000 | FastAPI + LangGraph orchestration, FinBERT, ChromaDB |
| `banana-market` | `banana-market:v1` | 8003 | Alpha Vantage real-time OHLCV via MCP |
| `banana-sec` | `banana-sec:v1` | 8001 | SEC EDGAR 10-K/10-Q filings via MCP |
| `banana-social` | `banana-social-mcp:v1` | 8003 | Social sentiment (VADER / Twitter/X) via MCP |

---

## Project Structure

```
.
├── banana_service/                  # Main application package
│   ├── main.py                      # FastAPI app + LangGraph graph + all pipeline nodes
│   ├── baseline_model.py            # Plain LLM pipeline (BASELINE experiment mode)
│   ├── optimized_pipeline.py        # RAG pipeline without LangGraph (legacy)
│   ├── logger.py                    # Structured logging setup
│   ├── agents/
│   │   ├── analyst.py               # FinBERT sentiment agent (AnalystAgent)
│   │   ├── reflection.py            # Confidence gate agent (ReflectionAgent)
│   │   ├── scribe.py                # Report formatter (ScribeAgent)
│   │   ├── researcher.py            # MCP fetch + vector retrieval (benchmark use)
│   │   └── orchestrator.py          # Unused — logic absorbed into main.py
│   ├── core/
│   │   ├── embedding_model.py       # Sentence transformer wrapper (all-MiniLM-L6-v2)
│   │   └── vector_store.py          # In-memory FAISS vector store (benchmark use)
│   ├── evaluation/
│   │   └── hallucination.py         # Sentence-overlap hallucination metric
│   └── ingestion/
│       └── mcp_client.py            # JSON-RPC 2.0 MCP client
│
├── mcp_servers/                     # MCP server implementations
│   ├── market_server.py             # Alpha Vantage TIME_SERIES_DAILY endpoint
│   ├── sec_server.py                # SEC EDGAR public REST API
│   └── social_server.py             # Social sentiment (placeholder / Twitter/X)
│
├── db/                              # ChromaDB persistent storage (gitignored)
│
├── # Kubernetes manifests
├── banana-api-deployment.yaml
├── banana-api-service.yaml
├── banana-market-deployment.yaml
├── banana-market-service.yaml
├── banana-sec-deployment.yaml
├── banana-sec-service.yaml
├── banana-social-deployment.yaml
├── banana-social-service.yaml
├── banana-configmap.yaml
│
├── Dockerfile                       # banana-api container image
├── docker-compose.yaml              # Local Docker alternative to Kubernetes
├── deploy.sh                        # Full Minikube deployment automation
├── requirements.txt                 # Python dependencies
│
├── # Test and benchmark scripts
├── test_service.py                  # Integration test suite (pipeline classification)
├── test_service-sentiment.py        # FinBERT sentiment labeled dataset tests
├── benchmark.py                     # BASELINE vs OPTIMIZED comparison benchmark
├── benchmark_page.py                # Benchmark chart generation
├── test_service_page.py             # Test service chart generation
│
├── # Data and reports
├── all-data.csv                     # Labeled FinBERT evaluation dataset
├── output.txt                       # Latest run output log
└── report.txt / report-mar8.txt     # Historical evaluation reports
```

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12+ | |
| Docker | 24+ | Must be running |
| Minikube | 1.32+ | `brew install minikube` |
| kubectl | 1.28+ | `brew install kubectl` |
| Alpha Vantage API key | — | [Free tier](https://www.alphavantage.co/support/#api-key): 25 req/day |

Optional for social sentiment:
- Twitter/X Bearer Token (free developer account)

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd source-code
export ALPHA_VANTAGE_API_KEY=your_key_here
# optional:
export TWITTER_BEARER_TOKEN=your_token_here
```

### 2. Deploy to Minikube (full automated)

```bash
chmod +x deploy.sh
./deploy.sh
```

This script:
- Starts Minikube if not running
- Switches Docker daemon to Minikube's environment
- Builds all four images inside Minikube
- Creates Kubernetes Secrets for API keys
- Applies all manifests
- Monitors rollout status

### 3. Access the API

```bash
# Recommended — stable localhost URL
kubectl port-forward service/banana-api-service 8080:8000

# Test it
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize recent SEC filing for AAPL."}'
```

### 4. Run locally with Docker Compose (no Kubernetes)

```bash
docker-compose up --build
# API available at http://localhost:8000
```

---

## Services

### banana-api

The core service. Hosts the FastAPI application and runs the full LangGraph pipeline. Exposes a single `POST /analyze` endpoint.

```bash
# Inside the pod
kubectl exec -it deployment/banana-api -- bash
```

Resource limits: `memory: 2Gi`, `cpu: 500m`

### banana-market

Wraps the Alpha Vantage `TIME_SERIES_DAILY` endpoint as an MCP server. Returns the most recent trading day's OHLCV data for a given ticker.

```bash
# Direct test (from inside cluster)
curl -X POST http://banana-market:8003/mcp \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "get_market_data", "arguments": {"ticker": "AAPL"}}'
```

**Rate limit:** Free tier = 25 req/day, 5 req/min. The pipeline does not implement retry or backoff on rate limit hits — silent failures will occur beyond the daily cap.

### banana-sec

Wraps the SEC EDGAR public REST API. Resolves ticker → CIK → filing metadata. Returns up to 5 recent 10-K/10-Q filings. No API key required.

### banana-social

Social sentiment MCP server. Production behavior depends on whether a Twitter/X Bearer Token is configured:

- **With token:** Queries recent tweets via Twitter/X API v2, scores with VADER, returns aggregate sentiment.
- **Without token:** Returns a graceful degradation message (no data). The `not_placeholder` validation check in `validate_node` ensures fabricated sentiment data cannot slip through as grounding evidence.

---

## Pipeline Design

The LangGraph pipeline runs 11 nodes in sequence. Each node either enriches the `AgentState` or sets a `block_reason` that causes all downstream nodes to skip.

```
intent_node → fetch_node → validate_node → store_node → retrieve_node
→ ticker_guard_node → evaluate_node → answer_node
→ analyst_node → reflection_node → scribe_node
```

### Guard nodes

| Node | Block reason | What it catches |
|---|---|---|
| `intent_node` | `BLOCKED_ADVISORY_QUERY` | Investment advice keywords: buy, sell, predict, overvalued, should i… |
| `fetch_node` | `BLOCKED_UNKNOWN_ENTITY` | Tickers/companies not in `ENTITY_REGISTRY` (AAPL, MSFT, TSLA, NVDA) |
| `fetch_node` | `BLOCKED_UNSUPPORTED_DOCUMENT_YEAR` | Years outside `[2022, 2023, 2024]` |
| `fetch_node` | `BLOCKED_STALE_DOCUMENTS` | ChromaDB documents older than the staleness threshold |
| `validate_node` | `BLOCKED_INVALID_TOOL_OUTPUT` | MCP responses failing any of 4 quality checks |
| `ticker_guard_node` | `BLOCKED_TICKER_MISMATCH` | Retrieved docs that don't mention the queried ticker |
| `evaluate_node` | `BLOCKED_LOW_SIMILARITY` | Cosine similarity below `SIMILARITY_THRESHOLD` (0.55) |

### Agents

**AnalystAgent** (`agents/analyst.py`) — runs `ProsusAI/finbert` on the assembled answer to produce a sentiment label (`positive` / `negative` / `neutral`) and confidence score. Loaded once at startup as a singleton.

**ReflectionAgent** (`agents/reflection.py`) — compares FinBERT confidence against `CONFIDENCE_THRESHOLD` (0.70). Routes to `scribe_node` if confidence ≥ threshold, otherwise terminates (`report = None`).

**ScribeAgent** (`agents/scribe.py`) — formats a structured Financial Report from the sentiment output. Extensible — report template accepts answer excerpts, tool sources, and timestamps.

### AgentState schema

```python
class AgentState(TypedDict):
    query: str                        # Original user query
    intent_blocked: bool              # True if advisory keywords detected
    block_reason: Optional[str]       # Reason code if pipeline halted
    fetched_data: Dict[str, str]      # Raw MCP responses keyed by source
    retrieved_docs: List[str]         # Docs retrieved from ChromaDB
    answer: str                       # Final answer text
    threshold: float                  # FinBERT confidence gate (default 0.70)
    sentiment: Optional[Dict]         # {label, confidence} from FinBERT
    proceed: Optional[bool]           # Set by ReflectionAgent
    report: Optional[str]             # Final report from ScribeAgent
```

---

## API Reference

### POST /analyze

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AAPL stock price?"}'
```

**Request body:**

```json
{ "query": "string" }
```

**Response:**

```json
{
  "answer": "AAPL (2026-04-01) — Open: $174.23, High: $176.01, Low: $173.88, Close: $175.42, Volume: 52341200",
  "grounded": true,
  "hallucination_rate": 0.0,
  "faithfulness_score": 1.0,
  "block_reason": null,
  "tools_used": ["market", "sec", "social"],
  "sentiment": {
    "label": "neutral",
    "confidence": 0.931
  },
  "report": "\nFinancial Report:\n\nSentiment: neutral\nConfidence: 0.931\n"
}
```

**Blocked response example:**

```json
{
  "answer": "INSUFFICIENT_EVIDENCE",
  "grounded": false,
  "hallucination_rate": 0.0,
  "faithfulness_score": null,
  "block_reason": "BLOCKED_ADVISORY_QUERY",
  "tools_used": [],
  "sentiment": null,
  "report": null
}
```

| Field | Type | Description |
|---|---|---|
| `answer` | string | Retrieved document text or `INSUFFICIENT_EVIDENCE` |
| `grounded` | boolean | True if answer is not `INSUFFICIENT_EVIDENCE` |
| `hallucination_rate` | float | Sentence-overlap score — 0.0 = fully grounded |
| `faithfulness_score` | float \| null | 1.0 = fully faithful; null if blocked |
| `block_reason` | string \| null | Guard that halted the pipeline |
| `tools_used` | string[] | MCP sources that returned valid data |
| `sentiment` | object \| null | FinBERT result: `{label, confidence}` |
| `report` | string \| null | ScribeAgent report; null if confidence < threshold |

---

## Configuration

All runtime configuration is managed through Kubernetes ConfigMaps and Secrets. Set at pod startup — not per-request.

| Variable | Source | Default | Description |
|---|---|---|---|
| `EXPERIMENT_MODE` | ConfigMap `banana-config` | `OPTIMIZED` | `OPTIMIZED` = RAG + agents, `BASELINE` = plain LLM |
| `MCP_MARKET_URL` | Deployment env | `http://banana-market:8003/mcp` | Market data MCP endpoint |
| `MCP_SEC_URL` | Deployment env | `http://banana-sec:8001/mcp` | SEC filings MCP endpoint |
| `MCP_SOCIAL_URL` | Deployment env | `http://banana-social:8003/mcp` | Social sentiment MCP endpoint |
| `ALPHA_VANTAGE_API_KEY` | Secret `alpha-vantage-secret` | — | Required for market data |
| `TWITTER_BEARER_TOKEN` | Secret (optional) | — | Required for live social sentiment |
| `CONFIDENCE_THRESHOLD` | `main.py` constant | `0.70` | FinBERT minimum confidence for report generation |
| `SIMILARITY_THRESHOLD` | `main.py` constant | `0.55` | ChromaDB minimum cosine similarity to accept a document |

To switch the running pipeline to BASELINE mode:

```bash
kubectl edit configmap banana-config
# change EXPERIMENT_MODE to BASELINE
kubectl rollout restart deployment/banana-api
```

---

## Testing

### Integration test suite

Tests the live `/analyze` API across 6 query categories (17 pipeline queries + 49 FinBERT sentiment samples).

```bash
# Start the API first
kubectl port-forward service/banana-api-service 8080:8000 &

# Run pipeline classification tests
python test_service.py

# Run FinBERT sentiment labeled tests
python test_service-sentiment.py
```

**Query categories:**

| Category | Count | Expected behaviour |
|---|---|---|
| `FACTUAL` | 4 | Grounded answer with sentiment and report |
| `ADVISORY` | 4 | Blocked — `BLOCKED_ADVISORY_QUERY` |
| `NONEXISTENT` | 3 | Blocked — `BLOCKED_UNKNOWN_ENTITY` |
| `FABRICATED` | 2 | Blocked — `BLOCKED_UNSUPPORTED_DOCUMENT_YEAR` |
| `CONFIDENTIAL` | 2 | Blocked — `BLOCKED_INVALID_TOOL_OUTPUT` |
| `HAL_PROBE` | 3 | Should block; if grounded = active vulnerability |

**April 1, 2026 results (live API):**

- 15/17 pipeline queries correctly blocked
- 1/3 hallucination probes leaked (MSFT revenue — temporal mismatch; see [Known Issues](#known-issues))
- FinBERT sentiment: 73.47% accuracy on 49 labeled samples

### Generate test charts

```bash
python test_service_page.py
# outputs: test_01_*.png through test_06_*.png
```

---

## Benchmarking

Compares `BASELINE` (plain TinyLlama, no retrieval) against `OPTIMIZED` (RAG + agentic pipeline) across hallucination rate, faithfulness, grounded rate, and latency.

```bash
# Against live API (recommended — tests guard chain)
python benchmark.py --url http://localhost:8080

# In-process (bypasses guards — metrics are misleading for blocking behaviour)
python benchmark.py
```

> **Important:** Always use `--url` when benchmarking guard behaviour. The in-process runner bypasses `intent_node`, `fetch_node`, and other API guards entirely. All 18 in-process queries reach `grounded=True` even for ADVISORY and NONEXISTENT categories.

**April 1, 2026 benchmark summary (in-process, MPS device):**

| Metric | Baseline | Optimized | Winner |
|---|---|---|---|
| Hallucination rate (avg) | 13.1% | 33.0% | Baseline\* |
| Faithfulness score (avg) | 86.9% | 67.0% | Baseline\* |
| Grounded responses | 0% | 100% | Optimized |
| Avg latency (s) | 178.9 | 159.4 | Optimized |

\* *The baseline "winning" on hallucination/faithfulness is a measurement artefact — the evaluator measures self-overlap of generic LLM text. The only meaningful metric is Grounded Responses: OPTIMIZED 100%, BASELINE 0%.*

### Generate benchmark charts

```bash
python benchmark_page.py
# outputs: benchmark_01_*.png through benchmark_06_*.png
```

---

## Deployment

### Full deploy (automated)

```bash
export ALPHA_VANTAGE_API_KEY=your_key_here
./deploy.sh
```

The script handles: Minikube startup, Docker daemon setup, image builds, Secret creation, manifest application, and rollout monitoring.

### Manual steps

```bash
# 1. Point Docker to Minikube
eval $(minikube docker-env)

# 2. Build images
docker build -t banana-api:v2 .
docker build -t banana-market:v1 ./mcp_servers/market_server_image/
docker build -t banana-sec:v1    ./mcp_servers/sec_server_image/
docker build -t banana-social-mcp:v1 ./mcp_servers/social_server_image/

# 3. Create secrets
kubectl create secret generic alpha-vantage-secret \
  --from-literal=ALPHA_VANTAGE_API_KEY=$ALPHA_VANTAGE_API_KEY

# 4. Apply manifests
kubectl apply -f banana-configmap.yaml
kubectl apply -f banana-api-deployment.yaml
kubectl apply -f banana-api-service.yaml
kubectl apply -f banana-market-deployment.yaml
kubectl apply -f banana-market-service.yaml
kubectl apply -f banana-sec-deployment.yaml
kubectl apply -f banana-sec-service.yaml
kubectl apply -f banana-social-deployment.yaml
kubectl apply -f banana-social-service.yaml

# 5. Check status
kubectl get pods
kubectl rollout status deployment/banana-api
```

### Access on macOS (Docker driver)

The NodePort is not directly accessible on macOS with the Docker driver. Use port-forward instead:

```bash
kubectl port-forward service/banana-api-service 8080:8000
# API now at http://localhost:8080
```

### Teardown

```bash
kubectl delete -f .
minikube stop
```

---

## Known Issues

### 1. MSFT revenue temporal mismatch (active vulnerability)

**Symptom:** Querying MSFT revenue for a specific historical year (e.g. FY2023) returns today's OHLCV price data as a grounded answer.

**Root cause:** No date-range metadata filter on ChromaDB retrieval. The ticker guard confirms the document mentions MSFT; the similarity gate accepts it. But the document is today's price data, not historical revenue.

**Fix:** Add a year filter to `retrieve_node`:

```python
# In retrieve_node, extract year from query and filter:
results = chroma_collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"year": {"$eq": extracted_year}}   # add this
)
```

### 2. BLOCKED_STALE_DOCUMENTS over-triggers

**Symptom:** Valid FACTUAL queries (AAPL stock price, AAPL FY2023 revenue) are blocked by the stale document guard even when fresh data is available.

**Fix:** Tune the staleness threshold in `main.py`. Current threshold may be too aggressive.

### 3. Benchmark in-process mode bypasses guards

**Symptom:** Running `benchmark.py` without `--url` shows 0 blocked queries across all categories, making OPTIMIZED appear to "fail" on ADVISORY/NONEXISTENT queries.

**Fix:** Always use `python benchmark.py --url http://localhost:8080`.

### 4. Alpha Vantage silent failure after rate limit

**Symptom:** After 25 API calls/day, `banana-market` returns empty or error responses with no warning to the caller.

**Workaround:** Monitor call count manually. No retry/backoff is implemented.

### 5. FinBERT misclassifies implicit positives and social media text

**Symptom:** 73.47% accuracy on labeled dataset; specific failures on M&A headlines, ironic financial phrasing, and short social posts.

**Workaround:** For inputs < 15 tokens, consider supplementing with VADER compound score as a tiebreaker.

---

## Technology Stack

| Component | Technology |
|---|---|
| API framework | FastAPI 0.110+ |
| Pipeline orchestration | LangGraph 0.1+ |
| Vector store | ChromaDB 0.5+ (persistent) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Sentiment model | `ProsusAI/finbert` |
| LLM (baseline mode) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Data protocol | MCP (Model Context Protocol) JSON-RPC 2.0 |
| Market data | Alpha Vantage API |
| SEC data | SEC EDGAR public REST API |
| Deployment | Kubernetes / Minikube |
| Containerisation | Docker |

---

## Data Sources

All external data is fetched live at query time — there is no pre-built static corpus.

| Source | Server | Auth required | Volume per query |
|---|---|---|---|
| Stock market OHLCV | `banana-market` | Alpha Vantage API key | 1 record (most recent trading day) |
| SEC EDGAR filings | `banana-sec` | None | Up to 5 recent 10-K/10-Q filings |
| Social sentiment | `banana-social` | Twitter/X Bearer Token (optional) | Up to 20 tweets per ticker |

Supported tickers (ENTITY_REGISTRY): `AAPL`, `MSFT`, `TSLA`, `NVDA`

Supported document years: `2022`, `2023`, `2024`

---


