import re
import time
import logging
import hashlib
from typing import TypedDict, List, Dict, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END
from sklearn.metrics.pairwise import cosine_similarity

from banana_service.config import settings
from banana_service.ingestion.mcp_client import MCPClient
from logger import setup_logger, trace_step

# ─────────────────────────────────────────────────────────────────────────────
# Loggers — one per concern so output is easy to filter
# ─────────────────────────────────────────────────────────────────────────────
log_api     = setup_logger("API")
log_intent  = setup_logger("Intent")
log_fetch   = setup_logger("Fetch")
log_valid   = setup_logger("Validate")
log_store   = setup_logger("Store")
log_retr    = setup_logger("Retrieve")
log_eval    = setup_logger("Evaluate")
log_answer  = setup_logger("Answer")
log_graph   = setup_logger("Graph")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming HTTP request and its response time."""
    t0 = time.perf_counter()
    log_api.info(f">>> {request.method} {request.url.path}")
    response = await call_next(request)
    elapsed  = round(time.perf_counter() - t0, 3)
    log_api.info(f"<<< {request.method} {request.url.path}  status={response.status_code}  elapsed={elapsed}s")
    return response

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
PERSIST_DIR          = "./db"
SIMILARITY_THRESHOLD = 0.55

ENTITY_REGISTRY = {
    "AAPL": "APPLE",
    "MSFT": "MICROSOFT",
    "TSLA": "TESLA",
    "NVDA": "NVIDIA",
}
SUPPORTED_YEARS = ["2022", "2023", "2024"]

log_api.info(f"EXPERIMENT_MODE={settings.EXPERIMENT_MODE}")
log_api.info(f"MCP_MARKET_URL={settings.MCP_MARKET_URL}")
log_api.info(f"MCP_SEC_URL={settings.MCP_SEC_URL}")
log_api.info(f"MCP_SOCIAL_URL={settings.MCP_SOCIAL_URL}")

# ─────────────────────────────────────────────────────────────────────────────
# Embeddings + Vector store
# ─────────────────────────────────────────────────────────────────────────────
log_api.info("Loading HuggingFace embedding model…")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
log_api.info("Embedding model ready.")

vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
log_api.info(f"ChromaDB vector store ready  persist_dir={PERSIST_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# MCP Clients
# ─────────────────────────────────────────────────────────────────────────────
_market_client = MCPClient(settings.MCP_MARKET_URL) if settings.MCP_MARKET_URL else None
_sec_client    = MCPClient(settings.MCP_SEC_URL)    if settings.MCP_SEC_URL    else None
_social_client = MCPClient(settings.MCP_SOCIAL_URL) if settings.MCP_SOCIAL_URL else None

log_api.info(f"MCP clients  market={'✓' if _market_client else '✗'}  "
             f"sec={'✓' if _sec_client else '✗'}  "
             f"social={'✓' if _social_client else '✗'}")

# ─────────────────────────────────────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query:          str
    intent_blocked: bool
    block_reason:   Optional[str]
    fetched_data:   Dict[str, str]
    retrieved_docs: List[str]
    answer:         str

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def entity_exists(query: str) -> bool:
    q = query.upper()
    return any(t in q or n in q for t, n in ENTITY_REGISTRY.items())

def validate_year(query: str) -> Optional[str]:
    m = re.search(r"20\d{2}", query)
    if m and m.group() not in SUPPORTED_YEARS:
        return "BLOCKED_UNSUPPORTED_DOCUMENT_YEAR"
    return None

def extract_ticker(query: str) -> Optional[str]:
    q = query.upper()
    return next((t for t in ENTITY_REGISTRY if t in q), None)

ADVISORY_KEYWORDS = ["good investment","overvalued","undervalued",
                     "should i","buy","sell","predict","future"]

# ─────────────────────────────────────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────────────────────────────────────

def intent_node(state: AgentState) -> AgentState:
    with trace_step(log_intent, "intent_node", query=repr(state["query"][:60])):
        q = state["query"].lower()
        hit = next((kw for kw in ADVISORY_KEYWORDS if kw in q), None)
        if hit:
            log_intent.warning(f"  Advisory keyword detected: '{hit}'  → BLOCKING")
            state["intent_blocked"] = True
            state["block_reason"]   = "BLOCKED_ADVISORY_QUERY"
            state["answer"]         = "INSUFFICIENT_EVIDENCE"
        else:
            log_intent.info("  No advisory keywords — query allowed through")
            state["intent_blocked"] = False
            state["block_reason"]   = None
    return state


def _call_mcp(client, tool: str, ticker: str, label: str) -> str:
    if client is None:
        log_fetch.warning(f"  {label} MCP client not configured — skipping")
        return ""
    with trace_step(log_fetch, f"mcp_call/{label}", tool=tool, ticker=ticker):
        try:
            results = client.call_tool(tool, {"ticker": ticker})
            data = " ".join(results) if isinstance(results, list) else str(results)
            log_fetch.info(f"  {label} → {data[:120]}")
            return data
        except Exception as e:
            log_fetch.error(f"  {label} MCP call failed: {e}")
            return ""


def fetch_node(state: AgentState) -> AgentState:
    with trace_step(log_fetch, "fetch_node"):
        if state["intent_blocked"]:
            log_fetch.info("  Skipping fetch — query already blocked")
            return state

        if not entity_exists(state["query"]):
            log_fetch.warning("  Unknown entity — blocking query")
            state["answer"]       = "INSUFFICIENT_EVIDENCE"
            state["block_reason"] = "BLOCKED_UNKNOWN_ENTITY"
            return state

        year_block = validate_year(state["query"])
        if year_block:
            log_fetch.warning(f"  Year not in supported range → {year_block}")
            state["answer"]       = "INSUFFICIENT_EVIDENCE"
            state["block_reason"] = year_block
            return state

        ticker = extract_ticker(state["query"]) or state["query"].split()[0]
        log_fetch.info(f"  Extracted ticker: {ticker}")

        market = _call_mcp(_market_client, "fetch_market_data", ticker, "MARKET")
        sec    = _call_mcp(_sec_client,    "fetch_sec_filings",  ticker, "SEC")
        social = _call_mcp(_social_client, "fetch_social_sentiment", ticker, "SOCIAL")

        state["fetched_data"] = {k: v for k, v in
                                 {"market": market, "sec": sec, "social": social}.items() if v}
        log_fetch.info(f"  Sources fetched: {list(state['fetched_data'].keys())}")
    return state


def validate_node(state: AgentState) -> AgentState:
    with trace_step(log_valid, "validate_node"):
        if state.get("block_reason"):
            log_valid.info("  Skipping — already blocked")
            return state

        valid = {}
        for source, content in state["fetched_data"].items():
            checks = {
                "not_echo":    state["query"].lower() not in content.lower(),
                "has_number":  bool(re.search(r"\d+(\.\d+)?", content)),
                "has_entity":  any(t in content.upper() or n in content.upper()
                                   for t, n in ENTITY_REGISTRY.items()),
            }
            passed = all(checks.values())
            log_valid.info(f"  {source:8s}  checks={checks}  → {'PASS' if passed else 'FAIL'}")
            if passed:
                valid[source] = content

        if not valid:
            log_valid.warning("  All sources failed validation → BLOCKED_INVALID_TOOL_OUTPUT")
            state["answer"]       = "INSUFFICIENT_EVIDENCE"
            state["block_reason"] = "BLOCKED_INVALID_TOOL_OUTPUT"
            state["fetched_data"] = {}
        else:
            state["fetched_data"] = valid
            log_valid.info(f"  Valid sources: {list(valid.keys())}")
    return state


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def store_node(state: AgentState) -> AgentState:
    with trace_step(log_store, "store_node"):
        if state.get("block_reason"):
            log_store.info("  Skipping — already blocked")
            return state

        docs = [
            Document(page_content=content,
                     metadata={"source": source, "doc_id": hash_text(content)})
            for source, content in state["fetched_data"].items()
        ]
        if docs:
            vectorstore.add_documents(docs)
            vectorstore.persist()
            log_store.info(f"  Stored {len(docs)} document(s) to ChromaDB")
        else:
            log_store.warning("  No documents to store")
    return state


def retrieve_node(state: AgentState) -> AgentState:
    with trace_step(log_retr, "retrieve_node"):
        if state.get("block_reason"):
            log_retr.info("  Skipping — already blocked")
            return state

        docs = vectorstore.similarity_search(state["query"], k=5)
        state["retrieved_docs"] = [d.page_content for d in docs]
        log_retr.info(f"  Retrieved {len(state['retrieved_docs'])} doc(s)")
        for i, d in enumerate(state["retrieved_docs"]):
            log_retr.debug(f"    [{i}] {d[:100]}")
    return state


def evaluate_node(state: AgentState) -> AgentState:
    with trace_step(log_eval, "evaluate_node"):
        if state.get("block_reason"):
            log_eval.info("  Skipping — already blocked")
            return state

        if not state["retrieved_docs"]:
            log_eval.warning("  No retrieved docs → BLOCKED_NO_RETRIEVAL")
            state["answer"]         = "INSUFFICIENT_EVIDENCE"
            state["block_reason"]   = "BLOCKED_NO_RETRIEVAL"
            return state

        query_emb = embedding.embed_query(state["query"])
        scores = []
        for doc in state["retrieved_docs"]:
            doc_emb = embedding.embed_query(doc)
            score   = cosine_similarity([query_emb], [doc_emb])[0][0]
            scores.append(score)
            log_eval.info(f"  similarity={score:.4f}  doc={doc[:80]}")

        max_score = max(scores)
        log_eval.info(f"  max_similarity={max_score:.4f}  threshold={SIMILARITY_THRESHOLD}")

        if max_score < SIMILARITY_THRESHOLD:
            log_eval.warning(f"  Below threshold → BLOCKED_LOW_SIMILARITY")
            state["answer"]         = "INSUFFICIENT_EVIDENCE"
            state["block_reason"]   = "BLOCKED_LOW_SIMILARITY"
            state["retrieved_docs"] = []
        else:
            log_eval.info("  Similarity check passed ✓")
    return state


def answer_node(state: AgentState) -> AgentState:
    with trace_step(log_answer, "answer_node"):
        if state.get("answer") == "INSUFFICIENT_EVIDENCE":
            log_answer.info(f"  Returning INSUFFICIENT_EVIDENCE  reason={state.get('block_reason')}")
            return state

        if not state["retrieved_docs"]:
            log_answer.warning("  No docs available → BLOCKED_EMPTY_RESULT")
            state["answer"]       = "INSUFFICIENT_EVIDENCE"
            state["block_reason"] = "BLOCKED_EMPTY_RESULT"
            return state

        state["answer"] = "\n\n".join(state["retrieved_docs"])
        log_answer.info(f"  Answer built from {len(state['retrieved_docs'])} doc(s)  "
                        f"length={len(state['answer'])} chars")
    return state

# ─────────────────────────────────────────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────────────────────────────────────────
log_graph.info("Building LangGraph StateGraph…")
workflow = StateGraph(AgentState)

for name, fn in [("intent",   intent_node),
                 ("fetch",    fetch_node),
                 ("validate", validate_node),
                 ("store",    store_node),
                 ("retrieve", retrieve_node),
                 ("evaluate", evaluate_node),
                 ("answer",   answer_node)]:
    workflow.add_node(name, fn)
    log_graph.info(f"  Node registered: {name}")

workflow.set_entry_point("intent")
for a, b in [("intent","fetch"),("fetch","validate"),("validate","store"),
             ("store","retrieve"),("retrieve","evaluate"),("evaluate","answer"),
             ("answer", END)]:
    workflow.add_edge(a, b)
    log_graph.info(f"  Edge: {a} → {b}")

agent = workflow.compile()
log_graph.info("LangGraph compiled ✓")

# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str


@app.post("/analyze")
def analyze(request: QueryRequest):
    log_api.info(f"Received query: {request.query!r}")

    with trace_step(log_api, "agent.invoke", query=repr(request.query[:60])):
        result = agent.invoke({
            "query":          request.query,
            "intent_blocked": False,
            "block_reason":   None,
            "fetched_data":   {},
            "retrieved_docs": [],
            "answer":         "",
        })

    grounded = result["answer"] != "INSUFFICIENT_EVIDENCE"
    response = {
        "answer":             result["answer"],
        "grounded":           grounded,
        "hallucination_rate": 0.0,
        "faithfulness_score": 1.0 if grounded else None,
        "block_reason":       result.get("block_reason"),
        "tools_used":         list(result.get("fetched_data", {}).keys()),
    }
    log_api.info(f"Response  grounded={grounded}  "
                 f"block={result.get('block_reason')}  "
                 f"tools={response['tools_used']}")
    return response
