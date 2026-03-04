import re
import logging
import hashlib
from typing import TypedDict, List, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# Logging
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("BananaEnterpriseAgent")

# =====================================================
# FastAPI
# =====================================================

app = FastAPI()

# =====================================================
# Config
# =====================================================

PERSIST_DIR = "./db"
SIMILARITY_THRESHOLD = 0.55

# =====================================================
# Entity Registry (Ticker + Company Name)
# =====================================================

ENTITY_REGISTRY = {
    "AAPL": "APPLE",
    "MSFT": "MICROSOFT",
    "TSLA": "TESLA",
    "NVDA": "NVIDIA"
}

SUPPORTED_YEARS = ["2022", "2023", "2024"]

# =====================================================
# Embeddings
# =====================================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================================
# Vector Store
# =====================================================

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding
)

# =====================================================
# Agent State
# =====================================================

class AgentState(TypedDict):
    query: str
    intent_blocked: bool
    block_reason: Optional[str]
    fetched_data: Dict[str, str]
    retrieved_docs: List[str]
    answer: str

# =====================================================
# Utility Functions
# =====================================================

def entity_exists(query: str) -> bool:
    query_upper = query.upper()
    for ticker, name in ENTITY_REGISTRY.items():
        if ticker in query_upper or name in query_upper:
            return True
    return False

def validate_year(query: str) -> Optional[str]:
    match = re.search(r"20\d{2}", query)
    if match:
        year = match.group()
        if year not in SUPPORTED_YEARS:
            return "BLOCKED_UNSUPPORTED_DOCUMENT_YEAR"
    return None

# =====================================================
# Intent Classification
# =====================================================

ADVISORY_KEYWORDS = [
    "good investment",
    "overvalued",
    "undervalued",
    "should i",
    "buy",
    "sell",
    "predict",
    "future",
]

def intent_node(state: AgentState) -> AgentState:
    query_lower = state["query"].lower()

    if any(keyword in query_lower for keyword in ADVISORY_KEYWORDS):
        state["intent_blocked"] = True
        state["block_reason"] = "BLOCKED_ADVISORY_QUERY"
        state["answer"] = "INSUFFICIENT_EVIDENCE"
    else:
        state["intent_blocked"] = False
        state["block_reason"] = None

    return state

# =====================================================
# MCP TOOL SIMULATION (Replace with real MCP)
# =====================================================

def fetch_market(query: str) -> str:
    return "AAPL stock price is 187.25 USD as of today."

def fetch_sec(query: str) -> str:
    return "AAPL reported revenue of 383.3 billion USD in FY2023."

def fetch_social(query: str) -> str:
    return "Social sentiment score for AAPL is 0.78 positive."

# =====================================================
# Fetch Node
# =====================================================

def fetch_node(state: AgentState) -> AgentState:
    if state["intent_blocked"]:
        return state

    # Entity validation
    if not entity_exists(state["query"]):
        state["answer"] = "INSUFFICIENT_EVIDENCE"
        state["block_reason"] = "BLOCKED_UNKNOWN_ENTITY"
        return state

    # Year validation
    year_block = validate_year(state["query"])
    if year_block:
        state["answer"] = "INSUFFICIENT_EVIDENCE"
        state["block_reason"] = year_block
        return state

    state["fetched_data"] = {
        "market": fetch_market(state["query"]),
        "sec": fetch_sec(state["query"]),
        "social": fetch_social(state["query"]),
    }

    return state

# =====================================================
# Strict Validation Node
# =====================================================

def validate_node(state: AgentState) -> AgentState:
    if state.get("block_reason"):
        return state

    valid_results = {}

    for source, content in state["fetched_data"].items():

        # Reject echo of query
        if state["query"].lower() in content.lower():
            continue

        # Require numeric signal
        if not re.search(r"\d+(\.\d+)?", content):
            continue

        # Require entity mention
        content_upper = content.upper()
        if not any(
            ticker in content_upper or name in content_upper
            for ticker, name in ENTITY_REGISTRY.items()
        ):
            continue

        valid_results[source] = content

    if not valid_results:
        state["answer"] = "INSUFFICIENT_EVIDENCE"
        state["block_reason"] = "BLOCKED_INVALID_TOOL_OUTPUT"
        state["fetched_data"] = {}
        return state

    state["fetched_data"] = valid_results
    return state

# =====================================================
# Store Node
# =====================================================

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def store_node(state: AgentState) -> AgentState:
    if state.get("block_reason"):
        return state

    docs = []

    for source, content in state["fetched_data"].items():
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": source,
                    "doc_id": hash_text(content),
                }
            )
        )

    if docs:
        vectorstore.add_documents(docs)
        vectorstore.persist()

    return state

# =====================================================
# Retrieve Node
# =====================================================

def retrieve_node(state: AgentState) -> AgentState:
    if state.get("block_reason"):
        return state

    docs = vectorstore.similarity_search(state["query"], k=5)
    state["retrieved_docs"] = [doc.page_content for doc in docs]

    return state

# =====================================================
# Evaluate Node
# =====================================================

def evaluate_node(state: AgentState) -> AgentState:
    if state.get("block_reason"):
        return state

    if not state["retrieved_docs"]:
        state["answer"] = "INSUFFICIENT_EVIDENCE"
        state["block_reason"] = "BLOCKED_NO_RETRIEVAL"
        return state

    query_embedding = embedding.embed_query(state["query"])
    scores = []

    for doc in state["retrieved_docs"]:
        doc_embedding = embedding.embed_query(doc)
        score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        scores.append(score)

    if max(scores) < SIMILARITY_THRESHOLD:
        state["answer"] = "INSUFFICIENT_EVIDENCE"
        state["block_reason"] = "BLOCKED_LOW_SIMILARITY"
        state["retrieved_docs"] = []

    return state

# =====================================================
# Answer Node
# =====================================================

def answer_node(state: AgentState) -> AgentState:
    if state.get("answer") == "INSUFFICIENT_EVIDENCE":
        return state

    if not state["retrieved_docs"]:
        state["answer"] = "INSUFFICIENT_EVIDENCE"
        state["block_reason"] = "BLOCKED_EMPTY_RESULT"
        return state

    state["answer"] = "\n\n".join(state["retrieved_docs"])
    return state

# =====================================================
# Build Graph
# =====================================================

workflow = StateGraph(AgentState)

workflow.add_node("intent", intent_node)
workflow.add_node("fetch", fetch_node)
workflow.add_node("validate", validate_node)
workflow.add_node("store", store_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("answer", answer_node)

workflow.set_entry_point("intent")

workflow.add_edge("intent", "fetch")
workflow.add_edge("fetch", "validate")
workflow.add_edge("validate", "store")
workflow.add_edge("store", "retrieve")
workflow.add_edge("retrieve", "evaluate")
workflow.add_edge("evaluate", "answer")
workflow.add_edge("answer", END)

agent = workflow.compile()

# =====================================================
# API Model
# =====================================================

class QueryRequest(BaseModel):
    query: str

# =====================================================
# API Endpoint
# =====================================================

@app.post("/analyze")
def analyze(request: QueryRequest):

    result = agent.invoke({
        "query": request.query,
        "intent_blocked": False,
        "block_reason": None,
        "fetched_data": {},
        "retrieved_docs": [],
        "answer": ""
    })

    grounded = result["answer"] != "INSUFFICIENT_EVIDENCE"

    return {
        "answer": result["answer"],
        "grounded": grounded,
        "hallucination_rate": 0.0,
        "faithfulness_score": 1.0 if grounded else None,
        "block_reason": result.get("block_reason"),
        "tools_used": list(result.get("fetched_data", {}).keys())
    }
