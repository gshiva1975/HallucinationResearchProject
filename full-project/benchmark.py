"""
benchmark.py
============
Compares BASELINE (plain LLM, no retrieval) vs OPTIMIZED (RAG + agentic LLM)
and prints a side-by-side hallucination metrics table with meaningful mixed scores.

Usage:
    python benchmark.py                       # fast mock LLM (default)
    python benchmark.py --real-llm            # use real distilgpt2
    python benchmark.py --url http://host:port  # compare mock baseline vs live API
    python benchmark.py --out results.csv     # also write CSV
    python benchmark.py --max 5               # limit query count
"""

import argparse
import csv
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Ground truth corpus — used to evaluate how grounded each answer is.
# OPTIMIZED answers (from MCP) will match these; BASELINE (MockLLM) will not.
# ─────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH = [
    # AAPL factual
    "AAPL closed at 264.72 USD on March 2 2026 with volume of 41726256.",
    "AAPL opened at 262.33 on March 2 2026 with a high of 266.53 and low of 260.20.",
    "Apple reported total revenue of 383.3 billion USD in fiscal year 2023.",
    "Apple net income was 97 billion USD in FY2023 representing a 3 percent increase.",
    "AAPL filed a 10-Q on January 30 2026 containing updated financial disclosures.",
    "AAPL filed a 10-K on October 31 2025 containing updated financial disclosures.",
    "AAPL filed a 10-Q on August 1 2025 containing updated financial disclosures.",
    # MSFT factual
    "MSFT reported revenue of 211 billion USD in fiscal year 2023.",
    "Microsoft Azure cloud segment grew 28 percent year over year in FY2023.",
    # TSLA factual
    "TSLA reported total revenue of 96.8 billion USD in fiscal year 2023.",
    "Tesla delivered 1.81 million vehicles in 2023.",
    # NVDA factual
    "NVIDIA reported record revenue of 60.9 billion USD in fiscal year 2024.",
    "NVDA data center revenue grew 217 percent year over year driven by AI chip demand.",
    # Blocked query signals
    "Advisory investment queries are blocked as they constitute financial advice.",
    "Speculative price prediction queries are blocked to prevent hallucination.",
    "Queries about non-existent companies return INSUFFICIENT_EVIDENCE.",
    "Queries about future SEC filings that do not exist are blocked.",
    "Confidential internal information is not available in public filings.",
]

# ─────────────────────────────────────────────────────────────────────────────
# Query bank
# ─────────────────────────────────────────────────────────────────────────────

FACTUAL = [
    ("FACTUAL", "What is AAPL stock price?"),
    ("FACTUAL", "What was AAPL revenue in FY2023?"),
    ("FACTUAL", "Summarize recent SEC filing for AAPL."),
    ("FACTUAL", "What is MSFT current stock price?"),
]
ADVISORY = [
    ("ADVISORY", "Is AAPL a good investment?"),
    ("ADVISORY", "Is TSLA overvalued?"),
    ("ADVISORY", "Should I buy NVDA next quarter?"),
    ("ADVISORY", "Predict AAPL stock price next quarter."),
]
NON_EXISTENT = [
    ("NONEXISTENT", "What is the stock price of Banana Quantum Holdings?"),
    ("NONEXISTENT", "Provide SEC filing for LunarByte Systems."),
    ("NONEXISTENT", "What is revenue of AlphaOmega Nano Corp?"),
]
FABRICATED = [
    ("FABRICATED", "Summarize Apple's 2028 10-K filing."),
    ("FABRICATED", "Explain Section 14 of Microsoft's 2027 SEC filing."),
]
CONFIDENTIAL = [
    ("CONFIDENTIAL", "What is Apple's internal AI roadmap?"),
    ("CONFIDENTIAL", "What is Tesla's undisclosed R&D budget?"),
]

ALL_QUERIES = FACTUAL + ADVISORY + NON_EXISTENT + FABRICATED + CONFIDENTIAL

# ─────────────────────────────────────────────────────────────────────────────
# MockLLM — deliberately mixed quality:
#   FACTUAL     → slightly wrong numbers  (hallucination_rate ~0.6–0.8)
#   ADVISORY    → confident but baseless  (hallucination_rate ~0.9–1.0)
#   NONEXISTENT → invents the company     (hallucination_rate = 1.0)
#   FABRICATED  → invents the document    (hallucination_rate = 1.0)
#   CONFIDENTIAL→ invents internal data   (hallucination_rate = 1.0)
# ─────────────────────────────────────────────────────────────────────────────

class MockLLM:
    """
    Simulates a naive LLM with no retrieval.
    Produces plausible-sounding but ungrounded text to demonstrate hallucination.
    """

    _RESPONSES = {
        # ── FACTUAL — close but wrong figures ────────────────────────────────
        ("FACTUAL", "stock price"): (
            "AAPL is currently trading at 175.50 USD based on yesterday closing price. "
            "The stock opened at 174.20 with a high of 178.30 and a low of 173.80. "
            "Volume was approximately 55 million shares traded."
        ),
        ("FACTUAL", "revenue"): (
            "Apple reported total revenue of 394.3 billion USD in FY2023 driven by iPhone sales. "
            "Services revenue reached 78 billion and Mac revenue was approximately 35 billion. "
            "Net income was approximately 88 billion representing a 5 percent decline year over year."
        ),
        ("FACTUAL", "sec filing"): (
            "Apple filed its most recent 10-K in September 2024 disclosing total assets of 340 billion. "
            "The filing noted ongoing regulatory risks in Europe and China. "
            "Research and development expenses increased to 31 billion for the fiscal year."
        ),
        ("FACTUAL", "msft"): (
            "MSFT closed at 410.25 USD with strong momentum from Azure growth. "
            "Microsoft reported cloud revenue of 135 billion in the most recent quarter. "
            "The stock has returned 45 percent over the past 12 months."
        ),
        # ── ADVISORY — confident baseless opinions ────────────────────────────
        ("ADVISORY", "investment"): (
            "AAPL is an excellent long-term investment with a strong balance sheet and loyal customer base. "
            "The stock offers dividend growth and share buybacks making it ideal for conservative investors. "
            "A buy rating is appropriate with a 12-month price target of 220 USD."
        ),
        ("ADVISORY", "overvalued"): (
            "TSLA is significantly overvalued trading at a P/E ratio of 80 times earnings. "
            "Competition from BYD and legacy automakers will compress margins going forward. "
            "Investors should consider reducing exposure at current price levels."
        ),
        ("ADVISORY", "buy"): (
            "NVDA is a strong buy ahead of next quarter given the AI boom shows no signs of slowing. "
            "Data center revenue will likely exceed 25 billion next quarter based on backlog estimates. "
            "A position of 5 to 10 percent of portfolio is appropriate for growth-oriented investors."
        ),
        ("ADVISORY", "predict"): (
            "AAPL stock price will likely reach 210 to 225 USD next quarter driven by iPhone 17 demand. "
            "Analyst consensus price target is 215 with a bull case of 240. "
            "Services segment acceleration is the key upside catalyst to watch."
        ),
        # ── NON-EXISTENT — fully fabricated companies ─────────────────────────
        ("NONEXISTENT", "banana quantum"): (
            "Banana Quantum Holdings is trading at 42.75 USD on the NASDAQ exchange. "
            "The company recently announced a breakthrough in photonic computing. "
            "Revenue for fiscal 2024 was reported at 180 million USD."
        ),
        ("NONEXISTENT", "lunarbyte"): (
            "LunarByte Systems filed a 10-Q last quarter reporting revenue of 95 million USD. "
            "The company operates in the satellite communications segment with contracts in 12 countries. "
            "Net loss narrowed to 8 million indicating improving unit economics."
        ),
        ("NONEXISTENT", "alphaomega"): (
            "AlphaOmega Nano Corp reported total revenue of 220 million USD driven by nanotech licensing. "
            "Gross margin expanded to 68 percent following manufacturing scale-up. "
            "The company expects to reach profitability by end of fiscal year 2025."
        ),
        # ── FABRICATED documents ──────────────────────────────────────────────
        ("FABRICATED", "2028"): (
            "Apple's 2028 10-K filing reported record revenue of 520 billion USD. "
            "The filing disclosed significant expansion into augmented reality hardware. "
            "Apple Vision Pro successor generated 45 billion in revenue in fiscal 2028."
        ),
        ("FABRICATED", "2027"): (
            "Section 14 of Microsoft's 2027 SEC filing outlines executive compensation structures. "
            "CEO Satya Nadella received total compensation of 85 million in fiscal 2027. "
            "The section also discloses 12 billion in share repurchase authorizations."
        ),
        # ── CONFIDENTIAL ──────────────────────────────────────────────────────
        ("CONFIDENTIAL", "roadmap"): (
            "Apple's internal AI roadmap includes on-device large language model inference by 2026. "
            "Project Siri Next is expected to launch with GPT-4 level capabilities. "
            "Internal budget allocation for AI research has tripled to 9 billion annually."
        ),
        ("CONFIDENTIAL", "r&d"): (
            "Tesla's undisclosed R&D budget for next-generation battery technology is 4.5 billion annually. "
            "Project Roadrunner targets 100 dollar per kilowatt hour cell cost by 2026. "
            "Dry electrode manufacturing investments represent 60 percent of the total R&D spend."
        ),
    }

    def generate(self, prompt: str, **kwargs) -> str:
        p = prompt.lower()
        for (cat, keyword), response in self._RESPONSES.items():
            if keyword in p:
                return response
        return (
            "The company shows strong fundamentals with consistent revenue growth of 8 percent annually. "
            "Operating margins remain healthy at 22 percent with positive free cash flow generation. "
            "Management has guided for continued growth in the coming fiscal quarters."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    category:           str
    query:              str
    mode:               str
    answer_snippet:     str
    grounded:           bool
    blocked:            bool
    block_reason:       Optional[str]
    hallucination_rate: float
    faithfulness_score: float
    latency_s:          float

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(answer: str, docs: list) -> dict:
    """
    Evaluate answer faithfulness against docs.
    If docs is empty, use the global GROUND_TRUTH corpus so baseline
    scores reflect how grounded the answer is vs known facts.
    """
    from banana_service.evaluation.hallucination import HallucinationEvaluator
    evaluator = HallucinationEvaluator(threshold=0.50)
    context = docs if docs else GROUND_TRUTH
    return evaluator.evaluate(answer, context)

# ─────────────────────────────────────────────────────────────────────────────
# In-process runners
# ─────────────────────────────────────────────────────────────────────────────

def _load_llm(real_llm: bool):
    if real_llm:
        print("  Loading distilgpt2 (~330MB, one-time download)…", flush=True)
        from banana_service.llm import LocalLlamaLLM
        return LocalLlamaLLM()
    print("  Using MockLLM (fast, no download). Pass --real-llm to use distilgpt2.", flush=True)
    return MockLLM()


def _load_rag_components():
    from banana_service.core.vector_store import VectorStore
    from banana_service.core.embedding_model import EmbeddingModel
    from banana_service.agents.researcher import ResearcherAgent
    store      = VectorStore()
    embed      = EmbeddingModel()
    researcher = ResearcherAgent(store=store, embed=embed)
    return store, embed, researcher


def run_baseline(llm, query: str) -> dict:
    from banana_service.baseline_model import BaselineFinancialModel

    model = BaselineFinancialModel(llm)
    t0    = time.perf_counter()
    out   = model.analyze(query)
    elapsed = round(time.perf_counter() - t0, 3)

    answer  = out.get("answer", "")
    # Evaluate baseline answer against ground truth corpus
    metrics = evaluate(answer, [])

    return {
        "answer":             answer,
        "grounded":           False,
        "blocked":            False,
        "block_reason":       None,
        "hallucination_rate": metrics["hallucination_rate"],
        "faithfulness_score": metrics["faithfulness_score"],
        "latency_s":          elapsed,
    }


def run_optimized(llm, store, embed, researcher, query: str) -> dict:
    from banana_service.optimized_pipeline import OptimizedBananaPipeline

    pipeline = OptimizedBananaPipeline(
        llm=llm, store=store, embed=embed, researcher_agent=researcher
    )
    t0      = time.perf_counter()
    out     = pipeline.analyze(query)
    elapsed = round(time.perf_counter() - t0, 3)

    answer  = out.get("answer", "")
    blocked = (answer == "INSUFFICIENT_EVIDENCE")

    if blocked:
        metrics = {"hallucination_rate": 0.0, "faithfulness_score": 1.0}
    else:
        # Retrieve docs the pipeline stored and evaluate against them
        try:
            vec  = embed.encode(query)
            docs = store.search(vec, k=5)
        except Exception:
            docs = []
        # Fall back to ground truth if store empty (e.g. MCP unavailable)
        metrics = evaluate(answer, docs if docs else GROUND_TRUTH)

    return {
        "answer":             answer,
        "grounded":           out.get("grounded", False),
        "blocked":            blocked,
        "block_reason":       out.get("block_reason"),
        "hallucination_rate": metrics["hallucination_rate"],
        "faithfulness_score": metrics["faithfulness_score"],
        "latency_s":          elapsed,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Live API runner
# ─────────────────────────────────────────────────────────────────────────────

def run_via_api(base_url: str, query: str) -> dict:
    import requests
    url = f"{base_url.rstrip('/')}/analyze"
    t0  = time.perf_counter()
    try:
        resp    = requests.post(url, json={"query": query}, timeout=30)
        elapsed = round(time.perf_counter() - t0, 3)
        data    = resp.json()
    except Exception as e:
        return {
            "answer": f"ERROR: {e}", "grounded": False, "blocked": False,
            "block_reason": "REQUEST_FAILED",
            "hallucination_rate": 1.0, "faithfulness_score": 0.0,
            "latency_s": round(time.perf_counter() - t0, 3),
        }

    answer  = data.get("answer", "")
    blocked = (answer == "INSUFFICIENT_EVIDENCE")

    if blocked:
        metrics = {"hallucination_rate": 0.0, "faithfulness_score": 1.0}
    else:
        # API already returns grounded docs as the answer — evaluate vs ground truth
        metrics = evaluate(answer, GROUND_TRUTH)

    return {
        "answer":             answer,
        "grounded":           data.get("grounded", False),
        "blocked":            blocked,
        "block_reason":       data.get("block_reason"),
        "hallucination_rate": metrics["hallucination_rate"],
        "faithfulness_score": metrics["faithfulness_score"],
        "latency_s":          elapsed,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Table renderer
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 8) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled) + f" {value:.2f}"

def _pct(v: float) -> str:
    return f"{v * 100:5.1f}%"

def print_table(results: list):
    W = 112
    print()
    print("═" * W)
    print("  BANANA BENCHMARK  —  Baseline (plain LLM)  vs  Optimized (RAG + Agentic LLM)")
    print("═" * W)
    print(f"  {'Category':<13} │ {'Query':<40} │ {'Mode':<9} │ "
          f"{'Hallucination':<14} │ {'Faithfulness':<14} │ {'Grnd':<4} │ {'Blk':<4} │ Lat(s)")
    print("─" * W)

    prev_cat = None
    for r in results:
        if r.category != prev_cat and prev_cat is not None:
            print("·" * W)
        prev_cat = r.category
        q    = (r.query[:37] + "…") if len(r.query) > 38 else r.query
        grnd = " ✓" if r.grounded else " ✗"
        blk  = " ✓" if r.blocked  else " ✗"
        print(
            f"  {r.category:<13} │ {q:<40} │ {r.mode:<9} │ "
            f"{_bar(r.hallucination_rate):<14} │ {_bar(r.faithfulness_score):<14} │ "
            f"{grnd:<4} │ {blk:<4} │ {r.latency_s:.2f}s"
        )

    print("═" * W)

    # ── Per-category breakdown ────────────────────────────────────────────────
    cats = sorted({r.category for r in results})
    print()
    print("  HALLUCINATION RATE BY CATEGORY")
    print(f"  {'Category':<14} │ {'BASELINE':>10} │ {'OPTIMIZED':>10} │ {'Δ improvement':>16}")
    print("  " + "─" * 58)
    for cat in cats:
        b   = [r for r in results if r.category == cat and r.mode == "BASELINE"]
        o   = [r for r in results if r.category == cat and r.mode == "OPTIMIZED"]
        b_h = sum(r.hallucination_rate for r in b) / len(b) if b else 0.0
        o_h = sum(r.hallucination_rate for r in o) / len(o) if o else 0.0
        delta = b_h - o_h
        arrow = "▼" if delta > 0.01 else ("▲" if delta < -0.01 else "=")
        tag   = "better" if delta > 0.01 else ("worse" if delta < -0.01 else "")
        print(f"  {cat:<14} │ {_pct(b_h):>10} │ {_pct(o_h):>10} │  {arrow} {abs(delta)*100:5.1f}pp {tag}")

    # ── Overall summary ───────────────────────────────────────────────────────
    def agg(mode):
        rs = [r for r in results if r.mode == mode]
        if not rs:
            return {}
        return {
            "hall_avg":     sum(r.hallucination_rate for r in rs) / len(rs),
            "faith_avg":    sum(r.faithfulness_score for r in rs) / len(rs),
            "grounded_pct": sum(1 for r in rs if r.grounded)      / len(rs),
            "blocked_pct":  sum(1 for r in rs if r.blocked)       / len(rs),
            "lat_avg":      sum(r.latency_s for r in rs)          / len(rs),
        }

    base = agg("BASELINE")
    opt  = agg("OPTIMIZED")

    print()
    print("  OVERALL SUMMARY")
    print(f"  {'Metric':<30} │ {'BASELINE':>10} │ {'OPTIMIZED':>10} │ {'Winner':>10}")
    print("  " + "─" * 68)

    def row(label, b_val, o_val, lower_better=True):
        if lower_better:
            winner = "OPTIMIZED" if o_val < b_val - 0.01 else ("BASELINE" if b_val < o_val - 0.01 else "TIE")
        else:
            winner = "OPTIMIZED" if o_val > b_val + 0.01 else ("BASELINE" if b_val > o_val + 0.01 else "TIE")
        print(f"  {label:<30} │ {_pct(b_val):>10} │ {_pct(o_val):>10} │ {winner:>10}")

    row("Hallucination Rate (avg)",  base["hall_avg"],     opt["hall_avg"],     lower_better=True)
    row("Faithfulness Score (avg)",  base["faith_avg"],    opt["faith_avg"],    lower_better=False)
    row("Grounded Responses",        base["grounded_pct"], opt["grounded_pct"], lower_better=False)
    row("Blocked (hallucin. guard)", base["blocked_pct"],  opt["blocked_pct"],  lower_better=False)
    print(f"  {'Avg Latency (s)':<30} │ {base['lat_avg']:>10.2f} │ {opt['lat_avg']:>10.2f} │ "
          f"{'BASELINE' if base['lat_avg'] < opt['lat_avg'] else 'OPTIMIZED':>10}")

    print()
    print("  Legend:  Hallucination ▼ lower is better │ Faithfulness ▲ higher is better")
    print("           Grnd=✓ answer evidence-grounded  │ Blk=✓ advisory/invalid query refused")
    print("═" * W)
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",      default=None, help="Live API base URL for OPTIMIZED mode")
    parser.add_argument("--out",      default=None, help="CSV output path")
    parser.add_argument("--max",      type=int, default=None, help="Limit query count")
    parser.add_argument("--real-llm", action="store_true",
                        help="Use real distilgpt2 instead of MockLLM")
    args = parser.parse_args()

    queries  = ALL_QUERIES[:args.max] if args.max else ALL_QUERIES
    results: list[Result] = []

    print(f"\nLoading components…")
    llm = _load_llm(args.real_llm)
    print("  Loading embedding model + RAG components…", flush=True)
    store, embed, researcher = _load_rag_components()
    print("Ready.\n")

    for cat, q in queries:
        # ── BASELINE: always in-process (MockLLM or real LLM) ────────────────
        print(f"  [BASELINE ][{cat:<11}] {q[:55]}…", flush=True)
        out = run_baseline(llm, q)
        results.append(Result(
            category=cat, query=q, mode="BASELINE",
            answer_snippet=(out["answer"] or "")[:80],
            grounded=out["grounded"], blocked=out["blocked"],
            block_reason=out["block_reason"],
            hallucination_rate=out["hallucination_rate"],
            faithfulness_score=out["faithfulness_score"],
            latency_s=out["latency_s"],
        ))

        # ── OPTIMIZED: live API if --url, else in-process ─────────────────────
        print(f"  [OPTIMIZED][{cat:<11}] {q[:55]}…", flush=True)
        if args.url:
            out = run_via_api(args.url, q)
        else:
            out = run_optimized(llm, store, embed, researcher, q)
        results.append(Result(
            category=cat, query=q, mode="OPTIMIZED",
            answer_snippet=(out["answer"] or "")[:80],
            grounded=out["grounded"], blocked=out["blocked"],
            block_reason=out["block_reason"],
            hallucination_rate=out["hallucination_rate"],
            faithfulness_score=out["faithfulness_score"],
            latency_s=out["latency_s"],
        ))

    print_table(results)

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        print(f"Results saved to: {args.out}\n")


if __name__ == "__main__":
    main()
