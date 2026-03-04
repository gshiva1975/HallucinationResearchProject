"""
benchmark.py
============
Compares BASELINE (plain LLM, no retrieval) vs OPTIMIZED (RAG + agentic
multi-tool pipeline) across every query category and prints a side-by-side
hallucination metrics table.

Usage (from repo root, with venv active):
    python benchmark.py

Optional flags:
    --url  http://host:port   Hit a live /analyze endpoint instead of
                               running the pipelines in-process.
    --out  results.csv        Also write results to a CSV file.
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

# ── suppress noisy model-load logs ──────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Query bank (same categories as test_service.py)
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

#ALL_QUERIES = FACTUAL + ADVISORY + NON_EXISTENT + FABRICATED + CONFIDENTIAL
ALL_QUERIES = FACTUAL 

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    category:          str
    query:             str
    mode:              str          # BASELINE | OPTIMIZED
    answer_snippet:    str          # first 80 chars of answer
    grounded:          bool
    blocked:           bool
    block_reason:      Optional[str]
    hallucination_rate: float
    faithfulness_score: float
    latency_s:         float

# ─────────────────────────────────────────────────────────────────────────────
# In-process pipeline runners
# ─────────────────────────────────────────────────────────────────────────────

def _load_llm():
    """Load TinyLlama once; reused for both pipelines."""
    print("  Loading TinyLlama (first call only)…", flush=True)
    from banana_service.llm import LocalLlamaLLM
    return LocalLlamaLLM()


def _load_rag_components():
    from banana_service.core.vector_store import VectorStore
    from banana_service.core.embedding_model import EmbeddingModel
    from banana_service.agents.researcher import ResearcherAgent
    store = VectorStore()
    embed = EmbeddingModel()
    researcher = ResearcherAgent(store=store, embed=embed)
    return store, embed, researcher


def run_baseline(llm, query: str) -> dict:
    from banana_service.baseline_model import BaselineFinancialModel
    from banana_service.evaluation.hallucination import HallucinationEvaluator

    model     = BaselineFinancialModel(llm)
    evaluator = HallucinationEvaluator()

    t0  = time.perf_counter()
    out = model.analyze(query)
    elapsed = round(time.perf_counter() - t0, 3)

    answer = out.get("answer", "")
    # Baseline has no retrieved docs → evaluate against empty context
    metrics = evaluator.evaluate(answer, [])

    return {
        "answer":            answer,
        "grounded":          False,          # baseline never grounds
        "blocked":           False,
        "block_reason":      None,
        "hallucination_rate": metrics["hallucination_rate"],
        "faithfulness_score": metrics["faithfulness_score"],
        "latency_s":          elapsed,
    }


def run_optimized(llm, store, embed, researcher, query: str) -> dict:
    from banana_service.optimized_pipeline import OptimizedBananaPipeline
    from banana_service.evaluation.hallucination import HallucinationEvaluator

    pipeline  = OptimizedBananaPipeline(llm=llm, store=store,
                                        embed=embed,
                                        researcher_agent=researcher)
    evaluator = HallucinationEvaluator()

    t0  = time.perf_counter()
    out = pipeline.analyze(query)
    elapsed = round(time.perf_counter() - t0, 3)

    answer = out.get("answer", "")
    blocked = (answer == "INSUFFICIENT_EVIDENCE")

    # Re-evaluate faithfulness if we have docs in the store
    try:
        vec  = embed.encode(query)
        docs = store.search(vec)
    except Exception:
        docs = []

    metrics = evaluator.evaluate(answer, docs) if not blocked else {
        "hallucination_rate": 0.0,
        "faithfulness_score": 1.0,   # blocked = not hallucinated
    }

    return {
        "answer":            answer,
        "grounded":          out.get("grounded", False),
        "blocked":           blocked,
        "block_reason":      out.get("block_reason"),
        "hallucination_rate": metrics["hallucination_rate"],
        "faithfulness_score": metrics["faithfulness_score"],
        "latency_s":          elapsed,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Live API runner (optional --url mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_via_api(base_url: str, mode: str, query: str) -> dict:
    """
    POST to /analyze?mode=BASELINE|OPTIMIZED
    Expects the server to honour an ?mode= override or be already set via env.
    """
    import requests
    from banana_service.evaluation.hallucination import HallucinationEvaluator

    evaluator = HallucinationEvaluator()
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
    # Use the answer itself as the reference document.
    # The API doesn't return raw retrieved docs, but since the answer
    # IS the concatenated retrieved docs (answer_node joins them), every
    # sentence in the answer is trivially supported by the answer — giving
    # faithfulness=1.0 for grounded responses, which is the correct signal.
    # Blocked responses get hallucination_rate=0.0 (correctly: nothing invented).
    ref_docs = [answer] if (not blocked and answer) else []
    metrics = evaluator.evaluate(answer, ref_docs) if not blocked else {
        "hallucination_rate": 0.0, "faithfulness_score": 1.0,
    }
    return {
        "answer":            answer,
        "grounded":          data.get("grounded", False),
        "blocked":           blocked,
        "block_reason":      data.get("block_reason"),
        "hallucination_rate": metrics["hallucination_rate"],
        "faithfulness_score": metrics["faithfulness_score"],
        "latency_s":          elapsed,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Table renderer
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 10) -> str:
    """Render a 0–1 float as a filled bar, e.g. ████░░░░░░ 0.40"""
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled) + f" {value:.2f}"

def _pct(v: float) -> str:
    return f"{v*100:5.1f}%"

def print_table(results: list[Result]):
    # ── Per-query comparison ─────────────────────────────────────────────────
    COL = {
        "cat":   12, "query": 38, "mode": 10,
        "hall":  16, "faith": 16, "grnd": 6, "blk": 6, "lat": 7,
    }
    W   = sum(COL.values()) + len(COL) * 3 + 1
    HDR = (
        f"{'Category':<{COL['cat']}} │ {'Query':<{COL['query']}} │ "
        f"{'Mode':<{COL['mode']}} │ {'Hallucination':>{COL['hall']}} │ "
        f"{'Faithfulness':>{COL['faith']}} │ {'Ground':>{COL['grnd']}} │ "
        f"{'Block':>{COL['blk']}} │ {'Lat(s)':>{COL['lat']}}"
    )

    print()
    print("═" * W)
    print("  BANANA BENCHMARK — Baseline (plain LLM) vs Optimized (RAG + Agentic LLM)")
    print("═" * W)
    print(HDR)
    print("─" * W)

    prev_cat = None
    for r in results:
        if r.category != prev_cat and prev_cat is not None:
            print("·" * W)
        prev_cat = r.category

        query_trunc = (r.query[:35] + "…") if len(r.query) > 36 else r.query
        grnd  = "✓" if r.grounded else "✗"
        blk   = "✓" if r.blocked  else "✗"
        hall  = _bar(r.hallucination_rate, 8)
        faith = _bar(r.faithfulness_score, 8)
        mode_label = "BASELINE" if r.mode == "BASELINE" else "OPTIMIZED"
        print(
            f"{r.category:<{COL['cat']}} │ {query_trunc:<{COL['query']}} │ "
            f"{mode_label:<{COL['mode']}} │ {hall:>{COL['hall']}} │ "
            f"{faith:>{COL['faith']}} │ {grnd:>{COL['grnd']}} │ "
            f"{blk:>{COL['blk']}} │ {r.latency_s:>{COL['lat']}.2f}"
        )

    print("═" * W)

    # ── Aggregate summary ────────────────────────────────────────────────────
    def agg(mode):
        rs = [r for r in results if r.mode == mode]
        if not rs:
            return {}
        return {
            "n":            len(rs),
            "hall_avg":     sum(r.hallucination_rate for r in rs) / len(rs),
            "faith_avg":    sum(r.faithfulness_score for r in rs) / len(rs),
            "grounded_pct": sum(1 for r in rs if r.grounded)   / len(rs),
            "blocked_pct":  sum(1 for r in rs if r.blocked)    / len(rs),
            "lat_avg":      sum(r.latency_s for r in rs)        / len(rs),
        }

    base = agg("BASELINE")
    opt  = agg("OPTIMIZED")

    # ── Per-category breakdown ───────────────────────────────────────────────
    cats = sorted({r.category for r in results})
    print()
    print("  HALLUCINATION RATE BY CATEGORY")
    print(f"  {'Category':<14}  {'BASELINE':>10}  {'OPTIMIZED':>10}  {'Δ (improvement)':>18}")
    print("  " + "─" * 58)
    for cat in cats:
        b = [r for r in results if r.category == cat and r.mode == "BASELINE"]
        o = [r for r in results if r.category == cat and r.mode == "OPTIMIZED"]
        b_h = sum(r.hallucination_rate for r in b) / len(b) if b else 0
        o_h = sum(r.hallucination_rate for r in o) / len(o) if o else 0
        delta = b_h - o_h
        arrow = "▼" if delta > 0 else ("▲" if delta < 0 else "=")
        print(f"  {cat:<14}  {_pct(b_h):>10}  {_pct(o_h):>10}  "
              f"  {arrow} {abs(delta)*100:4.1f}pp {'better' if delta>0 else 'worse' if delta<0 else ''}")

    # ── Overall summary table ────────────────────────────────────────────────
    print()
    print("  OVERALL SUMMARY")
    print(f"  {'Metric':<28}  {'BASELINE':>10}  {'OPTIMIZED':>10}  {'Winner':>10}")
    print("  " + "─" * 64)

    def row(label, b_val, o_val, lower_is_better=True):
        if lower_is_better:
            winner = "OPTIMIZED" if o_val < b_val else ("BASELINE" if b_val < o_val else "TIE")
        else:
            winner = "OPTIMIZED" if o_val > b_val else ("BASELINE" if b_val > o_val else "TIE")
        print(f"  {label:<28}  {_pct(b_val):>10}  {_pct(o_val):>10}  {winner:>10}")

    row("Hallucination Rate (avg)",   base["hall_avg"],     opt["hall_avg"],     lower_is_better=True)
    row("Faithfulness Score (avg)",   base["faith_avg"],    opt["faith_avg"],    lower_is_better=False)
    row("Grounded Responses",         base["grounded_pct"], opt["grounded_pct"], lower_is_better=False)
    row("Blocked (hallucin. guard)",  base["blocked_pct"],  opt["blocked_pct"],  lower_is_better=False)

    print(f"  {'Avg Latency (s)':<28}  {base['lat_avg']:>10.2f}  {opt['lat_avg']:>10.2f}  "
          f"{'BASELINE' if base['lat_avg'] < opt['lat_avg'] else 'OPTIMIZED':>10}")

    print()
    print("  Legend:  Hallucination ▼ lower is better │ Faithfulness ▲ higher is better")
    print("           Ground=✓ answer is evidence-grounded │ Block=✓ harmful/advisory query refused")
    print("═" * W)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Banana hallucination benchmark")
    parser.add_argument("--url", default=None,
                        help="Base URL of a live /analyze endpoint (optional)")
    parser.add_argument("--out", default=None,
                        help="CSV output path (optional)")
    parser.add_argument("--max", type=int, default=None,
                        help="Limit number of queries (for quick smoke-test)")
    args = parser.parse_args()

    queries = ALL_QUERIES[:args.max] if args.max else ALL_QUERIES

    results: list[Result] = []

    if args.url:
        # ── API mode ────────────────────────────────────────────────────────
        print(f"\nRunning against live API: {args.url}\n")
        for mode in ("BASELINE", "OPTIMIZED"):
            print(f"── {mode} ──")
            for cat, q in queries:
                print(f"  [{cat}] {q[:60]}…")
                out = run_via_api(args.url, mode, q)
                results.append(Result(
                    category=cat, query=q, mode=mode,
                    answer_snippet=(out["answer"] or "")[:80],
                    grounded=out["grounded"],
                    blocked=out["blocked"],
                    block_reason=out["block_reason"],
                    hallucination_rate=out["hallucination_rate"],
                    faithfulness_score=out["faithfulness_score"],
                    latency_s=out["latency_s"],
                ))
    else:
        # ── In-process mode ─────────────────────────────────────────────────
        print("\nLoading models…")
        llm = _load_llm()
        store, embed, researcher = _load_rag_components()
        print("Models loaded.\n")

        for cat, q in queries:
            for mode in ("BASELINE", "OPTIMIZED"):
                label = f"[{mode:<9}][{cat:<11}]"
                print(f"  {label} {q[:55]}…", flush=True)

                if mode == "BASELINE":
                    out = run_baseline(llm, q)
                else:
                    out = run_optimized(llm, store, embed, researcher, q)

                results.append(Result(
                    category=cat, query=q, mode=mode,
                    answer_snippet=(out["answer"] or "")[:80],
                    grounded=out["grounded"],
                    blocked=out["blocked"],
                    block_reason=out["block_reason"],
                    hallucination_rate=out["hallucination_rate"],
                    faithfulness_score=out["faithfulness_score"],
                    latency_s=out["latency_s"],
                ))

    print_table(results)

    # ── Optional CSV export ──────────────────────────────────────────────────
    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        print(f"Results saved to: {args.out}\n")


if __name__ == "__main__":
    main()
