"""
benchmark.py
============
Compares BASELINE (plain LLM, no retrieval) vs OPTIMIZED (RAG + agentic
multi-tool pipeline) across every query category and prints a side-by-side
hallucination metrics table.

Usage (from repo root, with venv active):
    python benchmark.py

Optional flags:
    --url   http://host:port   Hit a live /analyze endpoint instead of
                                running the pipelines in-process.
    --out   results.csv        Also write results to a CSV file.
    --max   N                  Limit number of queries (quick smoke-test).
    --debug                    Enable DEBUG-level trace logs (default: INFO).
"""

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup  — call setup_logging() after argparse so --debug can flip it
# ─────────────────────────────────────────────────────────────────────────────

LOG_FMT  = "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
DATE_FMT = "%H:%M:%S"

def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format=LOG_FMT, datefmt=DATE_FMT, stream=sys.stdout)
    # suppress extremely noisy third-party loggers unless in debug mode
    if not debug:
        for noisy in ("transformers", "torch", "urllib3", "httpx", "chromadb",
                      "sentence_transformers", "llama_cpp", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

log = logging.getLogger("benchmark")

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
# Trace helpers
# ─────────────────────────────────────────────────────────────────────────────

class _StepTimer:
    """Context manager that logs entry, exit, and elapsed time for a step."""
    def __init__(self, name: str, logger: logging.Logger = log):
        self.name   = name
        self.logger = logger
        self._t0    = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.logger.debug("  ┌─ START  %s", self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = round(time.perf_counter() - self._t0, 3)
        if exc_type:
            self.logger.error("  └─ ERROR  %s  (%.3fs)  %s: %s",
                              self.name, elapsed, exc_type.__name__, exc_val)
        else:
            self.logger.debug("  └─ DONE   %s  (%.3fs)", self.name, elapsed)
        return False   # do not suppress exceptions

def step(name):
    return _StepTimer(name)

# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_llm():
    log.info("Loading TinyLlama LLM — this may take 30–60 s on first run…")
    with step("LLM load"):
        from banana_service.llm import LocalLlamaLLM
        llm = LocalLlamaLLM()
    log.info("LLM ready ✓")
    return llm


def _load_rag_components():
    log.info("Loading RAG components (VectorStore, EmbeddingModel, ResearcherAgent)…")
    with step("VectorStore init"):
        from banana_service.core.vector_store import VectorStore
        store = VectorStore()
    log.debug("  VectorStore collection size: %s", getattr(store, "count", lambda: "?")())

    with step("EmbeddingModel init"):
        from banana_service.core.embedding_model import EmbeddingModel
        embed = EmbeddingModel()

    with step("ResearcherAgent init"):
        from banana_service.agents.researcher import ResearcherAgent
        researcher = ResearcherAgent(store=store, embed=embed)

    log.info("RAG components ready ✓")
    return store, embed, researcher

# ─────────────────────────────────────────────────────────────────────────────
# Baseline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(llm, query: str) -> dict:
    qlog = logging.getLogger("benchmark.baseline")
    qlog.info("BASELINE  query='%s'", query[:80])

    with step("BaselineFinancialModel import"):
        from banana_service.baseline_model import BaselineFinancialModel
        from banana_service.evaluation.hallucination import HallucinationEvaluator
        model     = BaselineFinancialModel(llm)
        evaluator = HallucinationEvaluator()

    t0 = time.perf_counter()
    qlog.debug("  Calling model.analyze()…")
    try:
        with step("model.analyze"):
            out = model.analyze(query)
    except Exception as e:
        qlog.error("  model.analyze() raised: %s", e, exc_info=True)
        raise
    elapsed = round(time.perf_counter() - t0, 3)

    answer = out.get("answer", "")
    qlog.info("  Answer received  len=%d  elapsed=%.3fs", len(answer), elapsed)
    qlog.debug("  Answer snippet: %s", answer[:120])

    ref_docs = [answer] if answer else []
    qlog.debug("  Evaluating hallucination  ref_docs=%d", len(ref_docs))
    with step("HallucinationEvaluator.evaluate"):
        metrics = evaluator.evaluate(answer, ref_docs) if ref_docs else {
            "hallucination_rate": 1.0, "faithfulness_score": 0.0,
        }
    qlog.info("  Metrics  hall=%.3f  faith=%.3f",
              metrics["hallucination_rate"], metrics["faithfulness_score"])

    return {
        "answer":             answer,
        "grounded":           False,
        "blocked":            False,
        "block_reason":       None,
        "hallucination_rate": metrics["hallucination_rate"],
        "faithfulness_score": metrics["faithfulness_score"],
        "latency_s":          elapsed,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Optimized runner
# ─────────────────────────────────────────────────────────────────────────────

def run_optimized(llm, store, embed, researcher, query: str) -> dict:
    qlog = logging.getLogger("benchmark.optimized")
    qlog.info("OPTIMIZED  query='%s'", query[:80])

    with step("OptimizedBananaPipeline import"):
        from banana_service.optimized_pipeline import OptimizedBananaPipeline
        from banana_service.evaluation.hallucination import HallucinationEvaluator
        pipeline  = OptimizedBananaPipeline(
            llm=llm, store=store, embed=embed, researcher_agent=researcher
        )
        evaluator = HallucinationEvaluator()

    t0 = time.perf_counter()
    qlog.debug("  Calling pipeline.analyze()…")
    try:
        with step("pipeline.analyze"):
            out = pipeline.analyze(query)
    except Exception as e:
        qlog.error("  pipeline.analyze() raised: %s", e, exc_info=True)
        raise
    elapsed = round(time.perf_counter() - t0, 3)

    answer   = out.get("answer", "")
    blocked  = (answer == "INSUFFICIENT_EVIDENCE")
    grounded = out.get("grounded", False)
    block_reason = out.get("block_reason")

    qlog.info("  Pipeline done  blocked=%s  grounded=%s  block_reason=%s  elapsed=%.3fs",
              blocked, grounded, block_reason, elapsed)
    qlog.debug("  Answer snippet: %s", answer[:120])

    if blocked:
        qlog.debug("  Blocked → skipping hallucination eval")
        metrics = {"hallucination_rate": 0.0, "faithfulness_score": 1.0}
    else:
        qlog.debug("  Searching vector store for ref docs…")
        try:
            with step("store.search"):
                vec  = embed.encode(query)
                docs = store.search(vec)
            qlog.debug("  store.search returned %d doc(s)", len(docs))
        except Exception as e:
            qlog.warning("  store.search failed: %s — falling back to [answer]", e)
            docs = []

        ref_docs = docs if docs else ([answer] if answer else [])
        qlog.debug("  Evaluating hallucination  ref_docs=%d", len(ref_docs))
        with step("HallucinationEvaluator.evaluate"):
            metrics = evaluator.evaluate(answer, ref_docs) if ref_docs else {
                "hallucination_rate": 1.0, "faithfulness_score": 0.0,
            }

    qlog.info("  Metrics  hall=%.3f  faith=%.3f",
              metrics["hallucination_rate"], metrics["faithfulness_score"])

    return {
        "answer":             answer,
        "grounded":           grounded,
        "blocked":            blocked,
        "block_reason":       block_reason,
        "hallucination_rate": metrics["hallucination_rate"],
        "faithfulness_score": metrics["faithfulness_score"],
        "latency_s":          elapsed,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Live API runner
# ─────────────────────────────────────────────────────────────────────────────

def run_via_api(base_url: str, mode: str, query: str) -> dict:
    alog = logging.getLogger("benchmark.api")
    alog.info("API  mode=%s  query='%s'", mode, query[:80])

    import requests
    from banana_service.evaluation.hallucination import HallucinationEvaluator
    evaluator = HallucinationEvaluator()

    url     = f"{base_url.rstrip('/')}/analyze"
    payload = {"query": query, "mode": mode}
    alog.debug("  POST %s  payload=%s", url, payload)

    t0 = time.perf_counter()
    try:
        with step("HTTP POST /analyze"):
            resp = requests.post(url, json=payload, timeout=30)
        elapsed = round(time.perf_counter() - t0, 3)
        alog.debug("  HTTP %d  elapsed=%.3fs", resp.status_code, elapsed)

        if resp.status_code != 200:
            alog.error("  Non-200 response: %d  body=%s", resp.status_code, resp.text[:200])

        with step("response JSON parse"):
            data = resp.json()
        alog.debug("  Response keys: %s", list(data.keys()))

    except Exception as e:
        elapsed = round(time.perf_counter() - t0, 3)
        alog.error("  Request failed after %.3fs: %s", elapsed, e, exc_info=True)
        return {
            "answer":             f"ERROR: {e}",
            "grounded":           False,
            "blocked":            False,
            "block_reason":       "REQUEST_FAILED",
            "hallucination_rate": 1.0,
            "faithfulness_score": 0.0,
            "latency_s":          elapsed,
        }

    answer       = data.get("answer", "")
    blocked      = (answer == "INSUFFICIENT_EVIDENCE")
    block_reason = data.get("block_reason")
    grounded     = data.get("grounded", False)

    alog.info("  blocked=%s  grounded=%s  block_reason=%s  answer_len=%d",
              blocked, grounded, block_reason, len(answer))
    alog.debug("  Answer snippet: %s", answer[:120])

    ref_docs = [answer] if (not blocked and answer) else []
    alog.debug("  Evaluating hallucination  ref_docs=%d", len(ref_docs))

    with step("HallucinationEvaluator.evaluate"):
        metrics = evaluator.evaluate(answer, ref_docs) if not blocked else {
            "hallucination_rate": 0.0, "faithfulness_score": 1.0,
        }

    alog.info("  Metrics  hall=%.3f  faith=%.3f",
              metrics["hallucination_rate"], metrics["faithfulness_score"])

    return {
        "answer":             answer,
        "grounded":           grounded,
        "blocked":            blocked,
        "block_reason":       block_reason,
        "hallucination_rate": metrics["hallucination_rate"],
        "faithfulness_score": metrics["faithfulness_score"],
        "latency_s":          elapsed,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Table renderer
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled) + f" {value:.2f}"

def _pct(v: float) -> str:
    return f"{v * 100:5.1f}%"

def print_table(results: list[Result]):
    COL = {"cat":12,"query":38,"mode":10,"hall":16,"faith":16,"grnd":6,"blk":6,"lat":7}
    W   = sum(COL.values()) + len(COL) * 3 + 1
    HDR = (
        f"{'Category':<{COL['cat']}} │ {'Query':<{COL['query']}} │ "
        f"{'Mode':<{COL['mode']}} │ {'Hallucination':>{COL['hall']}} │ "
        f"{'Faithfulness':>{COL['faith']}} │ {'Ground':>{COL['grnd']}} │ "
        f"{'Block':>{COL['blk']}} │ {'Lat(s)':>{COL['lat']}}"
    )
    print(); print("═"*W)
    print("  BANANA BENCHMARK — Baseline (plain LLM) vs Optimized (RAG + Agentic LLM)")
    print("═"*W); print(HDR); print("─"*W)

    prev_cat = None
    for r in results:
        if r.category != prev_cat and prev_cat is not None:
            print("·" * W)
        prev_cat = r.category
        query_trunc = (r.query[:35] + "…") if len(r.query) > 36 else r.query
        print(
            f"{r.category:<{COL['cat']}} │ {query_trunc:<{COL['query']}} │ "
            f"{'BASELINE' if r.mode=='BASELINE' else 'OPTIMIZED':<{COL['mode']}} │ "
            f"{_bar(r.hallucination_rate,8):>{COL['hall']}} │ "
            f"{_bar(r.faithfulness_score,8):>{COL['faith']}} │ "
            f"{'✓' if r.grounded else '✗':>{COL['grnd']}} │ "
            f"{'✓' if r.blocked  else '✗':>{COL['blk']}} │ "
            f"{r.latency_s:>{COL['lat']}.2f}"
        )
    print("═"*W)

    def agg(mode):
        rs = [r for r in results if r.mode == mode]
        if not rs: return {}
        return {
            "hall_avg":     sum(r.hallucination_rate for r in rs) / len(rs),
            "faith_avg":    sum(r.faithfulness_score for r in rs) / len(rs),
            "grounded_pct": sum(1 for r in rs if r.grounded) / len(rs),
            "blocked_pct":  sum(1 for r in rs if r.blocked)  / len(rs),
            "lat_avg":      sum(r.latency_s for r in rs)      / len(rs),
        }

    base = agg("BASELINE"); opt = agg("OPTIMIZED")

    cats = sorted({r.category for r in results})
    print("\n  HALLUCINATION RATE BY CATEGORY")
    print(f"  {'Category':<14}  {'BASELINE':>10}  {'OPTIMIZED':>10}  {'Δ':>18}")
    print("  " + "─"*58)
    for cat in cats:
        b   = [r for r in results if r.category==cat and r.mode=="BASELINE"]
        o   = [r for r in results if r.category==cat and r.mode=="OPTIMIZED"]
        b_h = sum(r.hallucination_rate for r in b)/len(b) if b else 0
        o_h = sum(r.hallucination_rate for r in o)/len(o) if o else 0
        d   = b_h - o_h
        print(f"  {cat:<14}  {_pct(b_h):>10}  {_pct(o_h):>10}  "
              f"  {'▼' if d>0 else '▲' if d<0 else '='} {abs(d)*100:4.1f}pp "
              f"{'better' if d>0 else 'worse' if d<0 else ''}")

    print("\n  OVERALL SUMMARY")
    print(f"  {'Metric':<28}  {'BASELINE':>10}  {'OPTIMIZED':>10}  {'Winner':>10}")
    print("  "+"─"*64)

    def row(label, bv, ov, lib=True):
        w = "OPTIMIZED" if (ov<bv if lib else ov>bv) else ("BASELINE" if (bv<ov if lib else bv>ov) else "TIE")
        print(f"  {label:<28}  {_pct(bv):>10}  {_pct(ov):>10}  {w:>10}")

    row("Hallucination Rate (avg)",  base["hall_avg"],     opt["hall_avg"],     lib=True)
    row("Faithfulness Score (avg)",  base["faith_avg"],    opt["faith_avg"],    lib=False)
    row("Grounded Responses",        base["grounded_pct"], opt["grounded_pct"], lib=False)
    row("Blocked (hallucin. guard)", base["blocked_pct"],  opt["blocked_pct"],  lib=False)
    print(f"  {'Avg Latency (s)':<28}  {base['lat_avg']:>10.2f}  {opt['lat_avg']:>10.2f}  "
          f"{'BASELINE' if base['lat_avg']<opt['lat_avg'] else 'OPTIMIZED':>10}")
    print("\n  Legend:  Hallucination ▼ lower is better │ Faithfulness ▲ higher is better")
    print("           Ground=✓ evidence-grounded │ Block=✓ harmful/advisory refused")
    print("═"*W); print()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Banana hallucination benchmark")
    parser.add_argument("--url",   default=None, help="Base URL of live /analyze endpoint")
    parser.add_argument("--out",   default=None, help="CSV output path")
    parser.add_argument("--max",   type=int, default=None, help="Limit query count")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG trace logs")
    args = parser.parse_args()

    setup_logging(debug=args.debug)
    log.info("Banana Benchmark starting  url=%s  max=%s  debug=%s",
             args.url, args.max, args.debug)

    queries = ALL_QUERIES[:args.max] if args.max else ALL_QUERIES
    log.info("Query plan: %d queries across %d categories",
             len(queries), len({c for c,_ in queries}))

    results: list[Result] = []

    if args.url:
        log.info("Mode: LIVE API  →  %s", args.url)
        for mode in ("BASELINE", "OPTIMIZED"):
            log.info("── %s pass ── (%d queries)", mode, len(queries))
            for i, (cat, q) in enumerate(queries, 1):
                log.info("[%d/%d] %s  [%s]  %s", i, len(queries), mode, cat, q[:60])
                out = run_via_api(args.url, mode, q)
                results.append(Result(
                    category=cat, query=q, mode=mode,
                    answer_snippet=(out["answer"] or "")[:80],
                    grounded=out["grounded"], blocked=out["blocked"],
                    block_reason=out["block_reason"],
                    hallucination_rate=out["hallucination_rate"],
                    faithfulness_score=out["faithfulness_score"],
                    latency_s=out["latency_s"],
                ))
                log.info("    ✓ done  hall=%.2f  faith=%.2f  lat=%.2fs",
                         out["hallucination_rate"], out["faithfulness_score"], out["latency_s"])
    else:
        log.info("Mode: IN-PROCESS  (no --url provided)")
        log.info("Step 1/3  Loading LLM…")
        llm = _load_llm()

        log.info("Step 2/3  Loading RAG components…")
        store, embed, researcher = _load_rag_components()

        log.info("Step 3/3  Running %d queries × 2 modes = %d total runs",
                 len(queries), len(queries)*2)

        total = len(queries) * 2
        run_n = 0
        for cat, q in queries:
            for mode in ("BASELINE", "OPTIMIZED"):
                run_n += 1
                log.info("[%d/%d] %s  [%s]  %s", run_n, total, mode, cat, q[:60])
                try:
                    if mode == "BASELINE":
                        out = run_baseline(llm, q)
                    else:
                        out = run_optimized(llm, store, embed, researcher, q)
                except Exception as e:
                    log.error("  RUN FAILED — %s: %s", type(e).__name__, e, exc_info=True)
                    out = {
                        "answer": f"ERROR: {e}", "grounded": False, "blocked": False,
                        "block_reason": "EXCEPTION",
                        "hallucination_rate": 1.0, "faithfulness_score": 0.0,
                        "latency_s": 0.0,
                    }
                results.append(Result(
                    category=cat, query=q, mode=mode,
                    answer_snippet=(out["answer"] or "")[:80],
                    grounded=out["grounded"], blocked=out["blocked"],
                    block_reason=out["block_reason"],
                    hallucination_rate=out["hallucination_rate"],
                    faithfulness_score=out["faithfulness_score"],
                    latency_s=out["latency_s"],
                ))
                log.info("    ✓ done  hall=%.2f  faith=%.2f  lat=%.2fs  blocked=%s",
                         out["hallucination_rate"], out["faithfulness_score"],
                         out["latency_s"], out["blocked"])

    log.info("All runs complete. Printing table…")
    print_table(results)

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        log.info("Results saved to: %s", args.out)

    log.info("Benchmark complete.")


if __name__ == "__main__":
    main()
