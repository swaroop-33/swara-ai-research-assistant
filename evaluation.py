"""
evaluation.py — Retrieval & RAG Evaluation Module
===================================================
Measures system performance with a set of test queries and prints metrics.

Metrics:
    - Retrieval latency (seconds)
    - Number of chunks retrieved
    - Average re-ranking score
    - RAG response time (seconds)
    - Estimated context tokens
    - Answer length (chars)

Usage:
    python evaluation.py

Or import for programmatic use:
    from evaluation import evaluate_query, run_full_evaluation
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from retriever import retrieve, RetrievedChunk
from rag_pipeline import ask

# ─── Logging ──────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/evaluation.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("evaluation")

# ─── Default test queries ──────────────────────────────────────────────────────
DEFAULT_TEST_QUERIES = [
    "What is reinforcement learning?",
    "Explain supervised learning algorithms.",
    "What are neural networks used for?",
    "Describe the attention mechanism in transformers.",
    "What is gradient descent?",
]


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    query: str
    retrieval_latency_s: float
    num_chunks_retrieved: int
    avg_rerank_score: float
    top_source: str
    top_page: int

    def display(self) -> str:
        return (
            f"  Query            : {self.query!r}\n"
            f"  Retrieval time   : {self.retrieval_latency_s:.3f}s\n"
            f"  Chunks retrieved : {self.num_chunks_retrieved}\n"
            f"  Avg rerank score : {self.avg_rerank_score:.4f}\n"
            f"  Top source       : {self.top_source} (p.{self.top_page})\n"
        )


@dataclass
class RAGMetrics:
    query: str
    total_latency_s: float
    context_tokens: int
    answer_length_chars: int
    num_sources: int
    answer_preview: str

    def display(self) -> str:
        return (
            f"  Query          : {self.query!r}\n"
            f"  Total latency  : {self.total_latency_s:.3f}s\n"
            f"  Context tokens : {self.context_tokens}\n"
            f"  Answer length  : {self.answer_length_chars} chars\n"
            f"  Num sources    : {self.num_sources}\n"
            f"  Answer preview : {self.answer_preview[:120]!r}\n"
        )


@dataclass
class EvaluationReport:
    queries_tested: int
    avg_retrieval_latency_s: float
    avg_rag_latency_s: float
    avg_chunks_retrieved: float
    avg_context_tokens: float
    avg_rerank_score: float
    retrieval_results: List[RetrievalMetrics] = field(default_factory=list)
    rag_results: List[RAGMetrics] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "EVALUATION REPORT",
            "=" * 60,
            f"  Queries tested          : {self.queries_tested}",
            f"  Avg retrieval latency   : {self.avg_retrieval_latency_s:.3f}s",
            f"  Avg RAG latency         : {self.avg_rag_latency_s:.3f}s",
            f"  Avg chunks retrieved    : {self.avg_chunks_retrieved:.1f}",
            f"  Avg context tokens      : {self.avg_context_tokens:.0f}",
            f"  Avg rerank score        : {self.avg_rerank_score:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ─── Core Evaluation Functions ────────────────────────────────────────────────

def evaluate_retrieval(query: str) -> RetrievalMetrics:
    """
    Measure retrieval performance for a single query.

    Returns a RetrievalMetrics dataclass with latency and chunk stats.
    """
    logger.info("Evaluating retrieval: %r", query)
    t0 = time.perf_counter()
    chunks: List[RetrievedChunk] = retrieve(query)
    elapsed = time.perf_counter() - t0

    avg_score = (
        sum(c.score for c in chunks) / len(chunks) if chunks else 0.0
    )
    top = chunks[0] if chunks else None

    metrics = RetrievalMetrics(
        query=query,
        retrieval_latency_s=round(elapsed, 3),
        num_chunks_retrieved=len(chunks),
        avg_rerank_score=round(avg_score, 4),
        top_source=top.source if top else "N/A",
        top_page=top.page if top else 0,
    )
    logger.info("Retrieval metrics: %s", metrics)
    return metrics


def evaluate_rag(query: str) -> RAGMetrics:
    """
    Measure full RAG pipeline performance for a single query.

    Returns a RAGMetrics dataclass.
    """
    logger.info("Evaluating full RAG: %r", query)
    result = ask(query)

    metrics = RAGMetrics(
        query=query,
        total_latency_s=result.get("latency_s", 0.0),
        context_tokens=result.get("tokens_in_context", 0),
        answer_length_chars=len(result.get("answer", "")),
        num_sources=len(result.get("sources", [])),
        answer_preview=result.get("answer", "")[:200],
    )
    logger.info("RAG metrics: %s", metrics)
    return metrics


def run_full_evaluation(
    queries: Optional[List[str]] = None,
    rag: bool = True,
) -> EvaluationReport:
    """
    Run retrieval (and optionally RAG) evaluation over a list of queries.

    Args:
        queries: List of query strings. Defaults to DEFAULT_TEST_QUERIES.
        rag:     Whether to also run the full RAG pipeline (slow — loads LLM).

    Returns:
        An EvaluationReport with per-query and aggregate metrics.
    """
    queries = queries or DEFAULT_TEST_QUERIES
    logger.info("Starting full evaluation over %d queries", len(queries))

    retrieval_results: List[RetrievalMetrics] = []
    rag_results: List[RAGMetrics] = []

    for q in queries:
        ret = evaluate_retrieval(q)
        retrieval_results.append(ret)
        print(ret.display())

        if rag:
            r = evaluate_rag(q)
            rag_results.append(r)
            print(r.display())

    # ── Aggregates ────────────────────────────────────────────────────────────
    n = len(queries)
    avg_ret_lat = sum(r.retrieval_latency_s for r in retrieval_results) / n
    avg_chunks = sum(r.num_chunks_retrieved for r in retrieval_results) / n
    avg_rerank = sum(r.avg_rerank_score for r in retrieval_results) / n

    avg_rag_lat = (
        sum(r.total_latency_s for r in rag_results) / len(rag_results)
        if rag_results else 0.0
    )
    avg_tok = (
        sum(r.context_tokens for r in rag_results) / len(rag_results)
        if rag_results else 0.0
    )

    report = EvaluationReport(
        queries_tested=n,
        avg_retrieval_latency_s=round(avg_ret_lat, 3),
        avg_rag_latency_s=round(avg_rag_lat, 3),
        avg_chunks_retrieved=round(avg_chunks, 2),
        avg_context_tokens=round(avg_tok, 1),
        avg_rerank_score=round(avg_rerank, 4),
        retrieval_results=retrieval_results,
        rag_results=rag_results,
    )

    print(report.summary())

    # Save JSON report
    report_path = Path("logs/evaluation_report.json")
    serialisable = {
        "summary": {
            "queries_tested": report.queries_tested,
            "avg_retrieval_latency_s": report.avg_retrieval_latency_s,
            "avg_rag_latency_s": report.avg_rag_latency_s,
            "avg_chunks_retrieved": report.avg_chunks_retrieved,
            "avg_context_tokens": report.avg_context_tokens,
            "avg_rerank_score": report.avg_rerank_score,
        },
        "retrieval_results": [asdict(r) for r in retrieval_results],
        "rag_results": [asdict(r) for r in rag_results],
    }
    report_path.write_text(
        json.dumps(serialisable, indent=2), encoding="utf-8"
    )
    logger.info("Evaluation report saved to %s", report_path)
    return report


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Research Assistant — Evaluation")
    parser.add_argument(
        "--no-rag", action="store_true",
        help="Only evaluate retrieval (skip LLM generation, much faster)",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Evaluate a single custom query",
    )
    args = parser.parse_args()

    test_queries = [args.query] if args.query else DEFAULT_TEST_QUERIES
    run_full_evaluation(queries=test_queries, rag=not args.no_rag)
