"""
document_insights.py — Document Analysis Module
=================================================
Provides three analysis modes for a given PDF:

    1. summarize_document   — Extractive summary via LLM
    2. extract_key_topics   — Top-N keyword/topic extraction via TF-IDF
    3. extract_key_concepts — Frequency-based key phrase extraction

These are called from the Streamlit "Insights" tab.
"""

from __future__ import annotations

import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Logging ──────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/insights.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("document_insights")

# ─── Stop words ───────────────────────────────────────────────────────────────
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "and", "or", "but", "not", "this",
    "that", "it", "its", "as", "we", "you", "he", "she", "they", "i",
    "which", "who", "what", "when", "where", "how", "if", "so", "also",
    "such", "than", "then", "there", "these", "those", "their", "our",
    "figure", "table", "section", "chapter", "et", "al",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_full_text(pdf_path: str) -> str:
    """Load all pages of a PDF and concatenate into a single string."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return " ".join(d.page_content for d in docs)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]


def _clean_tokens(text: str) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [t for t in tokens if t not in _STOP]


# ─── Public API ───────────────────────────────────────────────────────────────

def summarize_document(pdf_path: str, max_sentences: int = 8) -> str:
    """
    Produce an extractive summary of the PDF by selecting the top sentences.

    Scoring: each sentence is scored by the frequency of its non-stop-word
    tokens in the full document. Top *max_sentences* are returned in order.

    Args:
        pdf_path:      Absolute or relative path to the PDF.
        max_sentences: Number of sentences to include in the summary.

    Returns:
        A multi-sentence string summary.
    """
    logger.info("Summarising '%s'", pdf_path)
    try:
        text = _load_full_text(pdf_path)
        if not text.strip():
            return "Unable to extract text from this document."

        token_freq = Counter(_clean_tokens(text))
        sents = _sentences(text)
        if not sents:
            return text[:1000]

        def _score(sent: str) -> float:
            toks = _clean_tokens(sent)
            return sum(token_freq.get(t, 0) for t in toks) / max(len(toks), 1)

        scored = sorted(enumerate(sents), key=lambda x: _score(x[1]), reverse=True)
        top_indices = sorted(i for i, _ in scored[:max_sentences])
        summary = " ".join(sents[i] for i in top_indices)
        logger.info("Summary generated (%d sentences)", len(top_indices))
        return summary
    except Exception as exc:
        logger.error("summarize_document failed: %s", exc)
        return f"Error generating summary: {exc}"


def extract_key_topics(pdf_path: str, n_topics: int = 10) -> List[str]:
    """
    Extract the top-N key topics (unigrams + bigrams) using TF-IDF.

    Args:
        pdf_path: Path to the PDF.
        n_topics: Number of topics to return.

    Returns:
        Ordered list of topic strings, most important first.
    """
    logger.info("Extracting key topics from '%s'", pdf_path)
    try:
        text = _load_full_text(pdf_path)
        if not text.strip():
            return []

        # Split into pseudo-documents (paragraphs) for TF-IDF
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
        if len(paragraphs) < 2:
            paragraphs = [text[i:i+500] for i in range(0, len(text), 500)]

        stop_list = list(_STOP)
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=stop_list,
            max_features=200,
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1

        top_indices = scores.argsort()[::-1][:n_topics]
        topics = [feature_names[i] for i in top_indices]
        logger.info("Extracted %d topics", len(topics))
        return topics
    except Exception as exc:
        logger.error("extract_key_topics failed: %s", exc)
        return []


def extract_key_concepts(
    pdf_path: str,
    n_concepts: int = 15,
) -> List[Dict[str, object]]:
    """
    Extract key concepts as the most frequent meaningful tokens.

    Returns a list of dicts: [{"concept": str, "frequency": int}, …]

    Args:
        pdf_path:   Path to the PDF.
        n_concepts: Number of concepts to return.
    """
    logger.info("Extracting key concepts from '%s'", pdf_path)
    try:
        text = _load_full_text(pdf_path)
        tokens = _clean_tokens(text)
        if not tokens:
            return []

        freq = Counter(tokens)
        top = freq.most_common(n_concepts)
        concepts = [{"concept": word, "frequency": count} for word, count in top]
        logger.info("Extracted %d concepts", len(concepts))
        return concepts
    except Exception as exc:
        logger.error("extract_key_concepts failed: %s", exc)
        return []


def analyse_document(pdf_path: str) -> Dict[str, object]:
    """
    Run all three analyses and return a combined result dict.

    Returns:
        {
            "summary":  str,
            "topics":   List[str],
            "concepts": List[{"concept": str, "frequency": int}],
        }
    """
    return {
        "summary": summarize_document(pdf_path),
        "topics": extract_key_topics(pdf_path),
        "concepts": extract_key_concepts(pdf_path),
    }
