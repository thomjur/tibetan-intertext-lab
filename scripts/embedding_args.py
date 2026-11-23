#!/usr/bin/env python3
"""Shared CLI arguments for selecting the embedding backend (gemma vs TF-IDF)."""

from __future__ import annotations

import argparse
from typing import Any


def add_embedding_backend_args(parser: argparse.ArgumentParser) -> None:
    """Add --embedding-backend and --tfidf-* options to a parser."""
    parser.add_argument(
        "--embedding-backend",
        default="gemma",
        choices=["gemma", "tfidf"],
        help="Embedding backend: 'gemma' (neural, GPU-friendly) or 'tfidf' (CPU-only lexical baseline).",
    )
    parser.add_argument(
        "--tfidf-analyzer",
        default="botok",
        choices=["botok", "char"],
        help="TF-IDF tokenizer: botok word tokens (default) or char_wb n-grams (default 3-5).",
    )
    parser.add_argument("--tfidf-ngram-min", type=int, default=None, help="TF-IDF minimum n-gram size (default: 1 for botok, 3 for char).")
    parser.add_argument("--tfidf-ngram-max", type=int, default=None, help="TF-IDF maximum n-gram size (default: 1 for botok, 5 for char).")
    parser.add_argument("--tfidf-min-df", type=int, default=1, help="TF-IDF minimum document frequency.")
    parser.add_argument("--tfidf-max-df", type=float, default=1.0, help="TF-IDF maximum document frequency (proportion).")
    parser.add_argument("--tfidf-max-features", type=int, default=None, help="TF-IDF vocabulary cap.")
    parser.add_argument("--tfidf-sublinear-tf", action="store_true", help="Use sublinear TF scaling for TF-IDF.")


def embedding_backend_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Map parsed CLI args to the embedding_backend/tfidf_* keyword arguments.

    `tfidf_ngram_min`/`tfidf_ngram_max` stay None unless explicitly set, so the
    backend can apply its analyzer-specific default (botok 1-1, char 3-5).
    """
    return {
        "embedding_backend": getattr(args, "embedding_backend", "gemma"),
        "tfidf_analyzer": getattr(args, "tfidf_analyzer", "botok"),
        "tfidf_ngram_min": getattr(args, "tfidf_ngram_min", None),
        "tfidf_ngram_max": getattr(args, "tfidf_ngram_max", None),
        "tfidf_min_df": getattr(args, "tfidf_min_df", 1),
        "tfidf_max_df": getattr(args, "tfidf_max_df", 1.0),
        "tfidf_max_features": getattr(args, "tfidf_max_features", None),
        "tfidf_sublinear_tf": getattr(args, "tfidf_sublinear_tf", False),
    }
