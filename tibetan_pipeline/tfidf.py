"""TF-IDF vector backend for Tibetan sentence lists.

Unlike neural embedding models, TF-IDF is not stateless: the vocabulary and IDF
weights must be fitted jointly over all texts that should be comparable. Use
`fit`/`transform` (or the convenience wrappers `encode_pair` and
`encode_document_lists`) so that both sides of a comparison share one vector
space. `TfidfEmbedder` intentionally mirrors the `TextEmbedder` interface so it
can be swapped in via `create_embedder(backend="tfidf")`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .embeddings import EmbeddingResult
from .segmenters.base import TIBETAN_DOUBLE_SHAD, TIBETAN_SHAD, TER_TSHEG, TSHEG

TfidfAnalyzer = Literal["botok", "char"]

_SHAD_PATTERN = re.compile(
    "[" + re.escape(TSHEG + TIBETAN_SHAD + TIBETAN_DOUBLE_SHAD + TER_TSHEG) + "]"
)


def build_botok_tokenizer(
    botok_cache_dir: str | Path | None = ".cache/botok/dialect_packs",
) -> Callable[[str], list[str]]:
    """Build a botok word-tokenizer usable as a sklearn tokenizer function.

    Shad/tsheg marks are stripped from token edges and empty tokens are dropped,
    so TF-IDF features are clean word/syllable units consistent with the
    sentence segmentation engine.
    """
    from botok import Config, WordTokenizer

    config = Config(base_path=str(botok_cache_dir or ".cache/botok/dialect_packs"))
    word_tokenizer = WordTokenizer(config=config)

    def tokenize(text: str) -> list[str]:
        tokens: list[str] = []
        for token in word_tokenizer.tokenize(text):
            cleaned = _SHAD_PATTERN.sub("", token.text).strip()
            if cleaned:
                tokens.append(cleaned)
        return tokens

    return tokenize


class TfidfEmbedder:
    """TF-IDF embedder sharing the duck-typed TextEmbedder interface.

    `analyzer="botok"` (default) tokenizes with the same botok `WordTokenizer`
    used for segmentation; `analyzer="char"` uses character n-grams
    (`char_wb`, default 3-5), which needs no botok dialect packs.
    """

    backend = "tfidf"

    def __init__(
        self,
        *,
        analyzer: TfidfAnalyzer = "botok",
        ngram_range: tuple[int, int] | None = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_features: int | None = None,
        sublinear_tf: bool = False,
        botok_cache_dir: str | Path | None = ".cache/botok/dialect_packs",
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        if analyzer not in {"botok", "char"}:
            raise ValueError(f"Unsupported tfidf analyzer: {analyzer}")
        self.analyzer = analyzer
        if ngram_range is None:
            ngram_range = (1, 1) if analyzer == "botok" else (3, 5)
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.botok_cache_dir = botok_cache_dir
        if analyzer == "botok" and tokenizer is None:
            tokenizer = build_botok_tokenizer(botok_cache_dir)
        if analyzer == "char" and tokenizer is not None:
            raise ValueError("Custom tokenizer is only supported with analyzer='botok'.")
        self._tokenizer = tokenizer
        self._vectorizer: TfidfVectorizer | None = None
        self._empty_vocabulary = False
        self.model_id = self._build_model_id()

    def config(self) -> dict[str, Any]:
        """Backend configuration snapshot for run manifests."""
        return {
            "backend": self.backend,
            "analyzer": self.analyzer,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "vocabulary_size": self.vocabulary_size,
        }

    @property
    def vocabulary_size(self) -> int:
        if self._vectorizer is not None:
            return len(self._vectorizer.vocabulary_)
        return 0

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        """Fit vocabulary and IDF weights on the given texts."""
        kwargs: dict[str, Any] = {
            "lowercase": False,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "norm": "l2",
            "dtype": np.float32,
        }
        if self.analyzer == "botok":
            kwargs["analyzer"] = "word"
            kwargs["tokenizer"] = self._tokenizer
            kwargs["token_pattern"] = None
        else:
            kwargs["analyzer"] = "char_wb"
        self._vectorizer = TfidfVectorizer(**kwargs)
        self._empty_vocabulary = False
        if not texts:
            self._vectorizer = None
            self._empty_vocabulary = True
            return self
        try:
            self._vectorizer.fit(texts)
        except ValueError as exc:
            if "empty vocabulary" in str(exc).lower():
                self._vectorizer = None
                self._empty_vocabulary = True
            else:
                raise
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts into dense L2-normalized TF-IDF vectors."""
        if not texts:
            return np.empty((0, self.vocabulary_size), dtype=np.float32)
        if self._vectorizer is None:
            return np.zeros((len(texts), 0), dtype=np.float32)
        return self._vectorizer.transform(texts).toarray().astype(np.float32, copy=False)

    def encode(self, texts: list[str]) -> EmbeddingResult:
        """Encode texts; auto-fits on the given texts if not fitted yet."""
        if not texts:
            return EmbeddingResult(self.model_id, np.empty((0, 0), dtype=np.float32))
        if self._vectorizer is None and not self._empty_vocabulary:
            self.fit(texts)
        return EmbeddingResult(self.model_id, self.transform(texts))

    def encode_queries(self, texts: list[str]) -> EmbeddingResult:
        """TF-IDF is symmetric, so queries encode like any other text."""
        return self.encode(texts)

    def encode_corpus(self, texts: list[str]) -> EmbeddingResult:
        """TF-IDF is symmetric, so corpus passages encode like any other text."""
        return self.encode(texts)

    def encode_pair(
        self,
        texts_a: list[str],
        texts_b: list[str],
    ) -> tuple[EmbeddingResult, EmbeddingResult]:
        """Encode both sides with one joint fit so they share a vector space."""
        combined = list(texts_a) + list(texts_b)
        if not combined:
            empty = lambda: np.empty((0, 0), dtype=np.float32)  # noqa: E731
            return EmbeddingResult(self.model_id, empty()), EmbeddingResult(self.model_id, empty())
        self.fit(combined)
        return (
            EmbeddingResult(self.model_id, self.transform(texts_a)),
            EmbeddingResult(self.model_id, self.transform(texts_b)),
        )

    def encode_document_lists(
        self,
        document_lists: list[list[str]],
    ) -> list[EmbeddingResult]:
        """Encode many documents with one joint fit over all their sentences."""
        combined = [text for document in document_lists for text in document]
        if not combined:
            return [
                EmbeddingResult(self.model_id, np.empty((0, 0), dtype=np.float32))
                for _ in document_lists
            ]
        self.fit(combined)
        return [
            EmbeddingResult(self.model_id, self.transform(document))
            for document in document_lists
        ]

    def _fitted_vectorizer(self) -> TfidfVectorizer:
        if self._vectorizer is None:
            raise RuntimeError(
                "TfidfEmbedder is not fitted yet; call fit/encode/encode_pair first."
            )
        return self._vectorizer

    def vocabulary_terms(self, *, with_idf: bool = True) -> list[dict[str, Any]]:
        """Fitted vocabulary as rows of {term, index, idf?}, ordered by column index."""
        vectorizer = self._fitted_vectorizer()
        idf = vectorizer.idf_ if with_idf else None
        terms: list[dict[str, Any]] = []
        for term, index in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1]):
            row: dict[str, Any] = {"term": term, "index": int(index)}
            if idf is not None:
                row["idf"] = float(idf[index])
            terms.append(row)
        return terms

    def vector_terms(self, vector: np.ndarray) -> dict[str, float]:
        """Non-zero entries of one dense vector mapped back to {term: tf-idf weight}."""
        vectorizer = self._fitted_vectorizer()
        row = np.asarray(vector, dtype=np.float32).ravel()
        size = len(vectorizer.vocabulary_)
        if row.shape[0] != size:
            raise ValueError(
                f"Vector has {row.shape[0]} dimensions, but the fitted vocabulary has {size} terms."
            )
        inverse = {index: term for term, index in vectorizer.vocabulary_.items()}
        return {
            inverse[index]: float(row[index])
            for index in range(size)
            if row[index] != 0.0
        }

    def explain_pair(
        self,
        vector_a: np.ndarray,
        vector_b: np.ndarray,
        *,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Term-level comparison of two vectors from the same fitted vector space.

        Returns one row per term that is non-zero in either vector, with the value
        in each vector and the product (the term's contribution to the cosine
        score), sorted by descending contribution.
        """
        terms_a = self.vector_terms(vector_a)
        terms_b = self.vector_terms(vector_b)
        rows: list[dict[str, Any]] = []
        for term in sorted(set(terms_a) | set(terms_b)):
            value_a = terms_a.get(term, 0.0)
            value_b = terms_b.get(term, 0.0)
            rows.append(
                {
                    "term": term,
                    "value_a": value_a,
                    "value_b": value_b,
                    "contribution": value_a * value_b,
                }
            )
        rows.sort(key=lambda row: row["contribution"], reverse=True)
        if top_n is not None:
            rows = rows[:top_n]
        return rows

    def _build_model_id(self) -> str:
        return (
            f"tfidf:{self.analyzer}:ngram={self.ngram_range[0]}-{self.ngram_range[1]}"
            f":min_df={self.min_df}:max_df={self.max_df}"
            f":max_features={self.max_features}:sublinear_tf={self.sublinear_tf}"
        )
