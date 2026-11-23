"""Jupyter-friendly SDK surface for modular Tibetan pipeline experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .embeddings import DEFAULT_MODEL_ID, EmbeddingBackend, TextEmbedder, TorchDTypeName, create_embedder
from .tfidf import TfidfAnalyzer

if TYPE_CHECKING:
    from .tfidf import TfidfEmbedder
from .normalization import normalize_text
from .pairwise import PairMatch
from .pairwise_run import PairwiseMetrics, PairwiseSegment, make_segments, run_pairwise_similarity_core
from .pipeline import resolve_segmenter


@dataclass(slots=True)
class SegmentationView:
    """Notebook-friendly segmentation view for one source text."""

    original_text: str
    normalized_text: str
    source_format: str
    engine_name: str
    segments: list[str]
    spans: list[tuple[int, int]]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "segment_index": index,
                "start": start,
                "end": end,
                "segment_text": text,
            }
            for index, (text, (start, end)) in enumerate(zip(self.segments, self.spans))
        ]
        return pd.DataFrame(rows)


@dataclass(slots=True)
class EmbeddingView:
    """Notebook-friendly embedding view for sentence lists."""

    model_id: str
    device: str
    sentences: list[str]
    embeddings: np.ndarray

    def to_dataframe(self, include_vectors: bool = False) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for index, sentence in enumerate(self.sentences):
            row: dict[str, object] = {
                "sentence_index": index,
                "sentence_text": sentence,
            }
            if self.embeddings.size:
                row["vector_norm"] = float(np.linalg.norm(self.embeddings[index]))
                if include_vectors:
                    row["embedding"] = self.embeddings[index].tolist()
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass(slots=True)
class PairwiseView:
    """Notebook-friendly pairwise similarity outputs."""

    model_id: str
    device: str
    segments_a: list[str]
    segments_b: list[str]
    segment_records_a: list[PairwiseSegment]
    segment_records_b: list[PairwiseSegment]
    similarity_matrix: np.ndarray
    matches: list[PairMatch]
    metrics: PairwiseMetrics

    def topk_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(match) for match in self.matches])


class TibetanResearchSDK:
    """High-level SDK for segmentation, embeddings, and pairwise notebook workflows."""

    def __init__(
        self,
        *,
        engine: str = "botok_ours",
        source_format: str = "unicode",
        botok_cache_dir: str | Path | None = ".cache/botok/dialect_packs",
        min_syllables: int = 4,
        model_id: str = DEFAULT_MODEL_ID,
        embedding_backend: EmbeddingBackend = "gemma",
        batch_size: int = 8,
        device: Literal["auto", "cpu", "mps", "cuda"] = "auto",
        embedding_progress: Literal["off", "batch", "sentence"] = "off",
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool = False,
        low_cpu_mem_usage: bool | None = None,
        tfidf_analyzer: TfidfAnalyzer = "botok",
        tfidf_ngram_min: int | None = None,
        tfidf_ngram_max: int | None = None,
        tfidf_min_df: int | float = 1,
        tfidf_max_df: int | float = 1.0,
        tfidf_max_features: int | None = None,
        tfidf_sublinear_tf: bool = False,
    ) -> None:
        self.engine = engine
        self.source_format = source_format
        self.botok_cache_dir = botok_cache_dir
        self.min_syllables = min_syllables
        self.model_id = model_id
        self.embedding_backend = embedding_backend
        self.batch_size = batch_size
        self.device = device
        self.embedding_progress = embedding_progress
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.tfidf_analyzer = tfidf_analyzer
        self.tfidf_ngram_min = tfidf_ngram_min
        self.tfidf_ngram_max = tfidf_ngram_max
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_sublinear_tf = tfidf_sublinear_tf
        self._segmenter = resolve_segmenter(
            engine=engine,
            dialect_pack_dir=botok_cache_dir,
            min_syllables=min_syllables,
        )
        self._embedders: dict[tuple[object, ...], TextEmbedder | TfidfEmbedder] = {}

    def segment_text(self, text: str, *, source_format: str | None = None) -> SegmentationView:
        source_format = source_format or self.source_format
        normalized = normalize_text(text, source_format=source_format)
        segmented = self._segmenter.segment(normalized)
        segments = [segment.text for segment in segmented]
        spans = [(segment.start, segment.end) for segment in segmented]
        return SegmentationView(
            original_text=text,
            normalized_text=normalized,
            source_format=source_format,
            engine_name=self._segmenter.engine_name,
            segments=segments,
            spans=spans,
        )

    def embed_sentences(
        self,
        sentences: list[str],
        *,
        model_id: str | None = None,
        embedding_backend: EmbeddingBackend | None = None,
        batch_size: int | None = None,
        device: Literal["auto", "cpu", "mps", "cuda"] | None = None,
        embedding_progress: Literal["off", "batch", "sentence"] | None = None,
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool | None = None,
        low_cpu_mem_usage: bool | None = None,
        is_query: bool = False,
    ) -> EmbeddingView:
        model_id = model_id or self.model_id
        backend = embedding_backend or self.embedding_backend
        batch_size = batch_size if batch_size is not None else self.batch_size
        device = device or self.device
        embedding_progress = embedding_progress or self.embedding_progress
        torch_dtype = torch_dtype if torch_dtype is not None else self.torch_dtype
        device_map = device_map if device_map is not None else self.device_map
        load_in_8bit = load_in_8bit if load_in_8bit is not None else self.load_in_8bit
        low_cpu_mem_usage = low_cpu_mem_usage if low_cpu_mem_usage is not None else self.low_cpu_mem_usage
        embedder = self._get_embedder(
            model_id=model_id,
            embedding_backend=backend,
            batch_size=batch_size,
            device=device,
            embedding_progress=embedding_progress,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        encoded = embedder.encode_queries(sentences) if is_query else embedder.encode_corpus(sentences)
        return EmbeddingView(
            model_id=encoded.model_id,
            device="cpu" if backend == "tfidf" else device,
            sentences=sentences,
            embeddings=encoded.embeddings,
        )

    def embed_corpus_sides(
        self,
        documents_a: list[list[str]],
        documents_b: list[list[str]],
        *,
        model_id: str | None = None,
        embedding_backend: EmbeddingBackend | None = None,
        batch_size: int | None = None,
        device: Literal["auto", "cpu", "mps", "cuda"] | None = None,
        embedding_progress: Literal["off", "batch", "sentence"] | None = None,
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool | None = None,
        low_cpu_mem_usage: bool | None = None,
    ) -> tuple[list[EmbeddingView], list[EmbeddingView]]:
        """Embed two sides of a corpus run.

        For the TF-IDF backend this performs one joint fit over all sentences of
        both sides so every document shares the same vector space. For neural
        backends each document is encoded independently (A as query side, B as
        corpus side).
        """
        backend = embedding_backend or self.embedding_backend
        if backend == "tfidf":
            embedder = self._get_embedder(
                model_id=model_id or self.model_id,
                embedding_backend="tfidf",
                batch_size=batch_size if batch_size is not None else self.batch_size,
                device=device or self.device,
                embedding_progress=embedding_progress or self.embedding_progress,
                torch_dtype=torch_dtype if torch_dtype is not None else self.torch_dtype,
                device_map=device_map if device_map is not None else self.device_map,
                load_in_8bit=load_in_8bit if load_in_8bit is not None else self.load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage if low_cpu_mem_usage is not None else self.low_cpu_mem_usage,
            )
            encoded = embedder.encode_document_lists(list(documents_a) + list(documents_b))
            split = len(documents_a)
            views: list[EmbeddingView] = []
            for document, result in zip(list(documents_a) + list(documents_b), encoded):
                views.append(
                    EmbeddingView(
                        model_id=result.model_id,
                        device="cpu",
                        sentences=document,
                        embeddings=result.embeddings,
                    )
                )
            return views[:split], views[split:]
        views_a = [
            self.embed_sentences(
                document,
                model_id=model_id,
                embedding_backend=backend,
                batch_size=batch_size,
                device=device,
                embedding_progress=embedding_progress,
                torch_dtype=torch_dtype,
                device_map=device_map,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                is_query=True,
            )
            for document in documents_a
        ]
        views_b = [
            self.embed_sentences(
                document,
                model_id=model_id,
                embedding_backend=backend,
                batch_size=batch_size,
                device=device,
                embedding_progress=embedding_progress,
                torch_dtype=torch_dtype,
                device_map=device_map,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                is_query=False,
            )
            for document in documents_b
        ]
        return views_a, views_b

    def pairwise(
        self,
        text_a: str,
        text_b: str,
        *,
        top_k: int = 100,
        model_id: str | None = None,
        batch_size: int | None = None,
        device: Literal["auto", "cpu", "mps", "cuda"] | None = None,
        embedding_progress: Literal["off", "batch", "sentence"] | None = None,
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool | None = None,
        low_cpu_mem_usage: bool | None = None,
    ) -> PairwiseView:
        sentences_a = self._segment_text_to_sentences(text_a)
        sentences_b = self._segment_text_to_sentences(text_b)
        return self.pairwise_from_sentences(
            sentences_a,
            sentences_b,
            top_k=top_k,
            model_id=model_id,
            batch_size=batch_size,
            device=device,
            embedding_progress=embedding_progress,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    def pairwise_from_sentences(
        self,
        sentences_a: list[str],
        sentences_b: list[str],
        *,
        top_k: int = 100,
        model_id: str | None = None,
        batch_size: int | None = None,
        device: Literal["auto", "cpu", "mps", "cuda"] | None = None,
        embedding_progress: Literal["off", "batch", "sentence"] | None = None,
        torch_dtype: TorchDTypeName | None = None,
        device_map: str | dict[str, int | str] | None = None,
        load_in_8bit: bool | None = None,
        low_cpu_mem_usage: bool | None = None,
    ) -> PairwiseView:
        if self.embedding_backend == "tfidf":
            # TF-IDF needs one joint fit over both sentence lists.
            embedder = self._get_embedder(
                model_id=model_id or self.model_id,
                embedding_backend="tfidf",
                batch_size=batch_size if batch_size is not None else self.batch_size,
                device=device or self.device,
                embedding_progress=embedding_progress or self.embedding_progress,
                torch_dtype=torch_dtype if torch_dtype is not None else self.torch_dtype,
                device_map=device_map if device_map is not None else self.device_map,
                load_in_8bit=load_in_8bit if load_in_8bit is not None else self.load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage if low_cpu_mem_usage is not None else self.low_cpu_mem_usage,
            )
            encoded_a, encoded_b = embedder.encode_pair(sentences_a, sentences_b)
            embedding_a = EmbeddingView(
                model_id=encoded_a.model_id,
                device="cpu",
                sentences=sentences_a,
                embeddings=encoded_a.embeddings,
            )
            embedding_b = EmbeddingView(
                model_id=encoded_b.model_id,
                device="cpu",
                sentences=sentences_b,
                embeddings=encoded_b.embeddings,
            )
        else:
            embedding_a = self.embed_sentences(
                sentences_a,
                model_id=model_id,
                batch_size=batch_size,
                device=device,
                embedding_progress=embedding_progress,
                torch_dtype=torch_dtype,
                device_map=device_map,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                is_query=True,
            )
            embedding_b = self.embed_sentences(
                sentences_b,
                model_id=model_id,
                batch_size=batch_size,
                device=device,
                embedding_progress=embedding_progress,
                torch_dtype=torch_dtype,
                device_map=device_map,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                is_query=False,
            )
        result = run_pairwise_similarity_core(
            make_segments(sentences_a),
            embedding_a.embeddings,
            make_segments(sentences_b),
            embedding_b.embeddings,
            top_k=top_k,
        )
        return PairwiseView(
            model_id=embedding_a.model_id,
            device=embedding_a.device,
            segments_a=sentences_a,
            segments_b=sentences_b,
            segment_records_a=result.segments_a,
            segment_records_b=result.segments_b,
            similarity_matrix=result.similarity_matrix,
            matches=_compatibility_matches(result),
            metrics=result.metrics,
        )

    def pairwise_from_embedding_views(
        self,
        embedding_a: EmbeddingView,
        embedding_b: EmbeddingView,
        *,
        top_k: int = 100,
    ) -> PairwiseView:
        if embedding_a.model_id != embedding_b.model_id:
            raise ValueError("Embedding views must use the same model_id.")
        if embedding_a.embeddings.ndim != 2 or embedding_b.embeddings.ndim != 2:
            raise ValueError("Embedding views must contain rank-2 embedding arrays.")
        if embedding_a.embeddings.shape[0] != len(embedding_a.sentences):
            raise ValueError("Embedding view A row count must match its sentence count.")
        if embedding_b.embeddings.shape[0] != len(embedding_b.sentences):
            raise ValueError("Embedding view B row count must match its sentence count.")

        result = run_pairwise_similarity_core(
            make_segments(embedding_a.sentences),
            embedding_a.embeddings,
            make_segments(embedding_b.sentences),
            embedding_b.embeddings,
            top_k=top_k,
        )
        return PairwiseView(
            model_id=embedding_a.model_id,
            device=embedding_a.device,
            segments_a=embedding_a.sentences,
            segments_b=embedding_b.sentences,
            segment_records_a=result.segments_a,
            segment_records_b=result.segments_b,
            similarity_matrix=result.similarity_matrix,
            matches=_compatibility_matches(result),
            metrics=result.metrics,
        )

    def bidirectional_corpus_pairwise(self, dir_a: str | Path, dir_b: str | Path, output_dir: str | Path, **kwargs: object):
        """Run corpus pairwise analysis both directions using this SDK's defaults."""
        from .corpus_bidirectional import run_bidirectional_corpus_pairwise

        defaults = {
            "engine": self.engine,
            "source_format": self.source_format,
            "botok_cache_dir": self.botok_cache_dir,
            "min_syllables": self.min_syllables,
            "model_id": self.model_id,
            "embedding_backend": self.embedding_backend,
            "batch_size": self.batch_size,
            "device": self.device,
            "embedding_progress": self.embedding_progress,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "load_in_8bit": self.load_in_8bit,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "tfidf_analyzer": self.tfidf_analyzer,
            "tfidf_ngram_min": self.tfidf_ngram_min,
            "tfidf_ngram_max": self.tfidf_ngram_max,
            "tfidf_min_df": self.tfidf_min_df,
            "tfidf_max_df": self.tfidf_max_df,
            "tfidf_max_features": self.tfidf_max_features,
            "tfidf_sublinear_tf": self.tfidf_sublinear_tf,
        }
        defaults.update(kwargs)
        return run_bidirectional_corpus_pairwise(dir_a, dir_b, output_dir, **defaults)

    def _segment_text_to_sentences(self, text: str) -> list[str]:
        view = self.segment_text(text)
        return [segment for segment in view.segments if segment.strip()]

    def _get_embedder(
        self,
        *,
        model_id: str,
        embedding_backend: EmbeddingBackend | None = None,
        batch_size: int,
        device: Literal["auto", "cpu", "mps", "cuda"],
        embedding_progress: Literal["off", "batch", "sentence"],
        torch_dtype: TorchDTypeName | None,
        device_map: str | dict[str, int | str] | None,
        load_in_8bit: bool,
        low_cpu_mem_usage: bool | None,
    ) -> TextEmbedder | TfidfEmbedder:
        backend = embedding_backend or self.embedding_backend
        cache_key = (
            backend,
            model_id,
            device,
            torch_dtype,
            repr(device_map),
            load_in_8bit,
            low_cpu_mem_usage,
            self.tfidf_analyzer,
            self.tfidf_ngram_min,
            self.tfidf_ngram_max,
            self.tfidf_min_df,
            self.tfidf_max_df,
            self.tfidf_max_features,
            self.tfidf_sublinear_tf,
        )
        embedder = self._embedders.get(cache_key)
        if embedder is None:
            embedder = create_embedder(
                backend,
                model_id=model_id,
                text_embedder_cls=TextEmbedder,
                batch_size=batch_size,
                device=device,
                embedding_progress=embedding_progress,
                torch_dtype=torch_dtype,
                device_map=device_map,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                botok_cache_dir=self.botok_cache_dir,
                tfidf_analyzer=self.tfidf_analyzer,
                tfidf_ngram_min=self.tfidf_ngram_min,
                tfidf_ngram_max=self.tfidf_ngram_max,
                tfidf_min_df=self.tfidf_min_df,
                tfidf_max_df=self.tfidf_max_df,
                tfidf_max_features=self.tfidf_max_features,
                tfidf_sublinear_tf=self.tfidf_sublinear_tf,
            )
            self._embedders[cache_key] = embedder
        elif backend == "gemma":
            embedder.batch_size = batch_size
            embedder.embedding_progress = embedding_progress
        return embedder


def _compatibility_matches(result: object) -> list[PairMatch]:
    return [
        PairMatch(
            rank=match.rank,
            score=match.score,
            i=match.segment_a.index,
            j=match.segment_b.index,
            sentence_a=match.segment_a.text,
            sentence_b=match.segment_b.text,
        )
        for match in result.matches
    ]
