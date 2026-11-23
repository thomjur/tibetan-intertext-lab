"""Pipeline orchestration for Tibetan sentence segmentation and embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import DEFAULT_MODEL_ID, EmbeddingBackend, TextEmbedder, create_embedder
from .tfidf import TfidfAnalyzer
from .io import InputRecord
from .normalization import normalize_text
from .segmenters import (
    BaseSegmenter,
    BotokSegmenter,
    IntellexusBotokAdapter,
    IntellexusRegexAdapter,
)


@dataclass(slots=True)
class PipelineResult:
    """Single-record segmentation output."""

    record_id: str
    original_text: str
    normalized_text: str
    segments: list[str]
    segment_spans: list[tuple[int, int]]
    source_format: str
    engine_name: str


@dataclass(slots=True)
class PipelineArtifacts:
    """Generated artifact paths."""

    review_csv: Path
    embeddings_npy: Path | None = None
    embeddings_metadata_json: Path | None = None


class TibetanPipeline:
    """High-level orchestration around normalization, segmentation, and embedding."""

    def __init__(self, segmenter: BaseSegmenter) -> None:
        self.segmenter = segmenter

    def run_segmentation(
        self,
        records: list[InputRecord],
        source_format: str = "unicode",
    ) -> list[PipelineResult]:
        results: list[PipelineResult] = []
        for record in records:
            normalized = normalize_text(record.text, source_format=source_format)
            segmented = self.segmenter.segment(normalized)
            segments = [segment.text for segment in segmented]
            segment_spans = [(segment.start, segment.end) for segment in segmented]
            results.append(
                PipelineResult(
                    record_id=record.record_id,
                    original_text=record.text,
                    normalized_text=normalized,
                    segments=segments,
                    segment_spans=segment_spans,
                    source_format=source_format,
                    engine_name=self.segmenter.engine_name,
                )
            )
        return results

    def write_embeddings(
        self,
        results: list[PipelineResult],
        output_dir: str | Path,
        model_id: str = DEFAULT_MODEL_ID,
        embedding_backend: EmbeddingBackend = "gemma",
        botok_cache_dir: str | Path | None = ".cache/botok/dialect_packs",
        tfidf_analyzer: TfidfAnalyzer = "botok",
        tfidf_ngram_min: int | None = None,
        tfidf_ngram_max: int | None = None,
        tfidf_min_df: int | float = 1,
        tfidf_max_df: int | float = 1.0,
        tfidf_max_features: int | None = None,
        tfidf_sublinear_tf: bool = False,
    ) -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sentences: list[str] = []
        metadata: list[dict[str, str | int]] = []
        for result in results:
            for segment_index, segment in enumerate(result.segments):
                sentences.append(segment)
                metadata.append(
                    {
                        "record_id": result.record_id,
                        "segment_index": segment_index,
                        "segment_text": segment,
                    }
                )

        embedder = create_embedder(
            embedding_backend,
            model_id=model_id,
            text_embedder_cls=TextEmbedder,
            botok_cache_dir=botok_cache_dir,
            tfidf_analyzer=tfidf_analyzer,
            tfidf_ngram_min=tfidf_ngram_min,
            tfidf_ngram_max=tfidf_ngram_max,
            tfidf_min_df=tfidf_min_df,
            tfidf_max_df=tfidf_max_df,
            tfidf_max_features=tfidf_max_features,
            tfidf_sublinear_tf=tfidf_sublinear_tf,
        )
        encoded = embedder.encode(sentences)
        embeddings_path = output_dir / "embeddings.npy"
        metadata_path = output_dir / "embeddings_metadata.json"
        np.save(embeddings_path, encoded.embeddings)
        metadata_path.write_text(
            json.dumps(
                {
                    "model_id": encoded.model_id,
                    "embedding_backend": embedding_backend,
                    "embedding_config": embedder.config(),
                    "rows": metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return embeddings_path, metadata_path


def resolve_segmenter(
    engine: str,
    dialect_pack_dir: str | Path | None = None,
    min_syllables: int = 4,
) -> BaseSegmenter:
    """Resolve the requested segmenter backend."""
    engine = engine.lower().strip()

    if engine in {"botok", "botok_ours"}:
        return BotokSegmenter(
            min_syllables=min_syllables,
            dialect_pack_dir=dialect_pack_dir,
        )
    if engine == "botok_intellexus":
        return IntellexusBotokAdapter(
            min_syllables=min_syllables,
            dialect_pack_dir=dialect_pack_dir,
        )
    if engine == "regex_intellexus":
        return IntellexusRegexAdapter(min_syllables=min_syllables)
    raise ValueError(f"Unsupported engine: {engine}")
