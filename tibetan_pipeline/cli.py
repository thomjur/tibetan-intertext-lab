"""CLI entrypoint for the Tibetan pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .embeddings import DEFAULT_MODEL_ID
from .io import load_records
from .pipeline import PipelineArtifacts, TibetanPipeline, resolve_segmenter
from .review import write_review_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Tibetan segmentation and embedding pipeline.")
    parser.add_argument("--input", required=True, help="Input file path (.csv, .tsv, .jsonl, or .txt).")
    parser.add_argument("--output-dir", required=True, help="Output directory for pipeline artifacts.")
    parser.add_argument(
        "--input-format",
        default="unicode",
        choices=["unicode", "wylie"],
        help="Format of the incoming Tibetan text.",
    )
    parser.add_argument(
        "--engine",
        default="botok_ours",
        choices=["botok", "botok_ours", "botok_intellexus", "regex_intellexus"],
        help="Sentence segmentation backend.",
    )
    parser.add_argument(
        "--text-column",
        default="input_text",
        help="Column name containing the source text for CSV/TSV/JSONL inputs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick inspection.")
    parser.add_argument(
        "--botok-cache-dir",
        default=".cache/botok/dialect_packs",
        help="Directory where Botok dialect packs should be stored.",
    )
    parser.add_argument("--min-syllables", type=int, default=4, help="Minimum syllables before splitting on weak shad boundaries.")
    parser.add_argument("--embed", action="store_true", help="Run the embedding stage after segmentation.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id for embeddings.")
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
        help="TF-IDF tokenizer: botok word tokens (default) or char_wb n-grams.",
    )
    parser.add_argument("--tfidf-ngram-min", type=int, default=None, help="TF-IDF minimum n-gram size (default: 1 for botok, 3 for char).")
    parser.add_argument("--tfidf-ngram-max", type=int, default=None, help="TF-IDF maximum n-gram size (default: 1 for botok, 5 for char).")
    parser.add_argument("--tfidf-min-df", type=int, default=1, help="TF-IDF minimum document frequency.")
    parser.add_argument("--tfidf-max-df", type=float, default=1.0, help="TF-IDF maximum document frequency (proportion).")
    parser.add_argument("--tfidf-max-features", type=int, default=None, help="TF-IDF vocabulary cap.")
    parser.add_argument("--tfidf-sublinear-tf", action="store_true", help="Use sublinear TF scaling for TF-IDF.")
    return parser


def run(args: argparse.Namespace) -> PipelineArtifacts:
    records = load_records(args.input, text_column=args.text_column, limit=args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segmenter = resolve_segmenter(
        args.engine,
        dialect_pack_dir=args.botok_cache_dir,
        min_syllables=args.min_syllables,
    )
    pipeline = TibetanPipeline(segmenter)
    results = pipeline.run_segmentation(records, source_format=args.input_format)

    review_path = write_review_artifact(results, output_dir / "segmentation_review.csv")
    artifacts = PipelineArtifacts(review_csv=review_path)
    if args.embed:
        embeddings_npy, metadata_json = pipeline.write_embeddings(
            results,
            output_dir=output_dir,
            model_id=args.model_id,
            embedding_backend=getattr(args, "embedding_backend", "gemma"),
            botok_cache_dir=args.botok_cache_dir,
            tfidf_analyzer=getattr(args, "tfidf_analyzer", "botok"),
            tfidf_ngram_min=getattr(args, "tfidf_ngram_min", None),
            tfidf_ngram_max=getattr(args, "tfidf_ngram_max", None),
            tfidf_min_df=getattr(args, "tfidf_min_df", 1),
            tfidf_max_df=getattr(args, "tfidf_max_df", 1.0),
            tfidf_max_features=getattr(args, "tfidf_max_features", None),
            tfidf_sublinear_tf=getattr(args, "tfidf_sublinear_tf", False),
        )
        artifacts.embeddings_npy = embeddings_npy
        artifacts.embeddings_metadata_json = metadata_json
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
