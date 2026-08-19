"""CLI tests for the Tibetan pipeline."""

from __future__ import annotations

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_bidirectional_corpus_pairwise, run_corpus_pairwise_similarity, run_pairwise_text_similarity
from tibetan_pipeline.cli import run
from tibetan_pipeline.embeddings import EmbeddingResult
from tibetan_pipeline.pipeline import PipelineArtifacts
from tibetan_pipeline.segmenters.base import BaseSegmenter, Segment


class FakeSegmenter(BaseSegmenter):
    engine_name = "botok"

    def segment(self, text: str) -> list[Segment]:
        midpoint = max(1, len(text) // 2)
        return [
            Segment(text[:midpoint].strip(), 0, midpoint),
            Segment(text[midpoint:].strip(), midpoint, len(text)),
        ]


class CLITests(unittest.TestCase):
    def test_pairwise_cli_accepts_4bit_and_rejects_combined_quantization_flags(self) -> None:
        parser = run_pairwise_text_similarity.build_parser()
        args = parser.parse_args(
            [
                "--text-a",
                "a.txt",
                "--text-b",
                "b.txt",
                "--output-dir",
                "out",
                "--load-in-4bit",
            ]
        )

        self.assertTrue(args.load_in_4bit)
        self.assertFalse(args.load_in_8bit)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--text-a",
                    "a.txt",
                    "--text-b",
                    "b.txt",
                    "--output-dir",
                    "out",
                    "--load-in-8bit",
                    "--load-in-4bit",
                ]
            )

    def test_segmentation_only_writes_review_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            input_path.write_text("input_text\nབོད་ཡིག་། ཚིག་གཉིས་།\n", encoding="utf-8")
            args = Namespace(
                input=str(input_path),
                output_dir=str(Path(temp_dir) / "out"),
                input_format="unicode",
                engine="botok",
                text_column="input_text",
                limit=None,
                botok_cache_dir=str(Path(temp_dir) / "cache"),
                min_syllables=1,
                embed=False,
                model_id="unused",
            )

            with patch("tibetan_pipeline.cli.resolve_segmenter", return_value=FakeSegmenter()):
                artifacts = run(args)

            self.assertIsInstance(artifacts, PipelineArtifacts)
            self.assertTrue(artifacts.review_csv.exists())
            with artifacts.review_csv.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["engine"], "botok")

    def test_embedding_stage_writes_numpy_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            input_path.write_text("input_text\nབོད་ཡིག་། ཚིག་གཉིས་།\n", encoding="utf-8")
            args = Namespace(
                input=str(input_path),
                output_dir=str(Path(temp_dir) / "out"),
                input_format="unicode",
                engine="botok",
                text_column="input_text",
                limit=None,
                botok_cache_dir=str(Path(temp_dir) / "cache"),
                min_syllables=1,
                embed=True,
                model_id="fake/model",
            )

            with patch("tibetan_pipeline.cli.resolve_segmenter", return_value=FakeSegmenter()):
                with patch("tibetan_pipeline.pipeline.TextEmbedder.encode", return_value=EmbeddingResult("fake/model", np.ones((2, 3), dtype=np.float32))):
                    artifacts = run(args)

            self.assertTrue(artifacts.embeddings_npy.exists())
            self.assertTrue(artifacts.embeddings_metadata_json.exists())

    def test_corpus_pairwise_dry_run_reports_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dir_a = root / "a"
            dir_b = root / "b"
            dir_a.mkdir()
            dir_b.mkdir()
            (dir_a / "a1.txt").write_text("a1", encoding="utf-8")
            (dir_a / "a2.txt").write_text("a2", encoding="utf-8")
            (dir_b / "b1.txt").write_text("b1", encoding="utf-8")
            args = [
                "--dir-a",
                str(dir_a),
                "--dir-b",
                str(dir_b),
                "--output-dir",
                str(root / "out"),
                "--limit-a",
                "1",
                "--dry-run",
            ]

            with patch("builtins.print") as mock_print:
                exit_code = run_corpus_pairwise_similarity.main(args)

            self.assertEqual(exit_code, 0)
            printed = [call.args[0] for call in mock_print.call_args_list]
            self.assertIn("doc_count_a=1", printed)
            self.assertIn("doc_count_b=1", printed)
            self.assertIn("pair_count=1", printed)

    def test_bidirectional_corpus_pairwise_dry_run_reports_both_directions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dir_a = root / "a"
            dir_b = root / "b"
            dir_a.mkdir()
            dir_b.mkdir()
            (dir_a / "a1.txt").write_text("a1", encoding="utf-8")
            (dir_a / "a2.txt").write_text("a2", encoding="utf-8")
            (dir_b / "b1.txt").write_text("b1", encoding="utf-8")
            args = [
                "--dir-a",
                str(dir_a),
                "--dir-b",
                str(dir_b),
                "--output-dir",
                str(root / "out"),
                "--label-a",
                "SMDG",
                "--label-b",
                "Txt-18",
                "--limit-a",
                "1",
                "--dry-run",
            ]

            with patch("builtins.print") as mock_print:
                exit_code = run_bidirectional_corpus_pairwise.main(args)

            self.assertEqual(exit_code, 0)
            printed = [call.args[0] for call in mock_print.call_args_list]
            self.assertIn("forward=SMDG -> Txt-18", printed)
            self.assertIn("reverse=Txt-18 -> SMDG", printed)
            self.assertIn("doc_count_a=1", printed)
            self.assertIn("doc_count_b=1", printed)
            self.assertIn("pair_count_per_direction=1", printed)
