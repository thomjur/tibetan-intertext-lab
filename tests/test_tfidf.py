"""Tests for the TF-IDF embedding backend and its wiring."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tibetan_pipeline.embeddings import EmbeddingResult, TextEmbedder, create_embedder
from tibetan_pipeline.pairwise import run_pairwise_similarity
from tibetan_pipeline.pairwise_run import cosine_similarity_matrix
from tibetan_pipeline.sdk import EmbeddingView, PairwiseView, SegmentationView, TibetanResearchSDK
from tibetan_pipeline.segmenters.base import BaseSegmenter, Segment
from tibetan_pipeline.tfidf import _SHAD_PATTERN, TfidfEmbedder


def _split_tokenizer(text: str) -> list[str]:
    return text.split()


def _make_tfidf(**kwargs: object) -> TfidfEmbedder:
    kwargs.setdefault("analyzer", "botok")
    kwargs.setdefault("tokenizer", _split_tokenizer)
    return TfidfEmbedder(**kwargs)  # type: ignore[arg-type]


class TfidfEmbedderTests(unittest.TestCase):
    def test_encode_pair_fits_joint_vocabulary(self) -> None:
        embedder = _make_tfidf()
        result_a, result_b = embedder.encode_pair(["alpha beta", "gamma"], ["alpha beta", "delta alpha"])
        self.assertEqual(result_a.embeddings.shape[0], 2)
        self.assertEqual(result_b.embeddings.shape[0], 2)
        # Shared vector space: both sides use the vocabulary of A + B (4 terms).
        self.assertEqual(result_a.embeddings.shape[1], result_b.embeddings.shape[1])
        self.assertEqual(result_a.embeddings.shape[1], 4)
        self.assertEqual(embedder.vocabulary_size, 4)
        self.assertTrue(result_a.model_id.startswith("tfidf:botok"))
        self.assertEqual(result_a.model_id, result_b.model_id)

    def test_encode_pair_identical_sentences_have_cosine_one(self) -> None:
        embedder = _make_tfidf()
        result_a, result_b = embedder.encode_pair(["alpha beta"], ["alpha beta", "gamma"])
        matrix = cosine_similarity_matrix(result_a.embeddings, result_b.embeddings)
        self.assertAlmostEqual(float(matrix[0, 0]), 1.0, places=5)
        self.assertLess(float(matrix[0, 1]), 0.9)

    def test_rows_are_l2_normalized(self) -> None:
        embedder = _make_tfidf()
        result = embedder.encode(["alpha beta gamma", "alpha"])
        norms = np.linalg.norm(result.embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), rtol=1e-5)

    def test_encode_document_lists_uses_one_joint_fit(self) -> None:
        embedder = _make_tfidf()
        results = embedder.encode_document_lists([["alpha beta"], ["gamma delta"]])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].embeddings.shape[1], results[1].embeddings.shape[1])
        self.assertEqual(results[0].embeddings.shape[1], 4)

    def test_empty_inputs_return_empty_arrays(self) -> None:
        embedder = _make_tfidf()
        self.assertEqual(embedder.encode([]).embeddings.shape, (0, 0))
        result_a, result_b = embedder.encode_pair([], [])
        self.assertEqual(result_a.embeddings.shape, (0, 0))
        self.assertEqual(result_b.embeddings.shape, (0, 0))
        self.assertEqual(len(embedder.encode_document_lists([[], []])), 2)

    def test_empty_vocabulary_falls_back_to_zero_vectors(self) -> None:
        embedder = _make_tfidf()
        embedder.fit(["", "   "])  # split tokenizer yields no tokens
        result = embedder.encode(["alpha beta"])
        self.assertEqual(result.embeddings.shape, (1, 0))

    def test_char_analyzer_works_without_tokenizer(self) -> None:
        embedder = TfidfEmbedder(analyzer="char")
        self.assertEqual(embedder.ngram_range, (3, 5))
        result = embedder.encode(["alpha beta", "alpha beta"])
        self.assertEqual(result.embeddings.shape[0], 2)
        self.assertGreater(result.embeddings.shape[1], 0)
        self.assertTrue(result.model_id.startswith("tfidf:char"))

    def test_char_analyzer_rejects_custom_tokenizer(self) -> None:
        with self.assertRaises(ValueError):
            TfidfEmbedder(analyzer="char", tokenizer=_split_tokenizer)

    def test_config_reports_backend_details(self) -> None:
        embedder = _make_tfidf(min_df=2, sublinear_tf=True)
        embedder.encode(["a b", "a c"])
        config = embedder.config()
        self.assertEqual(config["backend"], "tfidf")
        self.assertEqual(config["analyzer"], "botok")
        self.assertEqual(config["ngram_range"], [1, 1])
        self.assertEqual(config["min_df"], 2)
        self.assertTrue(config["sublinear_tf"])
        self.assertGreaterEqual(config["vocabulary_size"], 1)

    def test_shad_pattern_strips_tibetan_marks(self) -> None:
        self.assertEqual(_SHAD_PATTERN.sub("", " sko\u0f0d"), " sko")
        self.assertEqual(_SHAD_PATTERN.sub("", "\u0f0b\u0f0e\u0f14"), "")


class TfidfInspectionTests(unittest.TestCase):
    def test_vocabulary_terms_returns_term_index_and_idf(self) -> None:
        embedder = _make_tfidf()
        embedder.encode_pair(["alpha beta", "gamma"], ["delta alpha"])
        terms = embedder.vocabulary_terms()
        self.assertEqual({row["term"] for row in terms}, {"alpha", "beta", "gamma", "delta"})
        self.assertEqual(sorted(row["index"] for row in terms), [0, 1, 2, 3])
        self.assertTrue(all(row["idf"] > 0 for row in terms))
        plain = embedder.vocabulary_terms(with_idf=False)
        self.assertNotIn("idf", plain[0])

    def test_vocabulary_terms_requires_fit(self) -> None:
        embedder = _make_tfidf()
        with self.assertRaises(RuntimeError):
            embedder.vocabulary_terms()

    def test_vector_terms_maps_nonzero_entries(self) -> None:
        embedder = _make_tfidf()
        result_a, _ = embedder.encode_pair(["alpha beta"], ["alpha gamma"])
        terms = embedder.vector_terms(result_a.embeddings[0])
        self.assertEqual(set(terms), {"alpha", "beta"})
        self.assertAlmostEqual(sum(value * value for value in terms.values()), 1.0, places=5)
        zero = np.zeros(embedder.vocabulary_size, dtype=np.float32)
        self.assertEqual(embedder.vector_terms(zero), {})

    def test_vector_terms_rejects_wrong_dimension(self) -> None:
        embedder = _make_tfidf()
        embedder.encode(["alpha beta"])
        with self.assertRaises(ValueError):
            embedder.vector_terms(np.zeros(embedder.vocabulary_size + 1, dtype=np.float32))

    def test_vector_terms_requires_fit(self) -> None:
        embedder = _make_tfidf()
        with self.assertRaises(RuntimeError):
            embedder.vector_terms(np.zeros(1, dtype=np.float32))

    def test_explain_pair_reports_values_and_contributions(self) -> None:
        embedder = _make_tfidf()
        result_a, result_b = embedder.encode_pair(["alpha beta"], ["alpha gamma"])
        rows = embedder.explain_pair(result_a.embeddings[0], result_b.embeddings[0])
        by_term = {row["term"]: row for row in rows}
        self.assertEqual(set(by_term), {"alpha", "beta", "gamma"})
        self.assertAlmostEqual(
            by_term["alpha"]["contribution"],
            by_term["alpha"]["value_a"] * by_term["alpha"]["value_b"],
            places=6,
        )
        self.assertGreater(by_term["alpha"]["contribution"], 0.0)
        self.assertEqual(by_term["beta"]["value_b"], 0.0)
        self.assertEqual(by_term["gamma"]["value_a"], 0.0)
        contributions = [row["contribution"] for row in rows]
        self.assertEqual(contributions, sorted(contributions, reverse=True))
        self.assertEqual(len(embedder.explain_pair(result_a.embeddings[0], result_b.embeddings[0], top_n=1)), 1)


class CreateEmbedderFactoryTests(unittest.TestCase):
    def test_tfidf_backend_returns_tfidf_embedder(self) -> None:
        with patch("tibetan_pipeline.tfidf.build_botok_tokenizer", return_value=_split_tokenizer):
            embedder = create_embedder("tfidf")
        self.assertIsInstance(embedder, TfidfEmbedder)
        self.assertEqual(embedder.backend, "tfidf")

    def test_gemma_backend_returns_text_embedder(self) -> None:
        embedder = create_embedder("gemma", model_id="fake/model", text_embedder_cls=TextEmbedder, device="cpu")
        self.assertIsInstance(embedder, TextEmbedder)
        self.assertEqual(embedder.backend, "gemma")
        self.assertEqual(embedder.model_id, "fake/model")

    def test_unknown_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            create_embedder("unknown")  # type: ignore[arg-type]

    def test_text_embedder_encode_pair_keeps_query_corpus_split(self) -> None:
        embedder = TextEmbedder(model_id="fake/model", device="cpu")
        with patch.object(TextEmbedder, "encode_queries", return_value=EmbeddingResult("fake/model", np.ones((1, 2), dtype=np.float32))) as mock_q:
            with patch.object(TextEmbedder, "encode_corpus", return_value=EmbeddingResult("fake/model", np.zeros((1, 2), dtype=np.float32))) as mock_c:
                result_a, result_b = embedder.encode_pair(["a"], ["b"])
        mock_q.assert_called_once_with(["a"])
        mock_c.assert_called_once_with(["b"])
        self.assertEqual(result_a.embeddings.sum(), 2.0)
        self.assertEqual(result_b.embeddings.sum(), 0.0)


class FakeSegmenter(BaseSegmenter):
    engine_name = "botok_ours"

    def segment(self, text: str) -> list[Segment]:
        return [Segment(part, 0, len(part)) for part in text.split("|") if part.strip()]


class PairwiseTfidfWiringTests(unittest.TestCase):
    def test_run_pairwise_similarity_with_tfidf_backend(self) -> None:
        sentences_a = ["alpha beta", "gamma"]
        sentences_b = ["alpha beta", "delta"]
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tibetan_pipeline.pairwise.segment_text_to_sentences", side_effect=[sentences_a, sentences_b]):
                with patch("tibetan_pipeline.tfidf.build_botok_tokenizer", return_value=_split_tokenizer):
                    artifacts = run_pairwise_similarity(
                        text_a="unused-a",
                        text_b="unused-b",
                        output_dir=temp_dir,
                        embedding_backend="tfidf",
                        top_k=3,
                    )
            manifest = json.loads(Path(artifacts.manifest_json).read_text(encoding="utf-8"))
            self.assertEqual(manifest["embedding_backend"], "tfidf")
            self.assertEqual(manifest["embedding_config"]["backend"], "tfidf")
            self.assertEqual(manifest["embedding_config"]["analyzer"], "botok")
            self.assertTrue(manifest["model_id"].startswith("tfidf:botok"))
            self.assertAlmostEqual(manifest["max_score"], 1.0, places=5)

            with artifacts.topk_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["sentence_a"], "alpha beta")
            self.assertEqual(rows[0]["sentence_b"], "alpha beta")
            self.assertAlmostEqual(float(rows[0]["score"]), 1.0, places=5)


class FakeCorpusSDK:
    """SDK double for the TF-IDF corpus flow (joint fit over both sides)."""

    instances: list["FakeCorpusSDK"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.corpus_side_calls: list[tuple[list[list[str]], list[list[str]]]] = []
        FakeCorpusSDK.instances.append(self)

    def segment_text(self, text: str) -> SegmentationView:
        segments = [part.strip() for part in text.split("|") if part.strip()]
        spans = [(0, len(segment)) for segment in segments]
        return SegmentationView(
            original_text=text,
            normalized_text=text,
            source_format="unicode",
            engine_name="fake",
            segments=segments,
            spans=spans,
        )

    def embed_corpus_sides(
        self,
        documents_a: list[list[str]],
        documents_b: list[list[str]],
        **kwargs: object,
    ) -> tuple[list[EmbeddingView], list[EmbeddingView]]:
        self.corpus_side_calls.append((documents_a, documents_b))
        embedder = _make_tfidf()
        encoded = embedder.encode_document_lists(list(documents_a) + list(documents_b))
        split = len(documents_a)
        views = [
            EmbeddingView(model_id=result.model_id, device="cpu", sentences=doc, embeddings=result.embeddings)
            for doc, result in zip(list(documents_a) + list(documents_b), encoded)
        ]
        return views[:split], views[split:]

    def pairwise_from_embedding_views(
        self,
        embedding_a: EmbeddingView,
        embedding_b: EmbeddingView,
        *,
        top_k: int = 100,
    ) -> PairwiseView:
        from tibetan_pipeline.pairwise_run import make_segments, run_pairwise_similarity_core
        from tibetan_pipeline.pairwise import PairMatch

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
            matches=[
                PairMatch(
                    rank=match.rank,
                    score=match.score,
                    i=match.segment_a.index,
                    j=match.segment_b.index,
                    sentence_a=match.segment_a.text,
                    sentence_b=match.segment_b.text,
                )
                for match in result.matches
            ],
            metrics=result.metrics,
        )


class CorpusTfidfWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeCorpusSDK.instances.clear()

    def test_corpus_pairwise_tfidf_fits_jointly_over_both_sides(self) -> None:
        from tibetan_pipeline.corpus_pairwise import run_corpus_pairwise_similarity

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dir_a = root / "a"
            dir_b = root / "b"
            out = root / "out"
            dir_a.mkdir()
            dir_b.mkdir()
            (dir_a / "a1.txt").write_text("alpha beta|gamma", encoding="utf-8")
            (dir_b / "b1.txt").write_text("alpha beta", encoding="utf-8")
            (dir_b / "b2.txt").write_text("delta", encoding="utf-8")

            with patch("tibetan_pipeline.corpus_pairwise.TibetanResearchSDK", FakeCorpusSDK):
                artifacts = run_corpus_pairwise_similarity(
                    dir_a=dir_a,
                    dir_b=dir_b,
                    output_dir=out,
                    embedding_backend="tfidf",
                    top_k=2,
                )

            sdk = FakeCorpusSDK.instances[0]
            self.assertEqual(sdk.kwargs["embedding_backend"], "tfidf")
            self.assertEqual(len(sdk.corpus_side_calls), 1)
            docs_a, docs_b = sdk.corpus_side_calls[0]
            self.assertEqual(docs_a, [["alpha beta", "gamma"]])
            self.assertEqual(docs_b, [["alpha beta"], ["delta"]])

            manifest = json.loads(Path(artifacts["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["embedding_backend"], "tfidf")
            self.assertEqual(manifest["tfidf_config"]["analyzer"], "botok")
            self.assertEqual(manifest["doc_count_a"], 1)
            self.assertEqual(manifest["doc_count_b"], 2)
            self.assertEqual(manifest["pair_count"], 2)

            # All document embeddings share the joint TF-IDF vector space (4 terms).
            with Path(artifacts["documents_a_csv"]).open(encoding="utf-8", newline="") as handle:
                rows_a = list(csv.DictReader(handle))
            with Path(artifacts["documents_b_csv"]).open(encoding="utf-8", newline="") as handle:
                rows_b = list(csv.DictReader(handle))
            dims = {np.load(row["embeddings_npy"]).shape[1] for row in rows_a + rows_b}
            self.assertEqual(dims, {4})

            pair_manifest = json.loads((out / "pairs" / "A001__B001" / "pair_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(pair_manifest["embedding_backend"], "tfidf")
            self.assertTrue(pair_manifest["model_id"].startswith("tfidf:botok"))
            self.assertAlmostEqual(pair_manifest["max_score"], 1.0, places=5)


class SDKTfidfTests(unittest.TestCase):
    def _make_sdk(self) -> TibetanResearchSDK:
        with patch("tibetan_pipeline.sdk.resolve_segmenter", return_value=FakeSegmenter()):
            return TibetanResearchSDK(embedding_backend="tfidf", botok_cache_dir="unused")

    def test_pairwise_from_sentences_uses_joint_tfidf_fit(self) -> None:
        sdk = self._make_sdk()
        with patch("tibetan_pipeline.tfidf.build_botok_tokenizer", return_value=_split_tokenizer):
            view = sdk.pairwise_from_sentences(["alpha beta", "gamma"], ["alpha beta", "delta"], top_k=2)
        self.assertEqual(view.similarity_matrix.shape, (2, 2))
        self.assertTrue(view.model_id.startswith("tfidf:botok"))
        self.assertAlmostEqual(float(view.similarity_matrix[0, 0]), 1.0, places=5)
        self.assertEqual(view.matches[0].sentence_a, "alpha beta")
        self.assertEqual(view.matches[0].sentence_b, "alpha beta")

    def test_embed_corpus_sides_shares_vector_space(self) -> None:
        sdk = self._make_sdk()
        with patch("tibetan_pipeline.tfidf.build_botok_tokenizer", return_value=_split_tokenizer):
            views_a, views_b = sdk.embed_corpus_sides([["alpha beta"]], [["gamma"], ["alpha"]])
        self.assertEqual(len(views_a), 1)
        self.assertEqual(len(views_b), 2)
        dims = {view.embeddings.shape[1] for view in views_a + views_b}
        self.assertEqual(len(dims), 1)
        self.assertEqual(dims.pop(), 3)

    def test_embed_sentences_tfidf_single_list(self) -> None:
        sdk = self._make_sdk()
        with patch("tibetan_pipeline.tfidf.build_botok_tokenizer", return_value=_split_tokenizer):
            view = sdk.embed_sentences(["alpha beta", "gamma"])
        self.assertEqual(view.embeddings.shape, (2, 3))
        self.assertEqual(view.device, "cpu")
        self.assertTrue(view.model_id.startswith("tfidf:botok"))


class SDKTfidfInspectionTests(unittest.TestCase):
    def _make_sdk(self, backend: str = "tfidf") -> TibetanResearchSDK:
        with patch("tibetan_pipeline.sdk.resolve_segmenter", return_value=FakeSegmenter()):
            return TibetanResearchSDK(embedding_backend=backend, botok_cache_dir="unused")  # type: ignore[arg-type]

    def test_tfidf_vocabulary_after_pairwise(self) -> None:
        sdk = self._make_sdk()
        with patch("tibetan_pipeline.tfidf.build_botok_tokenizer", return_value=_split_tokenizer):
            sdk.pairwise_from_sentences(["alpha beta", "gamma"], ["alpha beta", "delta"], top_k=2)
            vocab = sdk.tfidf_vocabulary()
            top_idf = sdk.tfidf_vocabulary(top_n=2, sort_by="idf")
        self.assertEqual({row["term"] for row in vocab}, {"alpha", "beta", "gamma", "delta"})
        self.assertEqual(len(top_idf), 2)
        idfs = [row["idf"] for row in top_idf]
        self.assertEqual(idfs, sorted(idfs, reverse=True))

    def test_tfidf_vocabulary_rejects_other_backends(self) -> None:
        sdk = self._make_sdk(backend="gemma")
        with self.assertRaises(ValueError):
            sdk.tfidf_vocabulary()

    def test_tfidf_vocabulary_requires_fitted_embedder(self) -> None:
        sdk = self._make_sdk()
        with patch("tibetan_pipeline.tfidf.build_botok_tokenizer", return_value=_split_tokenizer):
            with self.assertRaises(RuntimeError):
                sdk.tfidf_vocabulary()

    def test_tfidf_vector_terms_and_explain_pair_reuse_pairwise_fit(self) -> None:
        sdk = self._make_sdk()
        with patch("tibetan_pipeline.tfidf.build_botok_tokenizer", return_value=_split_tokenizer):
            pair_view = sdk.pairwise_from_sentences(["alpha beta", "gamma"], ["alpha beta", "delta"], top_k=2)
            # Cached embedder is already jointly fitted: no refit, same vector space.
            view_a = sdk.embed_sentences(["alpha beta", "gamma"])
            view_b = sdk.embed_sentences(["alpha beta", "delta"])
            terms = sdk.tfidf_vector_terms(view_a, 0)
            rows = sdk.tfidf_explain_pair(view_a, 0, view_b, 0)
        self.assertEqual(set(terms), {"alpha", "beta"})
        self.assertEqual(rows[0]["term"], "alpha")
        self.assertGreater(rows[0]["contribution"], 0.0)
        self.assertAlmostEqual(
            sum(row["contribution"] for row in rows),
            float(pair_view.similarity_matrix[0, 0]),
            places=5,
        )

    def test_tfidf_explain_pair_rejects_non_tfidf_view(self) -> None:
        sdk = self._make_sdk()
        with patch("tibetan_pipeline.tfidf.build_botok_tokenizer", return_value=_split_tokenizer):
            view = sdk.embed_sentences(["alpha beta"])
        foreign = EmbeddingView(model_id="gemma/fake", device="cpu", sentences=["x"], embeddings=np.ones((1, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            sdk.tfidf_explain_pair(view, 0, foreign, 0)

    def test_tfidf_vector_terms_rejects_out_of_range_index(self) -> None:
        sdk = self._make_sdk()
        with patch("tibetan_pipeline.tfidf.build_botok_tokenizer", return_value=_split_tokenizer):
            view = sdk.embed_sentences(["alpha beta"])
            with self.assertRaises(IndexError):
                sdk.tfidf_vector_terms(view, 5)


if __name__ == "__main__":
    unittest.main()
