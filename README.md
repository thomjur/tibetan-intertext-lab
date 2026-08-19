# Tibetan Text Pipeline

This repository provides an end-to-end Tibetan research pipeline for:
- sentence segmentation (`botok_ours` default)
- Gemma-Mitra embeddings (`buddhist-nlp/gemma-2-mitra-e`)
- pairwise text-to-text sentence similarity (`A x B`) with global top-k matches
- corpus-level document-to-document similarity with reusable embeddings
- notebook-first experimentation via `TibetanResearchSDK`

## Project Layout
- `tibetan_pipeline/`: core package (normalization, segmenters, embeddings, pairwise core/adapters, corpus workflows, SDK)
- `scripts/`: runnable workflows (`run_tibetan_pipeline.py`, `run_pairwise_text_similarity.py`, etc.)
- `notebooks/01_research_sdk_starter.ipynb`: starter notebook for modular testing
- `notebooks/03_sdk_starter_v2.ipynb`: starter notebook with SDK cache checks and pairwise-from-embeddings workflow
- `notebooks/04_embedding_protocol_comparison.ipynb`: canonical protocol-analysis notebook comparing `query_corpus`, reversed `query_corpus`, `query_query`, and `raw_raw` on both toy passages and real cross-corpus Tibetan excerpts
- `docs/analysis/corpus_similarity_deep_dive.html`: standalone synthesis of the analysis loop, methodological lessons, and corpus-scale next steps
- `tests/`: `unittest` suite
- `output/`: local run artifacts (git-ignored)

## Install
Preferred uv setup:
```bash
uv sync --extra notebooks --extra viz
uv run python -m ipykernel install --user --name embedding-tibetan-env --display-name "Python (embedding-tibetan-env)"
```

Run tests through uv:
```bash
uv run python -m unittest discover -s tests -v
```

Legacy conda setup:
```bash
conda env create -f environment.yml
conda run -n embedding-tibetan-env python -m pip install --no-build-isolation botok pyewts
conda run -n embedding-tibetan-env python -m ipykernel install --user --name embedding-tibetan-env --display-name "Python (embedding-tibetan-env)"
conda activate embedding-tibetan-env
```

If you prefer pip in an existing environment:
```bash
python -m pip install --no-build-isolation -r requirements.txt
python -m ipykernel install --user --name embedding-tibetan-env --display-name "Python (embedding-tibetan-env)"
```

Optional but recommended for faster first run:
```bash
uv run python scripts/download_gemma_mitra.py
```

## Core Workflows

### 1) Segmentation-only pipeline
```bash
python scripts/run_tibetan_pipeline.py \
  --input data/your_input.csv \
  --output-dir output/segmentation_smoke \
  --engine botok_ours \
  --input-format unicode
```

### 2) Pairwise text similarity (two .txt files)
```bash
python scripts/run_pairwise_text_similarity.py \
  --text-a path/to/text_a.txt \
  --text-b path/to/text_b.txt \
  --output-dir output/pairwise_run \
  --engine botok_ours \
  --input-format unicode \
  --model-id buddhist-nlp/gemma-2-mitra-e \
  --device cpu \
  --top-k 100 \
  --embedding-progress batch
```

Outputs:
- `topk_pairs.csv`
- `topk_pairs.jsonl`
- `run_manifest.json`
- optional `similarity_matrix.npy` (`--save-similarity-npy`)

Manifest notes:
- includes aggregate matrix metrics such as `max_score`, `p95_score`, `mean_best_a_to_b`, and `mean_best_b_to_a`
- these metrics now come from the canonical pairwise core used by the script, SDK, and corpus workflow

### 3) Corpus-level pairwise workflow

CLI:
```bash
uv run python scripts/run_corpus_pairwise_similarity.py \
  --dir-a path/to/corpus_a \
  --dir-b path/to/corpus_b \
  --output-dir output/corpus_pairwise_run \
  --engine botok_ours \
  --input-format unicode \
  --device cuda \
  --torch-dtype bfloat16 \
  --batch-size 1 \
  --top-k 100 \
  --embedding-progress batch
```

Before spending GPU time, inspect the selected file counts:
```bash
uv run python scripts/run_corpus_pairwise_similarity.py \
  --dir-a path/to/corpus_a \
  --dir-b path/to/corpus_b \
  --output-dir output/corpus_pairwise_run \
  --dry-run
```

For a tiny smoke run:
```bash
uv run python scripts/run_corpus_pairwise_similarity.py \
  --dir-a path/to/corpus_a \
  --dir-b path/to/corpus_b \
  --output-dir output/corpus_pairwise_smoke \
  --limit-a 2 \
  --limit-b 2
```

Python API:
```python
from tibetan_pipeline.corpus_pairwise import run_corpus_pairwise_similarity

artifacts = run_corpus_pairwise_similarity(
    dir_a="path/to/corpus_a",
    dir_b="path/to/corpus_b",
    output_dir="output/corpus_pairwise_run",
    engine="botok_ours",
    source_format="unicode",
    model_id="buddhist-nlp/gemma-2-mitra-e",
    device="cpu",
    top_k=100,
)
```

Corpus outputs:
- `documents_a.csv` and `documents_b.csv`
- one `pairs/<pair_id>/` directory per document pair
- `document_pair_summary.csv`
- `corpus_manifest.json`

Each document directory includes:
- `<doc_id>_sentences.csv`, mapping sentence indices to text and source spans
- `<doc_id>_embeddings.npy`, the persisted sentence embedding matrix for that document

Each pair directory includes:
- `sentences_a.csv` and `sentences_b.csv` with `sentence_index`, `sentence_text`, `start`, and `end`
- `similarity_matrix.npy`, the full durable score matrix
- `topk_pairs.csv` and `topk_pairs.jsonl`, generated convenience views
- `pair_manifest.json`, with artifact paths and aggregate matrix metrics

The initial `top_k` is not a hard analytical limit. Because the full matrix and sentence indexes are stored, you can regenerate any later top-k view without re-segmenting or re-embedding:

```bash
uv run python scripts/export_pair_topk.py \
  --pair-dir output/corpus_pairwise_run/pairs/A001__B001 \
  --k 250
```

Use `--mode` to choose how repeated sentence matches are handled:
- `raw`: global top-k matrix cells; allows repeated A and B sentences
- `unique_a`: each A sentence appears at most once
- `unique_b`: each B sentence appears at most once
- `unique_both`: greedy one-to-one sentence matches
- `diverse_both`: one-to-one matches with nearby sentence-index clumps suppressed

```bash
uv run python scripts/export_pair_topk.py \
  --pair-dir output/corpus_pairwise_run/pairs/A001__B001 \
  --k 100 \
  --mode unique_both
```

Generate a local interactive report for a completed corpus run:
```bash
uv run python scripts/generate_corpus_pairwise_report.py \
  --run-dir output/corpus_pairwise_run
```

The report writes `report/index.html` and `report/report_data.js` under the run directory. It includes run metrics, document embedding metadata, a corpus-level heatmap, per-pair downsampled sentence heatmaps, and interactive top-k match browsing across the same raw/unique/diverse modes.

### 4) Bidirectional corpus workflow

Run both prompt directions and generate the forward, reverse, and synthesis reports:
```bash
uv run python scripts/run_bidirectional_corpus_pairwise.py \
  --dir-a path/to/SMDG \
  --dir-b path/to/Txt-18 \
  --label-a SMDG \
  --label-b Txt-18 \
  --output-dir output/corpus_pairwise_bidirectional \
  --device cuda \
  --torch-dtype bfloat16 \
  --batch-size 1 \
  --top-k 100
```

No-cost planning check:
```bash
uv run python scripts/run_bidirectional_corpus_pairwise.py \
  --dir-a path/to/SMDG \
  --dir-b path/to/Txt-18 \
  --output-dir output/corpus_pairwise_bidirectional \
  --label-a SMDG \
  --label-b Txt-18 \
  --dry-run
```

Regenerate only reports from existing forward/reverse artifacts:
```bash
uv run python scripts/run_bidirectional_corpus_pairwise.py \
  --dir-a path/to/SMDG \
  --dir-b path/to/Txt-18 \
  --output-dir output/corpus_pairwise_bidirectional \
  --label-a SMDG \
  --label-b Txt-18 \
  --reports-only
```

Python API:
```python
from tibetan_pipeline.corpus_bidirectional import run_bidirectional_corpus_pairwise

artifacts = run_bidirectional_corpus_pairwise(
    dir_a="path/to/SMDG",
    dir_b="path/to/Txt-18",
    output_dir="output/corpus_pairwise_bidirectional",
    label_a="SMDG",
    label_b="Txt-18",
    device="cuda",
    generate_reports=True,
)
```

Output layout:
- `forward/`: corpus A encoded as query side, corpus B encoded as corpus side
- `reverse/`: corpus B encoded as query side, corpus A encoded as corpus side
- `synthesis/report/`: bidirectional survival report with `synthesis.csv`, `synthesis.json`, and `index.html`

## Notebook SDK
`TibetanResearchSDK` supports segmentation, embeddings, and pairwise analysis in Jupyter.

```python
from tibetan_pipeline import TibetanResearchSDK

sdk = TibetanResearchSDK(
    engine="botok_ours",
    device="auto",
    embedding_progress="batch",  # off | batch | sentence
)

seg = sdk.segment_text(text)
emb_q = sdk.embed_sentences(seg.segments, is_query=True)
emb_c = sdk.embed_sentences(other_segments, is_query=False)
view = sdk.pairwise_from_embedding_views(emb_q, emb_c, top_k=20)
```

Useful SDK behaviors:
- SDK embedder instances are cached by heavyweight load settings (`model_id`, `device`, `torch_dtype`, `device_map`, `load_in_8bit`, `load_in_4bit`) so repeated calls in the same Python process reuse the loaded model.
- `sdk.pairwise(...)` still embeds the input texts again; use `sdk.pairwise_from_embedding_views(...)` when you already have precomputed embeddings in memory.
- pairwise SDK views now expose shared aggregate metrics and rich segment records in addition to the raw similarity matrix and top-k rows.
- The starter notebook at `notebooks/03_sdk_starter_v2.ipynb` demonstrates both flows.

## Pairwise Architecture Notes
- The canonical pairwise core lives in `tibetan_pipeline/pairwise_run.py`.
- That core is pure and stateless: rich segment metadata + embeddings in, similarity result + metrics out.
- `tibetan_pipeline/pairwise.py` is now a compatibility adapter for the two-text script surface.
- `tibetan_pipeline/corpus_pairwise.py` and `TibetanResearchSDK` both reuse the same canonical pairwise semantics.

## Embedding Backend Notes
For `buddhist-nlp/gemma-2-mitra-e`, the backend follows model-card retrieval behavior:
- query/corpus asymmetric encoding (`encode_queries` vs `encode_corpus`)
- last non-padding token pooling from final hidden state
- L2-normalized vectors for cosine similarity via dot product

Model loading controls exposed by `TibetanResearchSDK` and `TextEmbedder`:
- `device`: `auto`, `cpu`, `mps`, or `cuda`
- `torch_dtype`: `auto`, `float16`, `bfloat16`, or `float32`
- `device_map`: pass-through Transformers device placement
- `load_in_8bit`: optional 8-bit loading for compatible CUDA environments
- `load_in_4bit`: optional 4-bit loading for compatible CUDA environments

The 8-bit and 4-bit options are mutually exclusive and require a working
`bitsandbytes` installation. For example, a pairwise run can use 4-bit loading
with `--device cuda --load-in-4bit`.

Performance notes:
- The first embedding call in a fresh Python process still pays model load/materialization cost.
- Hugging Face model files are cached in the standard HF cache unless you explicitly override that outside the current runtime path.
- Pairwise on long texts is dominated by embedding time, not cosine similarity time.

## Tests
Install development tools:
```bash
uv sync --extra dev
```

```bash
uv run python -m unittest discover -s tests -v
```

Lint focused files:
```bash
uv run ruff check tibetan_pipeline scripts tests
```

Focused pairwise seam tests:
```bash
uv run python -m unittest tests.test_pairwise_run tests.test_pairwise tests.test_sdk -v
```

## Troubleshooting
- uv/pyewts build issues: this repo configures uv to build `pyewts` without build isolation and pins `setuptools<81`, matching the packaging workaround already proven in the conda setup.
- HF warning about unauthenticated requests: set `HF_TOKEN` for better rate limits.
- MPS OOM on Apple Silicon: start with `device="mps"` and a small `batch_size` such as `1`; if the model still does not fit, rerun with `device="cpu"` or move the workload to CUDA.
- Slow repeated notebook pairwise runs: avoid calling `sdk.pairwise(...)` after separately embedding the same texts; use `sdk.pairwise_from_embedding_views(...)` instead.
- Corpus workflows can create many pair directories quickly; start on a tiny folder slice before launching a broad all-to-all comparison.
