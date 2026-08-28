# Preprocessor Stage

## Purpose

The preprocessor converts BRIGHT, ReasonIR, and MS MARCO into stable files for
training, mining, BM25, and dense evaluation. It owns data validation, IDs, and
serialization—not model logic, mining scores, or evaluation metrics.

## End-to-End Flow

```text
MIXED TRAINING
BRIGHT documents ──> ID-to-text map ─┐
ReasonIR HQ rows ────────────────────┴──> train_hq.jsonl ──────┐
ReasonIR VL rows ───────────────────────> train_vl.jsonl ──────┼──> shared training mixture
MS MARCO triplets ─────────────────────> train_msmarco.jsonl ─┘          │
                                                                          └──> corpus + queries + qrels

BRIGHT EVALUATION
BRIGHT documents + examples ──> per-domain corpus + queries + qrels + exclusions

SEPARATE ANCE-MS REPRODUCTION
MS MARCO Tevatron corpus/train/dev ──> real-ID reproduction artifacts ──> ANCE-MS
```

A normal run loads BRIGHT, publishes the mixture, immediately derives its shared
corpus/query/qrel set, and finally writes BRIGHT evaluation files. Derivation
comes first so a domain failure cannot interrupt the dependent step.

## Inputs and Artifacts

[`config/config.yaml`](../config/config.yaml) owns dataset names, limits, seed,
domains, and paths. Paths resolve through `get_path()` and `DATA_BASE_DIR`.

| Artifact | Role |
|---|---|
| `training_mixture/train_{hq,vl,msmarco}.jsonl` | Common-schema ReasonIR HQ, ReasonIR VL, and sampled MS MARCO training components. |
| `reasonir_corpus.jsonl` | Deduplicated mixture passages encoded for hard-negative mining. |
| `train_queries.jsonl` / `train_qrels.txt` | Mixture queries and positive pairs used to mask positives while mining. |
| `<domain>_{corpus,queries}.jsonl` | Per-domain BRIGHT encoding inputs. |
| `<domain>_qrels.txt` | BRIGHT relevance judgments for one domain. |
| `<domain>_excluded.json` | For each query in that domain, document IDs that BRIGHT requires removing from its ranking; the domain itself is still evaluated. |
| MS MARCO reproduction files | Full corpus plus real-ID train/dev artifacts for the separate ANCE-MS recipe. |

Mixture rows contain `query_id`, `query`, `positive_passages`, and
`negative_passages`; each passage contains only `docid` and `text`. Corpus and
query JSONL likewise contain only the fields read by the pinned Tevatron version.

Preprocessing only preserves BRIGHT's exclusion metadata. Applying it to ranked
results belongs to the evaluation stage and must be reviewed there separately.

## Implementation Details

[`bright_loader.py`](../src/data/bright_loader.py) understands the BRIGHT source
schema. [`preprocessor.py`](../src/data/preprocessor.py) creates experiment-ready
artifacts.

### Load and extract BRIGHT

`BRIGHTLoader` performs no constructor-side download. Network work is explicit,
so failures surface at the requested operation instead of later as empty data.

| Method | Role |
|---|---|
| `load_dataset()` | Load the BRIGHT documents and configured examples. |
| `validate_example_domains_have_corpora()` | Reject example domains without a matching corpus. |
| `get_corpus()` / `get_queries()` | Normalize a domain's documents and queries. Corpus IDs must be nonblank and unambiguous. |
| `get_qrels()` | Validate `gold_ids`, require them in the corpus, and emit unique pairs with `relevance: 1`. BRIGHT provides binary gold IDs, so `1` means relevant. |
| `get_excluded_ids()` | Preserve real exclusions while dropping blanks, `N/A`, and duplicates. |
| `get_data_split()` | Return corpus, queries, qrels, and exclusions for one domain. |
| `get_all_documents_id_map()` | Resolve HQ positives across domains and reject an ID mapped to conflicting text. |

### Write standard formats

| Method | Role |
|---|---|
| `prepare_tevatron_corpus()` / `prepare_tevatron_queries()` | Write minimal Tevatron encode JSONL. |
| `prepare_pyserini_corpus()` | Write `{id, contents}` JSONL for Lucene/Pyserini. |
| `prepare_trec_qrels()` | Write exactly four TREC columns and reject IDs that would split across columns. |
| `prepare_bright_excluded()` | Write query-to-excluded-document JSON. |

Each writer uses an atomic temporary file, preventing a failed write from
publishing a truncated replacement.

### Build training records

| Method | Role |
|---|---|
| `prepare_hq_train_data()` | Resolve HQ positive document IDs through BRIGHT and assign stable source-derived IDs. |
| `prepare_vl_train_data()` | Skip the configured low-quality prefix and reject unusable passage sets. |
| `prepare_msmarco_train_data()` | Seed a local shuffle, reject empty passages, skip equal positive/negative pairs, and fill the requested usable count. |

These generators share one minimal schema. Small normalization helpers and
`_is_valid()` handle blank text, missing passages, and positive/negative overlap.

The VL cutoff is project-specific, not part of the official ReasonIR recipe. The
first 95,000 rows are dominated by label-style data (about 63,000 have one-word
positives); quality improves around row 87,000, so 95,000 is a conservative cutoff
that leaves roughly 150,000 cleaner rows. Only 37 usable retained rows look
single-token under whitespace splitting, mostly because they contain CJK text,
URLs, or hyphenated text, so a minimum-word filter would remove valid formats. The
official paper reports using all 244,970 VL rows; all experiments here instead use
the same filtered mixture, keeping method comparisons fair.

The separate reproduction methods preserve real MS MARCO document IDs:

| Method | Role |
|---|---|
| `prepare_msmarco_full_corpus()` | Export the full corpus for ANN indexing. |
| `prepare_msmarco_tevatron_train()` | Stream usable train rows and matching queries/qrels. |
| `prepare_msmarco_dev()` | Stream development queries and available judgments for MRR. |

### Validate and derive the shared set

`MIXTURE_FILES` declares the mixed recipe; `MSMARCO_ONLY_FILES` declares ANCE-MS.
A missing HQ or VL file is never inferred to mean an MS-MARCO-only experiment.

| Function | Role |
|---|---|
| `require_mixture_files()` | Require exactly the non-empty mixture files expected by the selected experiment. |
| `_load_mixture()` | Parse before writing and reject legacy schemas, duplicate query IDs, and conflicting document mappings. |
| `run_setup()` | Build corpus, queries, and qrels in staging and publish them only when all three succeed. |
| `_derive()` | Deduplicate passage text, make whitespace-containing document IDs TREC-safe, and keep corpus/qrels IDs aligned. |
| `require_derived_artifacts()` | Verify the derived set without generating or mutating data. |

Passages are deduplicated by text, with positives processed first. For example,
if positive ID `p1` and negative ID `n9` contain the same text, the corpus keeps
only `p1`, qrels continue to point to `p1`, and `n9` cannot be mined as a negative.

### Publish and orchestrate

`_generate_training_mixture()` stages all three mixture sources; `run_setup()`
stages and rolls back its three derived files. These set-level guarantees assume
a clean target. Individual BRIGHT and MS MARCO reproduction files are atomic;
ANCE-MS detects and rebuilds an incomplete reproduction set.

`main()` runs the complete flow. `--derive-only` skips downloads and source
generation. `run_setup()` refuses to overwrite any existing derived output, and
consumers call read-only validators rather than building data themselves.

## Safety Rules

- Every expected mixture file must exist and contain data; training also rejects
  unexpected JSONL files in the mixture directory.
- Rows with missing fields, duplicate query IDs, conflicting document text, or
  unusable passages fail before derived files are published.
- The MS MARCO sample is reproducible from the configured seed.
- When repeated passage text uses different IDs, one positive ID is retained and
  qrels are updated to that same ID.
- Document IDs containing whitespace are escaped consistently in the derived corpus
  and qrels, so every qrel remains an unambiguous four-column TREC row.
- The mixed-training set and the three derived files are staged so a failed clean
  build does not publish only part of the set.
- BRIGHT exclusions are validated and saved for the later evaluation stage.
- Training entry points check their required data before expensive model work.

The files do not record which exact mixture generated them. Before a full cluster
regeneration, delete the complete old mixture and all three derived files. Deleting
or replacing only one related file can combine data from different runs.

## How We Made the Stage Solid

We added tests for the important success and failure cases instead of trusting one
successful run.

[`preprocessor_test.py`](../tests/preprocessor_test.py) checks that:

- Tevatron can read every file we create.
- The same random seed selects the same MS MARCO rows.
- Bad rows, missing files, duplicate IDs, and conflicting document text are caught.
- If the same passage has a positive ID and another ID, the corpus keeps the
  positive ID so the miner does not mistake the duplicate for a negative.
- A failed write does not leave half-finished files behind, and the next try works.
- Every training pipeline checks its required files before starting.
- The separate ANCE-MS data path produces usable training rows.

For preprocessing, [`bright_exclusions_test.py`](../tests/bright_exclusions_test.py)
checks that BRIGHT exclusion IDs are cleaned and saved correctly. How evaluation
uses those IDs will be reviewed with the evaluation stage.

[`consolidation_preproc_check.py`](../scripts/dev/consolidation_preproc_check.py)
checks that a known small input still produces the same output files. These are CPU
tests; a clean full run on DelftBlue still checks downloads, storage, and paths.

## Running and Consumers

Run the complete stage:

```bash
python src/data/preprocessor.py
```

After removing the old derived triplet, rebuild it from an existing mixture:

```bash
python src/data/preprocessor.py --derive-only
```

| Consumer | Inputs |
|---|---|
| In-batch and cross-batch | Mixed-training JSONL. |
| ANCE ReasonIR | Mixture plus derived corpus, queries, and qrels. |
| GRASS, Fast-GRASS, and async Fast-GRASS | Mixture plus the derived mining/index artifacts they use. |
| ANCE-MS | Separate MS MARCO reproduction artifacts. |
| BM25 and dense evaluation | Per-domain BRIGHT corpus, queries, qrels, and exclusions. |
