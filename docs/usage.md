# Data, construction, and evaluation

Run the build and MLP calibration steps in the root README first.

## Input contract

Objects and queries are keyword sets. Use a single UTF-8 `Title` column:

```text
Title
learned bitmap index
keyword query index
bitmap query
```

Normalize to lowercase `[a-z0-9]+` tokens separated by spaces, without CSV
quoting, punctuation, blank records, or additional columns. Preserve object row order from
construction through evaluation. Construction rejects unknown query terms and
empty queries. C++ evaluation deduplicates terms; an unknown term is preserved
as an impossible keyword, giving an empty answer. Empty evaluation lines are skipped.

## Construction and evaluation

```bash
export LBTREE_QUERY_MODEL="$PWD/build-local/mq.pt"
python test.py --objects /path/objects.csv --workload /path/train_queries.csv \
  --eval-workload /path/eval_queries.csv --profile paper_disjoint \
  --out tree_out/custom --tradeoff 0.5 --epochs 300 --lr 0.0005 --threads 1
./build-local/lbtree tree_out/custom/vocab.json /path/objects.csv \
  /path/eval_queries.csv tree_out/custom/binary tree_out/custom/fanout --verify
```

Outputs are `vocab.json`, `binary/tree.bin`, and `fanout/tree.bin`. Use a fresh
output directory to preserve existing indexes. The latency MLP is frozen during
clustering, but gradients flow through its inputs. Flat-index reference costs are
initialized once per build, never once per node.

The default `partition_method='learned'` uses `model/paper_bbc.py`. It does not
build object-object edges, freeze the encoder after root training, or use a
straight-through surrogate. Both encoder and clustering head warm-start between
splits and remain trainable. Hash width 16, embedding width 64, a two-layer head,
ReLU, Adam, and IDF smoothing are implementation choices where the paper does not
specify constants. Samples default to at least two objects, approximately 1% per
node; depth/resource limits are explicit safeguards rather than paper criteria.

`K_graph`/`graph_policy` apply to legacy experiments, not the factorized default.
`legacy_gcn`, `fi_*`, k-means, and the other experimental partition choices are
outside the paper-method validation record. Native affine streaming is used only
with explicit analytic diagnostics. Nonlinear MLP costs use exact per-query
Python streaming decisions, including parent bitmap deltas.

Verification allocates a separate flat bitmap baseline outside query timing.
Omit `--verify` for timing after correctness checks if memory is limited.
The runtime uses ten warmup passes and one measured workload pass. Storage is an
estimate of components in MiB (bytes / 1024²), not process RSS or serialized size.

## Dataset protocols in Section VII-A

The following table specifies the paper-target dataset profile. For other
snapshots, record actual counts and the corresponding generator in
[`configs/reproducibility.json`](../configs/reproducibility.json). The
[experiment guide](reproducibility.md) describes profile selection and checks.

| Dataset | Objects | Keywords | Keyword–object pairs | Construction / evaluation queries |
|---|---:|---:|---:|---|
| MARCO | 8,841,823 | 1,004,269 | 118,760,684 | 10,000 / 10,000 disjoint, sampled from 1,010,916 available queries |
| SciDocs | 25,657 | 55,104 | 961,358 | 1,000 / 1,000 from 2,000 distinct conjunctions sampled from document co-occurrences |
| DBLP | 7,772,904 | 773,157 | 82,308,835 | 10,000 / 10,000 disjoint; 2–4 title co-occurring terms, length probabilities 0.5/0.3/0.2, sampling proportional to `f(q)^0.5` |
| SYN | 10,000,000 | 10,000 | 1,000,000,000 | 10,000 / 10,000 disjoint; 50 topics of 200 terms; each object has 90 within-topic and 10 outside-topic terms; query topics follow Zipf exponent 1.1, with 2–10 distinct within-topic terms |

The paper's environment is single-threaded Linux on an Intel Xeon Platinum
8369C at 2.90 GHz with 512 GB RAM. Record the hardware and software of each run
in its manifest when evaluating another environment.

## Reproducible construction records

`test.py` requires `--eval-workload` so that the split can be checked before
training. The default `paper_disjoint` profile compares sorted keyword sets,
rejecting even a reordered duplicate across train/test. `historical_shared`
requires identical ordered workloads and labels the result as in-workload.
`--validation-workload` must be disjoint from both. Without validation, use only
the predeclared defaults or report the whole parameter sweep; do not select
parameters by evaluation latency.

`--dataset NAME --enforce-paper-counts` also rejects deviations from the paper
target object, active-keyword, pair and query counts. Ordinary runs allow
different scales but record the differences in `run_manifest.json`. A fresh
output directory is required to preserve previous manifests and indexes.
Input counts are streamed once in preflight; vocabulary and query sets are kept
in memory for validation. The current text entry point rejects blank objects,
so a historical binary dataset containing empty objects must use a separately
documented adapter rather than silently deleting them or shifting IDs.

Model/NumPy/Python seed 42 is the default, and new repeated builds use 42/43/44.
These are the explicit seed settings of the release.
The historical object fractions are independently sampled without replacement
using `default_rng(42)` and sorted source IDs; they are not nested by construction.
Current `paper_bbc.py` performs every configured epoch at every split. The
`training_patience=8` argument retained in legacy modules does not early-stop this
paper path; it affected earlier GNN reruns.

Use the workload benchmark with `--warmup 2 --rounds 5 --order-seed 42` when
comparing with the historical baseline harness. These options are independent
of `--expected`; never silently compare its previous 2/5 setting against the
10/30 default or against `lbtree`'s single timed pass. Exported tree directories
must be evaluated with the checked evaluation workload, not the training file.
