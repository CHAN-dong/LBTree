# Data, construction, and evaluation

Run all commands from the repository root with the Python environment activated.
The README provides the dependency and C++ build steps. No dataset download is
required for `benchmarks/train_smoke.py`; all of its inputs are synthetic.

## Input contract

Provide UTF-8 files with exactly one `Title` column. For example, `objects.csv`:

```text
Title
learned bitmap index
keyword query index
bitmap query
```

And `train_queries.csv`:

```text
Title
bitmap index
keyword query
```

Each object is a keyword set; all keywords in a query must match. Use lowercase
`[a-z0-9]+` tokens separated by spaces. Python tokenizes text using this pattern,
whereas C++ splits existing tokens without lowercasing or stripping general
punctuation. Normalizing files beforehand keeps these readers aligned. Avoid
quoted CSV fields, additional columns, empty records, and duplicate query terms.
Object IDs follow input row order, starting at zero; do not reorder or replace
objects after training. C++ ignores unknown query terms and drops queries that
become empty. This evaluates queries after vocabulary filtering, so use only
in-vocabulary terms when you need the original conjunction semantics.

## Build an index on your data

Choose a fresh output directory: construction may replace previous tree files.
The following Bash example takes paths from environment variables and uses
the current learned partitioner, exact graph policy, and analytical cost model:

```bash
export OBJECTS_CSV=/path/to/objects.csv
export TRAIN_QUERIES_CSV=/path/to/train_queries.csv
export INDEX_OUT=tree_out/custom
python - <<'PY'
import os
from pathlib import Path
import numpy as np
import torch
from utils.text_io import build_Aow_from_objects, build_Aqw_from_workload
from model.tree_builder import TreeBuilder
from model.tree_node_break import break_redundant_nodes

torch.manual_seed(42)
np.random.seed(42)
out = Path(os.environ['INDEX_OUT'])
out.mkdir(parents=True, exist_ok=True)
objects, vocab, _ = build_Aow_from_objects(
    os.environ['OBJECTS_CSV'], save_vocab_path=str(out / 'vocab.json'))
queries, _ = build_Aqw_from_workload(os.environ['TRAIN_QUERIES_CSV'], vocab)
builder = TreeBuilder(
    objects, queries, out_dir=str(out / 'binary'),
    N_sample=None, K_graph=None, epochs=300, lr=0.0005,
    balance_lambda=1e-3, min_objects=0, max_depth=100,
    cost_improve_eps=0.0, device='cpu')
root = builder.build_tree(np.arange(objects.shape[0]), np.arange(queries.shape[0]))
info = break_redundant_nodes(
    builder, root_id=root, out_dir_break=str(out / 'fanout'), copy_leaves=False)
print(info)
PY
```

Replace `/path/to/...` with your own files; these are not bundled resources.
`N_sample=None` selects approximately 1% of each node's objects, with a minimum
of two samples where possible. `K_graph=None` derives the graph neighbor count
from the sample size. Three hundred epochs is the configured training budget;
non-root training can stop early. These settings match the current `test.py`
construction settings, but this example explicitly selects CPU execution.

Outputs are `vocab.json`, `binary/tree.bin`, and `fanout/tree.bin`, with additional
construction metadata as emitted by the builder. Use an evaluation workload
separate from the training workload when assessing generalization:

```bash
export EVAL_QUERIES_CSV=/path/to/eval_queries.csv
./build-local/lbtree "$INDEX_OUT/vocab.json" "$OBJECTS_CSV" \
  "$EVAL_QUERIES_CSV" "$INDEX_OUT/binary" "$INDEX_OUT/fanout" --verify
```

Verification checks every loaded evaluation query against the flat bitmap
baseline for both trees outside the measured workload. Evaluation uses ten
warmup passes and one measured pass per tree. Reported storage is an estimate
of index components, not process memory or serialized file size. The output
labels storage as `MB`, but the calculation divides bytes by 1024 squared (MiB).
Verification adds a full flat bitmap index, which can require substantial memory
on large inputs; omit `--verify` for timing after checking correctness separately.

## Configuration and limitations

- `parameter.py` defaults to equal query and storage weights (`LAMBDA=0.5`).
  `TreeBuilder.build_tree` initializes flat-index reference costs once per build;
  `set_tradeoff(value)` changes the query weight before construction. The
  analytical coefficients are supplied constants, not newly fitted measurements.
- The main construction path uses these analytical coefficients and does not
  load `parameter.CKPT_PATH`. Cost-model fitting utilities have separate,
  machine-specific data paths and optional dependencies such as pandas and
  openpyxl; they are not part of the minimal workflow.
- `test.py` retains legacy absolute dataset paths. Edit its active object and
  workload paths and output directories before invoking `python test.py`, or
  use the environment-variable example above without editing source files.
- CMake builds the optional native streaming backend into `build-local/`, where
  Python discovers it automatically. Other build directories require
  `LBTREE_NATIVE_LIBRARY` pointing to the shared library. Set
  `LBTREE_DISABLE_NATIVE=1` to select the Python fallback.
- Use the default `partition_method='learned'` and `graph_policy='exact'` for
  the documented workflow. Some experimental alternatives still reference
  native libraries under the excluded `experiment/` directory and are not
  standalone reproduction entry points in this release.
- The smoke script fixes NumPy and PyTorch seeds to 42 and uses one CPU thread.
  It trains 256 objects with 32 keywords and 32 single-keyword queries for three
  epochs, sampling 64 objects with five graph neighbors. It can export a single
  leaf; `small_correctness` separately exercises multi-node queries.
- This release documents static index construction and query evaluation. It
  does not provide a documented end-to-end object-update or workload-drift API.
  No complete paper benchmark suite, prepared real-world datasets, or external
  baseline source trees are included. A smoke run is not evidence of paper-scale
  performance or full reproduction of the reported comparisons.

## Validation record

The source release was checked on November 3, 2026, using WSL, GCC 15.2.0,
CMake 4.2.3, Python 3.12.15, and CPU PyTorch 2.15.0. Python dependencies used
were NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, and Matplotlib 3.11.2.
These record the validation environment, not minimum supported versions.

- The CMake Release build completed for all four targets using upstream
  nlohmann JSON 3.11.3 headers staged with `benchmarks/json_dependency.py`.
- `ctest --test-dir build-local --output-on-failure` passed `small_correctness`,
  including multi-node queries compared with a brute-force object scan.
- `python benchmarks/train_smoke.py` completed training and both exports. This
  run produced single-leaf trees. The README's subsequent `lbtree ... --verify`
  invocation matched both trees against the flat baseline for all 32 queries.
- Source synchronization, Python syntax, documentation links and anchors, Bash
  example syntax, and the absence of excluded artifacts in the release were
  checked. No full-scale experiments or GPU validation were performed.
