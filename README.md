# LBTree: A Learned Index for Conjunctive Keyword Queries

Implementation of **LBTree: A Learned Index for Conjunctive Keyword Queries**.
LBTree organizes keyword sets using both object distributions and a historical
query workload. Cost-aware binary bitmap clustering constructs LBTree-B;
tree-based dynamic programming then removes selected internal nodes to obtain
an adaptive-fanout LBTree. Python builds and exports the indexes, and C++
evaluates conjunctive queries using the exported bitmaps.

## Setup and build

Use Linux or WSL, Python 3.12, CMake 3.14 or newer, and a C++17 compiler.
Run the following commands from the repository root:

```bash
git clone https://github.com/CHAN-dong/LBTree.git
cd LBTree
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release
cmake --build build-local -j 4
ctest --test-dir build-local --output-on-failure
```

CMake uses an installed `nlohmann_json` package (3.11+) or downloads upstream
3.11.3. For networks that cannot access GitHub releases, stage its headers first:

```bash
python -m pip install nlohmann_json==3.11.3
python benchmarks/json_dependency.py
```

Then repeat the CMake configure and build commands. GPU training is optional;
install the PyTorch build appropriate for your CUDA environment instead of CPU PyTorch.

## Minimal run

```bash
python benchmarks/train_smoke.py
./build-local/lbtree benchmarks/trained/vocab.json \
  benchmarks/trained/objects.csv benchmarks/trained/queries.csv \
  benchmarks/trained/binary benchmarks/trained/fanout --verify
```

This generates 256 synthetic objects and 32 queries, trains for three epochs
with a depth limit of two, and exports both trees as `tree.bin` files.
`--verify` compares both trees with a flat bitmap index before timing queries.
This is a workflow smoke check; it does not reproduce the paper's experiments
and may produce a single leaf if no split improves the modeled cost.

## Your data

Prepare object and workload files with a `Title` header and one space-separated
keyword set per line. Use lowercase alphanumeric tokens without CSV quoting,
punctuation, or blank rows, and preserve object row order between construction
and evaluation. The runtime drops unknown query tokens and skips queries with
no remaining tokens, so prepare evaluation queries within the object vocabulary.

Follow [the data and construction guide](docs/usage.md) to build both trees from
your own files and evaluate a separate workload:

```bash
./build-local/lbtree VOCAB_JSON OBJECTS_CSV EVAL_QUERIES_CSV \
  BINARY_TREE_DIR FANOUT_TREE_DIR --verify
```

Replace the uppercase arguments with your paths. The program reports estimated
index storage in MiB and mean query latency in nanoseconds.

## Code and reproduction scope

- `model/`: clustering, streaming assignment, binary construction, and fanout DP.
- `index/`: C++ bitmap index, tree loader, and query engine.
- `utils/`: tokenization, sparse matrices, bitmap packing, and binary export.
- `parameter.py`: query/storage weights and analytical cost coefficients.
- `benchmarks/`: small correctness, training/export, and query evaluation tools.

Datasets, paper files, experiment records, and external baseline implementations
are excluded from this source release. The commands above cover construction,
export, and query validation, rather than the paper's complete figure/table suite.
See [configuration and limitations](docs/usage.md#configuration-and-limitations)
before using the legacy `test.py` entry point or experimental options.
