# LBTree: A Learned Index for Conjunctive Keyword Queries

Static construction and evaluation of LBTree-B and LBTree, aligned with Sections
II–V of **LBTree: A Learned Index for Conjunctive Keyword Queries**, revision
`LBTree_icde (3).pdf`. The default path uses sparse object–keyword–query relation
encoding, the product soft-OR cost objective, cost-driven streaming assignment,
and node-breaking DP with per-query access records.

## Build

Use Linux/WSL, Python 3.12, CMake 3.14+, and a C++17 compiler:

```bash
git clone https://github.com/CHAN-dong/LBTree.git
cd LBTree
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release
cmake --build build-local -j 4
ctest --test-dir build-local --output-on-failure
```

If JSON headers cannot be downloaded, run `python -m pip install nlohmann_json==3.11.3`
and `python benchmarks/json_dependency.py`, then configure again.

## Query latency calibration

Equation (2) requires an offline-trained MLP with exactly three inputs:
`(num_objects, query_keyword_count, result_count)`. The paper's trained weights
and original calibration samples are not included. Supply a compatible checkpoint
or collect measurements and train on the machine being evaluated:

```bash
./build-local/lbtree_calibrate build-local/latency.csv 100000 65536
python benchmarks/train_query_model.py build-local/latency.csv build-local/mq.pt
export LBTREE_QUERY_MODEL="$PWD/build-local/mq.pt"
```

This collector is a local bitmap microbenchmark. Adjust its object-count range
and protocol for the deployment. The example does not reproduce the paper's
calibration or its reported results. Paper-mode construction fails if its checkpoint
is missing; `LBTREE_COST_MODE=analytic` explicitly selects an affine diagnostic,
not the paper model. GPU use requires an appropriate PyTorch installation.

## Construct and evaluate

Inputs have one UTF-8 `Title` column, followed by space-separated lowercase
alphanumeric keywords. Construction queries must be nonempty and use the object
vocabulary. Keep construction and evaluation workloads disjoint.

```bash
python test.py --objects /path/objects.csv --workload /path/train_queries.csv \
  --out tree_out/custom --query-model build-local/mq.pt
./build-local/lbtree tree_out/custom/vocab.json /path/objects.csv \
  /path/eval_queries.csv tree_out/custom/binary tree_out/custom/fanout --verify
```

Defaults: query weight 0.5; per-node sample `max(2, ceil(n/100))`, capped at `n`;
learning rate 0.0005; 300 epochs per split; one CPU thread. Explicit depth/resource
limits are implementation safeguards. Unknown evaluation terms produce no matches,
rather than being removed from a conjunction. Estimated storage is reported in MiB.

## Checks and scope

```bash
python benchmarks/check_paper_contract.py -v
python benchmarks/train_smoke.py
./build-local/lbtree benchmarks/trained/vocab.json benchmarks/trained/objects.csv \
  benchmarks/trained/queries.csv benchmarks/trained/binary benchmarks/trained/fanout --verify
```

The smoke run uses 256 objects, 32 queries, and three epochs; it may produce one
leaf. Formula checks also export a multi-node fixture. These are correctness
checks, not paper performance results.

This release does not include the complete Section VI object-update/drift
maintenance lifecycle, the paper's trained MLP weights, prepared datasets,
external baseline suites, or complete figure/table reproduction. Retained legacy
experiment modules are not the default BBC. See [usage](docs/usage.md) and the
[paper audit](docs/paper_alignment.md) for equations, dataset protocols, and scope.
