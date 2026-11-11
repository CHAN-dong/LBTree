# Learned Indexing for Conjunctive Keyword Queries

Static construction and evaluation of LBTree-B and LBTree, aligned with Sections
II–V of **Learned Indexing for Conjunctive Keyword Queries**, revision
`Learned_Indexing_for_Conjunctive_Keyword_Queries (1).pdf`. The default path uses sparse object–keyword–query relation
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
```

## Query cost model

Equation (2) uses a frozen three-input MLP with inputs
`(num_objects, query_keyword_count, result_count)`. Supply a compatible trained
checkpoint through `LBTREE_QUERY_MODEL`. The architecture and checkpoint loader
are in `model/query_latency.py`. `LBTREE_COST_MODE=analytic` selects the separate
affine diagnostic profile. GPU use requires an appropriate PyTorch installation.

## Construct and evaluate

Inputs have one UTF-8 `Title` column, followed by space-separated lowercase
alphanumeric keywords. Construction queries must be nonempty and use the object
vocabulary. Use disjoint construction and evaluation workloads for the paper protocol.
Python construction is exposed through `model/tree_builder.py` and
`model/tree_node_break.py`; integrate these modules in your own driver.
For an already exported index, run:

```bash
./build-local/lbtree /path/vocab.json /path/objects.csv \
  /path/eval_queries.csv /path/binary /path/fanout --verify
```

Defaults: query weight 0.5; per-node sample `max(2, ceil(n/100))`, capped at `n`;
learning rate 0.0005; 300 epochs per split; one CPU thread. Explicit depth/resource
limits are implementation safeguards. Unknown evaluation terms produce no matches,
rather than being removed from a conjunction. Estimated storage is reported in MiB.

## Experiment configuration

[`configs/reproducibility.json`](configs/reproducibility.json) defines dataset
profiles, sampling, seeds, training settings and baseline parameter grids.

| Dataset | Objects | Keywords | Keyword-object pairs | Construction / evaluation queries |
|---|---:|---:|---:|---:|
| MARCO | 8,841,823 | 1,004,269 | 118,760,684 | 10,000 / 10,000 |
| AOL | 1,467,565 | 1,253,271 | 8,954,829 | 10,000 / 10,000; at least two distinct keywords; March 1-15, 2006 |
| SciDocs | 25,657 | 55,104 | 961,358 | 1,000 / 1,000 |
| DBLP | 7,772,904 | 773,157 | 82,308,835 | 10,000 / 10,000 |
| SYN | 10,000,000 | 10,000 | 1,000,000,000 | 10,000 / 10,000 |

The five-dataset table specifies the paper-target profile. For another corpus snapshot,
record the observed counts and source hashes in the run manifest. The
[dataset profile CSV](docs/dataset_protocols.csv) also records historical inputs.

The configuration distinguishes disjoint construction/evaluation workloads from
historical shared workloads. Use a separate disjoint validation workload for
parameter selection, or use fixed defaults and report the complete sweep.

Per-node samples are uniform without replacement and have size
`min(n, max(2, ceil(n/100)))`. Defaults are seed 42, 300 epochs, learning rate
0.0005, query weight 0.5 and one CPU thread. Use seeds 42/43/44 for repeated
builds. Historical object subsets use seed 42, and SYN object/query generation
uses seeds 42/43. These explicit release defaults make new runs repeatable.

The six baselines are Bitmap, Roaring, TEB, BigSet, MILC and Bitlist. Their
defaults, grids and implementation profiles are in the [parameter CSV](docs/baseline_parameters.csv).
Inverted is an additional reference. See [the experiment guide](docs/reproducibility.md)
for shared inputs, parameter selection, timing and storage accounting.

## Scope

The release provides Python model/construction modules and the C++ index query
program. Supply a compatible three-input Mq checkpoint. External baseline
implementations follow the profiles in the parameter matrix. See
[usage](docs/usage.md), [method mapping](docs/paper_alignment.md), and
[the experiment guide](docs/reproducibility.md) for input and configuration details.
Pareto experiments are comparisons, not the default construction algorithm;
full-vector dominance additionally requires a monotone cost model.
