# Method-to-code mapping

The static construction and query path follows Sections II–V of
`Learned_Indexing_for_Conjunctive_Keyword_Queries (1).pdf`. The table maps formulas and algorithms to their implementation.

| Paper definition | Code |
|---|---|
| Conjunctive keyword sets, Section II | Binary CSR construction; C++ intersections; unknown terms remain unsatisfiable |
| Storage, Eq. (1) | `utils/cost_model.py`, `node_cost`, bitmap accounting |
| Offline-trained three-input Mq, Eq. (2) | `model/query_latency.py`; compatible externally trained checkpoint |
| Fixed global normalization, Eq. (4) | `parameter.py`, initialized once in `TreeBuilder.build_tree` |
| Factorized graph/encoder, Eqs. (6)–(7) | `RelationEncoder`, frequency/hash features, query propagation, normalized IDF object aggregation |
| Sigmoid and 0.5 threshold, Eqs. (8)–(9) | `model/paper_bbc.py`; probability at least 0.5 goes left |
| Soft sizes, product soft-OR, partition-specific results/gates, Eqs. (10)–(18) | `paper_loss`, `keyword_presence`, `product_gate`, exact sparse object-query hits |
| Parent plus two children, Eq. (19) | `paper_loss`, without STE, balance, or entropy terms |
| Child and parent streaming deltas, Eqs. (20)–(21) | `streaming_assign.py`; parent and child deltas, left wins ties |
| Strict cost-improving recursion, Algorithm 3 | `TreeBuilder`, trainable warm-started encoder and head |
| Per-query RV and root state 1, Eq. (22), Algorithm 4 | `tree_node_break.py`, nonlinear Mq evaluated per query |

The approximate DP keeps one lowest-cost configuration per exposed branch count, as
specified in Section V-B. Nonlinear Mq costs use the per-query RV access records.

## Execution profiles

The `learned` path uses the factorized relation encoder, product cost objective,
cost-driven Streaming-Assign and DP access records. The Mq predictor is frozen
during clustering while its input gradients remain active. Query preprocessing
preserves unknown terms as unsatisfiable conjuncts.

`LBTREE_COST_MODE=analytic` selects the affine diagnostic profile. Its predictor
and timing records are kept separate from MLP-mode experiments. Other legacy
partition choices retain their own explicit settings.

See [the experiment guide](reproducibility.md) for datasets, seeds, sample rules,
baseline implementation profiles and measurement settings.

The minimum is over retained child states. Single-state compression has no global-optimality guarantee; no Pareto replacement is enabled by default.
