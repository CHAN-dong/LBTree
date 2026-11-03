# Alignment with LBTree_icde (3)

Audit reference: the supplied 15-page PDF, November 3, 2026. This audit concerns
the current default source path, not historical revisions or full reproduction
of the paper's experimental results.

| Paper definition | Code |
|---|---|
| Conjunctive keyword sets, Section II | Binary CSR construction; C++ intersections; unknown terms remain unsatisfiable |
| Storage, Eq. (1) | `utils/cost_model.py`, `node_cost`, bitmap accounting |
| Offline-trained three-input Mq, Eq. (2) | `model/query_latency.py`; measured collector and `benchmarks/train_query_model.py` |
| Fixed global normalization, Eq. (4) | `parameter.py`, initialized once in `TreeBuilder.build_tree` |
| Factorized graph/encoder, Eqs. (6)–(7) | `RelationEncoder`, frequency/hash features, query propagation, normalized IDF object aggregation |
| Sigmoid and 0.5 threshold, Eqs. (8)–(9) | `model/paper_bbc.py`; probability at least 0.5 goes left |
| Soft sizes, product soft-OR, partition-specific results/gates, Eqs. (10)–(18) | `paper_loss`, `keyword_presence`, `product_gate`, exact sparse object-query hits |
| Parent plus two children, Eq. (19) | `paper_loss`, without STE, balance, or entropy terms |
| Child and parent streaming deltas, Eqs. (20)–(21) | `streaming_assign.py`; left wins ties; no fabricated seed queries |
| Strict cost-improving recursion, Algorithm 3 | `TreeBuilder`, trainable warm-started encoder and head |
| Per-query RV and root state 1, Eq. (22), Algorithm 4 | `tree_node_break.py`, nonlinear Mq evaluated per query |

The DP keeps one lowest-cost configuration per fanout, as specified in Section
V-B. It is not a globally optimal search over all configurations. A scalar sum
of branch visits cannot replace per-query RV with a nonlinear Mq.

## Corrections

The previous default used object-object edges, an exponential/straight-through
keyword surrogate, unpartitioned hit counts for both children, and affine query
costs. The default now uses the factorized relation model and product objective.
Streaming includes parent cost changes. Evaluation preserves unknown conjuncts.
Dataset protocols, commands, and scope statements were updated. Old alternatives
remain explicitly identified as legacy experiments, not default paper methods.

## Boundaries and a paper ambiguity

- The supplied repository contained neither the paper's trained Mq weights nor
  its original offline measurements. Paper mode requires a compatible checkpoint;
  it never silently substitutes constants or invented pretrained weights.
- The new collector/trainer is a local calibration workflow. Its bitmap
  implementation, ranges, and samples need separate deployment validation;
  running it does not establish equivalence to the paper's cost predictor.
- Complete Section VI insertion, deletion, drift detection, and subtree
  maintenance are not supplied by this static construction API.
- Prepared datasets, full baseline/figure/table suites, paper-scale performance
  reproduction, and GPU validation are not included.
- The opening paragraph of Section IV says the method avoids incremental
  assignment, while Section IV-C explicitly specifies greedy incremental
  Streaming-Assign. The implementation follows detailed Eqs. (20)–(21) and
  Algorithm 3. The source cannot simultaneously satisfy these incompatible
  descriptions; this wording ambiguity remains in the supplied PDF.

## Local validation, November 3, 2026

Environment: WSL, GCC 15.2.0, Python 3.12.15, CPU PyTorch 2.15.0, NumPy 2.5.4,
and SciPy 1.18.1.

- C++ Release build and `small_correctness` CTest passed.
- Seven contract tests passed: dense relation-encoding comparison; product loss
  and separate child result counts; finite-difference gradients; streaming versus
  full child-plus-parent recomputation; unknown training-term rejection; missing-MLP rejection; and
  per-query DP records with multi-node export.
- Calibration with 1,200 measured samples, at most 8,192 objects, and 100 fitting
  epochs exercised Mq training/loading. This was a workflow check, not the
  paper's calibration or an evaluation of its reported accuracy.
- The three-epoch smoke run exported single-leaf trees and matched 32 C++ queries
  to a flat bitmap baseline. A separately constructed multi-level binary tree
  and its DP output matched eight queries, including unknown and duplicate terms.

These checks establish only the tested semantics. They do not prove that omitted
subsystems or paper performance results have been reproduced.
