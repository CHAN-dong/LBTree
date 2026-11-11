# Experiment guide

## Configuration and dataset profiles

The versioned [configuration](../configs/reproducibility.json) specifies the
paper-target corpus counts and generators, historical input profiles, sample
sizes, seeds, model training and six-baseline grids. The
[dataset CSV](dataset_protocols.csv) lists both sets of inputs with their
corresponding query-generation rules. Active keywords count terms appearing in
objects; vocabulary columns may additionally include query-only terms.

`paper_disjoint` uses disjoint normalized keyword sets for construction and
evaluation. `historical_shared` uses the identical ordered workload in both
roles. Select the intended profile explicitly when reporting results. Provide a
third, combination-disjoint validation workload for parameter selection, or use
the declared defaults and report complete sweeps. Evaluation-query latency is
not used to select winning configurations.

Normalize text to lowercase `[a-z0-9]+` keyword sets. Preserve object order and
source-object IDs. Corpus snapshots with empty objects require an adapter that
preserves their IDs; the text interface accepts nonempty normalized rows.
Compare observed corpus counts with the selected dataset configuration and
record the effective counts alongside each run.

## Sampling and seeds

Sample `min(n,max(2,ceil(n/100)))` objects uniformly without replacement at each
binary node. Fixed samples and sampling caps are separate configurations.
Default model seeds are 42/43/44 for repeated builds, with 42 as the single-run
default. Historical fractions use `default_rng(42).choice` without replacement,
followed by sorted source IDs; fractions are independently sampled rather than
constructed as nested subsets. SYN object/query seeds are 42/43. These values
are explicit release settings. The relation encoder trains for all configured
epochs; legacy GNN profiles record their non-root patience separately.

## Baseline configurations

| Method | Default | Grid | Implementation profile |
|---|---|---|---|
| Bitmap | Flat bitmap | None | Bitmap AND and complete result enumeration |
| Roaring | 16-bit chunks, runOptimize | None | Pinned CRoaring library |
| TEB | Rank block 512 bits | 128, 256, 512, 1024 | Author reader/rank/skip with sparse BFS builder |
| BigSet | k=16, c=2 | k: 12,14,16; c: 1,2,4 | Signature bitmap and sparse buckets, posting cutoff 32 |
| MILC | ell=160 | 80,160,320 | Core bounded DP and offset-packing profile |
| Bitlist | B=32 | 8,16,32,64,128 | Base block-pair profile with original object order |

The [parameter CSV](baseline_parameters.csv) and JSON configuration identify
source revisions and profiles. Pin CRoaring to
`c9193d3d37243635ab54f45c321ef6513202b79b`, TEB to
`a4ab056f2cefa7843b27c736833b08977b56649c`, and its DTL dependency to
`bae83545a091555e48b5495669c7adcb99fd2047`. The BigSet grid is the declared
reproduction setting. Inverted is an extra reference rather than a paper baseline.

## Measurement

Use the same selected objects, dense ID mapping, vocabulary, evaluation queries,
compiler settings, hardware and thread budget for every method. Compare complete
sorted result IDs with independent postings intersections before timing.
Record warmup passes, measured rounds and query-order seeds explicitly.
Time complete answer materialization and destruction; exclude parsing, file IO,
logging and correctness checks. Report mean and tail latency, separating empty
and nonempty answers. Repeat learned builds with the declared model seeds.

Construction includes model/clustering work, assignment, DP and index export.
Record shared input preparation and offline Mq calibration separately. Distinguish
serialized bytes, logical encoded payload and peak process memory. Use MB for
bytes/1e6 and MiB for bytes/2**20. Keep analytic estimates in separately labeled
measurement records, outside measured-latency or construction-speedup aggregates.

Record input, configuration, predictor and source hashes, source revision,
environment, effective settings, build status and index hashes with the raw
measurements. Use a fresh output directory per run.

## Current paper profile

The reference is `Learned_Indexing_for_Conjunctive_Keyword_Queries (1).pdf` (SHA-256 in the configuration). The five datasets include AOL; see [AOL preparation](aol.md). Storage-table fractions are 20%, 40%, 60% and 100%; the lambda sweep is 0, 0.25, 0.5, 0.75 and 1. Prior input profiles are retained as separate historical provenance. Paper-reported counts are targets, not a declaration that bundled fixtures reproduce the reported measurements.
