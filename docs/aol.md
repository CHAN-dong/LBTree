# AOL protocol

Reference: `Learned_Indexing_for_Conjunctive_Keyword_Queries (1).pdf`, Section VII-A.

| Paper-reported property | Value |
|---|---:|
| Page-title objects | 1,467,565 |
| Distinct keywords | 1,253,271 |
| Keyword-object pairs | 8,954,829 |
| Construction queries | 10,000 |
| Evaluation queries | 10,000 |

These are the paper's target counts, not counts inferred by this repository from
a downloaded corpus. Validate actual prepared inputs with the preflight command.

1. Obtain AOL-IA through https://ir-datasets.com/aol-ia.html and its access terms.
2. Extract `doc_id` and `title` from `docs_iter()`. Normalize titles to lowercase
   keyword sets; deduplicate within an object. The release text interface expects
   nonempty lowercase ASCII alphanumeric tokens. Record any source-to-input
   filtering and source IDs; do not silently relabel a different corpus snapshot.
3. Use unique queries from `queries_iter()` and event times from `qlogs_iter()`.
   Deduplicate search events by `(user_id, query_id, time)`; multiple clicked
   results must not inflate request counts. Keep aggregate counts, not user IDs.
4. Select events in `[2006-03-01, 2006-03-16)`, using source timestamps. Retain
   queries with at least two distinct normalized keywords. Do not invent a source
   timezone. Select 20,000 query keyword sets with seed 42, without replacement,
   and split into disjoint sets of 10,000 each for the release paper profile.
   The paper specifies the date range and sample sizes; seed 42 is a release
   convention. Record OOV handling before enforcing the title-corpus input contract.
5. Write normalized single-column CSV files with the `Title` header, then run:

```bash
python benchmarks/reproducibility.py --objects /path/aol_titles.csv \
  --train /path/aol_train.csv --evaluation /path/aol_eval.csv \
  --dataset AOL --profile paper_disjoint --enforce-counts \
  --out build-local/aol_manifest.json
```

Supply these same files to `test.py --dataset AOL --profile paper_disjoint`
with the frozen three-input Mq checkpoint and ordinary construction arguments.
The normalized CSV does not contain timestamps; preflight cannot independently
verify date selection. Preserve the upstream extraction manifest for that check.

For Section VII's workload drift, use March 15-May 31 logs and replace 20% of
the original evaluation workload every 15 days. Checkpoints are March 15,
March 30, April 14, April 29, May 14 and May 29 (0-100% replacement). Preserve
each cohort, sampling seed and event-counting rule. Replacement share measures
workload slots, not a guarantee that every incoming query text is previously
unseen. Do not interpret vocabulary coverage as evidence of nonempty AND results.

Use the shared six-baseline configuration, lambda sweep 0/0.25/0.5/0.75/1,
and dataset fractions 20/40/60/100% from the paper's storage table. Keep
analytically estimated Bitmap storage separate from measured quantities.
