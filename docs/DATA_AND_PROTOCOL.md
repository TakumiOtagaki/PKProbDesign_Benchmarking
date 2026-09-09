# Data and protocol

## Targets and candidate pools

The original PseudoBase++ snapshot was collected on 21 April 2026. Targets were
filtered to length at most 100 nt, strict density-2, and at least three enclosed
nucleotides for every base pair (`j - i - 1 >= 3`). Exact target dot-bracket
deduplication reduced 325 entries to 254 targets.

`targets.csv` preserves the original target, its recoloring, and the
crossing-graph decomposition. `G_big` maximizes the number of pairs in the
scaffold by orienting each connected component; `G_small` is the other lane.
Both lanes are pseudoknot-free, disjoint, and their union is the target.
`target_index` is the original 1-based deduplicated job-list index used for
PKProbDesign seeds. It is not the numeric part of the PseudoBase++ identifier.

The main comparison uses `G_big` as the common scaffold. DesiRNA was rerun on
targets recolored so that this scaffold was represented by parentheses. It is
not merely a rescore of DesiRNA designs made with another scaffold. In the
supplementary comparison, PKProbDesign and DesiRNA were run with `G_small` as
scaffold. MODENA and antaRNA do not use this decomposition during design; their
candidates were evaluated under each common assignment.

Each method contributes exactly 20 candidate occurrences per target. Duplicate
sequences are not silently deduplicated. MODENA's 50 outputs were reduced to 20
using the recorded deterministic downsampling rule and seed 20260629
(`prepare_common_gbig_comparison_candidates.py`). The released pools are the
post-downsampling pools used for comparison, not all 50 original MODENA outputs.

## Columns

All log probabilities are natural logarithms. `combined_score` is
`logP_G + logP_G_prime_given_G_S`; larger (less negative) values are better.
It is not the negative-log objective displayed as `f(S)` by the design CLI.

`candidate_uid` identifies a candidate occurrence, while `rank` records the
source rank. A representative maximizes `combined_score` within a method/target
pool; ties are broken by lower rank, then sequence, then UID. Structural
prediction distance is never used to choose representatives.

Structure columns contain dot-bracket strings. `two_stage_*` denotes
LinearFold followed by conditional SCFG2 Viterbi prediction. `viterbi_*` in the
released representatives denotes **HotKnots-guided SCFG2 Viterbi**, not a direct
standalone HotKnots prediction. `ipknot_*` denotes the separate IPknot evaluation.
Distances count the symmetric difference of base-pair sets, normalized by
sequence length when the column ends in `_normalized`.

The supplied sequences are designed candidate sequences. Natural sequences and
the original database downloads are not included. PseudoBase++ IDs and target
structures are provided to identify the benchmark and permit structural analysis.

## Zero probability and tests

The `G_small` comparison can contain `-inf` log probabilities. These recorded
values are retained: an evaluator returning negative infinity without an error
was classified as zero probability. They are not replaced with finite numbers.
The `G_small` summary distinguishes finite-only and all-record summaries;
pairwise tests use finite pairs and explicitly report excluded pair counts.

Paired tests use the original two-sided Wilcoxon normal approximation, including
tie and continuity correction. The tables include both per-metric (six pairwise
method comparisons) and global (36 tests) Bonferroni corrections. The manuscript's
main probability comparison uses the per-metric correction.

The compact release reproduces the method summaries, best counts and paired
tests for both assignments. It does not contain every exploratory run, discarded
candidate, cross-scoring experiment, optimizer trace or intermediate file from
the development repository. Historical aggregators contain optional figure and
cross-score routines that require those unbundled inputs; they are not invoked
by the snapshot entry point.

## Export provenance

`tools/export_snapshot.py` is a maintainer utility, not an installation step.
Given the original completed benchmark tree, it projects allowlisted columns
without numeric conversion, records hashes, and refuses to overwrite an existing
nonempty export. Source-relative paths in `manifest.json` identify the completed
run; they are provenance, not dependencies that readers need to obtain.
