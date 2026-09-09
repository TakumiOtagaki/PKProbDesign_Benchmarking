# HPC execution and limitations

## What is and is not reproducible from this archive

| Task | Availability |
|---|---|
| Read designed sequences, probability terms and predictions | Included CSV files; no external software |
| Recompute selection, distances, summaries and statistical tests | Supported by `scripts/analyze_snapshot.py` |
| Recompute probabilities / two-stage predictions | Requires the PKProbDesign helper binaries and DP09 parameter file |
| Rerun MODENA or antaRNA | Requires separately obtained software and compatible Linux / Python 2 environments |
| Rerun DesiRNA | Requires its own Python/ViennaRNA dependencies, separate from this analysis environment |
| Recompute HotKnots-guided SCFG2 predictions | Original integration/runtime is not fully bundled publicly; recorded predictions are included |
| Rebuild the original Apptainer images byte-for-byte | Not supported by this archive |

The original experiments ran on TSUBAME, using site modules, scheduler jobs,
scratch directories and Apptainer bind mounts. No claim is made that those
workflows can run unchanged on another cluster or on macOS. The data-analysis
entry point is independent of them. Tool revisions and recorded binary/image
hashes are listed in `tool_provenance.json`; a hash identifies an artifact but
does not make that artifact available for download.

## Recorded design settings

| Method | Settings |
|---|---|
| PKProbDesign | 100 iterations, 30 optimization samples, 30 distribution samples, 15 threads, top 20, seed = deduplicated target index |
| DesiRNA | 60 seconds, 10 replicas, exchange interval 100, 20 results, Turner 1999, `Ed-Epf:1.0`, `-acgu off`, `-nd off`, `-seed 0` |
| MODENA v0040 | IPknot backend, 50 candidates with original program defaults; deterministic reduction to 20 |
| antaRNA + pKiss | GC target 0.5, `-t 60`, `-n 20`, `-p -pkP pKiss`, MFE output |

DesiRNA `-seed 0` uses its default stochastic seeding, not a fixed reproducible
random stream. No fixed antaRNA seed was supplied by the original wrapper.
Time-limited stochastic searches may yield different sequences on rerun, even
with the same software. The published candidate pools, not an assumed exact
regeneration of stochastic searches, anchor the analysis here.

Probability evaluation used LinearPartition beam size 100 and the SCFG2 exact
runtime with explicit DP09 parameters. IPknot evaluation used version 1.1.0,
model `LinearPartition-C`, with default thresholds. HotKnots-guided evaluation
used top 20 hotspots, threshold 400 and the empty-scaffold candidate.

## External software

- [PKProbDesign](https://github.com/TakumiOtagaki/PKProbDesign): follow its own
  build instructions. Configure paths to `linearpartition_logprob`,
  `scfg2_exact_adapter`, `linearfold_predict`, and `rna_DirksPierce09.par` explicitly.
  The tested public snapshot is recorded in `tool_provenance.json`. Older
  snapshots name the main binary `samplingpkdesign`; newer ones may use
  `pkprobdesign`. No design binary is needed to analyze the supplied CSV files.
- [DesiRNA](https://github.com/fryzjergda/DesiRNA): obtain separately and follow
  its installation instructions. Do not assume this repo's minimal analysis
  environment also supplies its dependencies.
- [antaRNA](https://github.com/RobertKleinkauf/antarna): the recorded version
  requires Python 2.7; pKiss and RNAfold must also be visible at runtime.
- [IPknot](https://github.com/satoken/ipknot): native executable or Apptainer image.
  The included `scripts/wrappers/ipknot` supports both and traces backend calls
  for MODENA. Set `IPKNOT_BIN` to an absolute path, different from the wrapper
  itself, and add `scripts/wrappers` to `PATH`.
- MODENA v0040 and HotKnots: obtain from their original distributors under their
  own terms. Their executables are not redistributed in this repository.

`environments/antarna_py2.def` preserves the original Apptainer recipe as a
reference. Its old base image and some Python package specifications are not
fully pinned; rebuilding today may fail or produce a different image. It does
not install pKiss or include the antaRNA checkout. It expects the checkout to be
mounted at `/work/submodules/antarna`, with the script under `antarna/antarna.py`.
It has not been rebuilt for this release. Treat the Python 2 environment as
an isolated research dependency, not a maintained general-purpose environment.

For HPC adaptation, provide your own module setup, allocation, scratch path and
container paths. Bind the task working directory at the same path inside the
container; MODENA uses relative input files. Exposing only a host executable
directory may not be sufficient when it depends on shared libraries outside the
container. Record the actual resolved tool versions and hashes for each rerun.
This archive does not include private scheduler account names or host paths.

## Optional probability rescore of one target

After building the public PKProbDesign helpers, extract one target's candidate
rows from either `candidate_pool.csv` with a CSV reader and pass them to:

```bash
uv run --frozen python scripts/evaluate_candidate_probabilities.py \
  --input-csv work/one_target.csv --output work/one_target_probabilities.csv \
  --linearpartition-bin /absolute/path/to/linearpartition_logprob \
  --cparty-bin /absolute/path/to/scfg2_exact_adapter \
  --cparty-param-file /absolute/path/to/rna_DirksPierce09.par --beamsize 100
```

The absolute paths above are placeholders to configure, not files bundled here.
Check `probability_error` and process status before interpreting output; a failed
evaluation is not a valid zero probability. The retained evaluators are source
snapshots, not a promise of compatibility with future versions of external tools.
