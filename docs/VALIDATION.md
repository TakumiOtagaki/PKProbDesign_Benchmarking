# Release validation

Validated on 9 September 2026 on macOS / Apple Silicon, CPython 3.13.11,
using a fresh virtual environment created from this repository's `uv.lock`.

- All 11 exported CSV checksums verified.
- 254 target identities, length/loop criteria and component assignments checked.
- 20,320 candidate occurrences and 1,016 representatives validated for each
  of the `G_big` and `G_small` comparisons.
- Probability-only representative selection reproduced for both comparisons.
- Base-pair distances recalculated from all recorded representative predictions.
- Method summaries, best-method counts and paired statistical tables reproduced
  against the recorded reference tables (floating tolerance `rtol=1e-9`).
- 29 unit tests passed, including target recoloring, zero-probability handling,
  representative selection, statistics and mocked MODENA/IPknot wrapper calls.

The mocked wrapper tests do **not** execute MODENA, IPknot or Apptainer itself.
They validate command construction, backend-call tracing and path handling.
Two test-only portability changes resolve macOS temporary-directory symlinks
and locate `true` through `PATH` instead of assuming `/bin/true`. Function-style
tests for reversed targets are also registered with the standard-library test
runner so they are not silently skipped.

The reanalysis preserves CSV floating-point round trips. Default CSV parsing can
perturb the last bits of normalized distances, change exact ties, and slightly
alter Wilcoxon statistics. No source data values or statistical methods were
changed to obtain agreement.

Not validated for this release: full design reruns, rebuilding Apptainer images,
TSUBAME job submission, fresh Linux installations of all competitors, or a
public rebuild of the HotKnots-guided SCFG2 integration. See
[HPC_REPRODUCTION.md](HPC_REPRODUCTION.md).
