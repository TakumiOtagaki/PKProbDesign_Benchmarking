# PKProbDesign Benchmarking

Benchmark data and analysis scripts accompanying **PKProbDesign: RNA inverse
folding including pseudoknots by optimizing thermodynamic folding probability**.
The design software is maintained separately in
[PKProbDesign](https://github.com/TakumiOtagaki/PKProbDesign).

## Scope and reproducibility

This is a **data-first research archive**, not a portable installation of every
competing design tool. The supplied data allow readers to reproduce candidate
selection, structural distances, summary statistics and pairwise tests without
running RNA design software or using an HPC system.

The original design and folding calculations used the TSUBAME HPC environment,
including Linux executables, environment modules and Apptainer images. Some of
these environments are not included here and have not been rebuilt or tested
outside TSUBAME. Full sequence generation and folding reruns may require
additional installation and site-specific adaptation. In particular, the
HotKnots-guided SCFG2 integration is not part of the public PKProbDesign runtime
snapshot tested for this release. Its recorded predictions are included, but a
complete public rebuild of that stage is not provided.

## Reanalyze the published data

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/TakumiOtagaki/PKProbDesign_Benchmarking.git
cd PKProbDesign_Benchmarking
uv sync --locked
uv run --frozen python scripts/analyze_snapshot.py
uv run --frozen python -m unittest discover -s tests -v
```

The analysis writes CSV tables and `validation.json` under `output/snapshot/`.
It refuses to overwrite a nonempty output directory; use `--output output/run2`
for another run. No folding binaries, Apptainer, private repositories or network
access are needed after the Python dependencies have been installed.

The checks cover file hashes, 254 target identities and decompositions, 20
candidates per method and target, probability-only representative selection,
distances recalculated from the stored structures, and agreement with the
reference summary/statistical tables. CSV floats are read with round-trip
precision to preserve ties in rank-based tests.

## Included data

| Location | Contents |
|---|---|
| `data/targets.csv` | 254 PseudoBase++ IDs, target structures, component assignments and original target indices |
| `data/common_gbig/candidate_pool.csv` | Main comparison: 20,320 candidates and their probability terms |
| `data/common_gbig/representatives.csv` | Main comparison: 1,016 selected sequences and folding predictions |
| `data/common_gsmall/` | Corresponding pools and representatives for the reversed-scaffold comparison |
| `data/common_g*/reference/` | Recorded method summaries, best-method counts and paired tests |
| `data/manifest.json` | Source/export hashes and column-projection provenance |

The main comparison gives unique-best counts of 245 (PKProbDesign), 6
(DesiRNA), 3 (MODENA-IPknot), and 0 (antaRNA-pKiss). The reversed-scaffold
comparison gives 110, 141, 3 and 0, respectively.

Data were extracted from the completed benchmark without rerunning design or
changing the retained values. Personal paths, raw logs and embedded raw-record
metadata were removed. This repository starts with a new Git history. It does
not include natural RNA sequences, the original PseudoBase++ download, third-party
source code, executables, or container images.

## Additional documentation

- [Data fields and analysis protocol](docs/DATA_AND_PROTOCOL.md)
- [HPC environments and rerun limitations](docs/HPC_REPRODUCTION.md)
- [Recorded tool revisions and checksums](docs/tool_provenance.json)
- [Validation performed for this release](docs/VALIDATION.md)

The other scripts are selected components of the original workflow. Their
individual `--help` interfaces are retained; some expect the original staged
directory layout or explicitly supplied binary paths. Use `analyze_snapshot.py`
as the supported entry point for the compact data release.

## Attribution

Targets originate from [PseudoBase++](https://pseudobaseplusplus.utep.edu/).
Please cite the PKProbDesign manuscript and the original database/tool papers
when using this benchmark. External software and database material remain
subject to their respective terms; this repository does not relicense them.
