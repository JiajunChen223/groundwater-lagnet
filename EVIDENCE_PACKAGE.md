# Evidence Package

The repository is intentionally lightweight. Full paper-table reproduction depends on a separate evidence package containing archived run metrics, summaries, and selected prediction artifacts.

## Location

The table script resolves the evidence package in this order:

1. Explicit `--package` argument.
2. `GROUNDWATER_EVIDENCE_PACKAGE` environment variable.
3. A sibling directory whose name ends with `20260602` and contains `summary_csv`.

Example:

```powershell
python scripts\make_final_paper_tables.py --package ..\evidence_package_20260602 --out-dir results\final_paper_tables
```

## Required Layout

The package must contain:

```text
summary_csv/
01_FrenchPiezo_main/
02_BC_PGOWN128_external/
```

The table script reads summary CSV files, run metrics, and selected prediction artifacts from this package. Large artifacts such as checkpoints, `predictions.npz`, complete run directories, processed arrays, and graph matrices should remain in the evidence package or a release artifact, not in the code repository.

## Included in This Repository

Only lightweight final table snapshots are kept under:

```text
results/final_paper_tables/
```

These CSVs are useful for quick inspection. The evidence package is the source used to regenerate them and should be cited for paper-table numeric provenance.
