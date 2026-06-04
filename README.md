# Groundwater-LAGNet

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-EE4C2C?logo=pytorch&logoColor=white)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scope](https://img.shields.io/badge/scope-final%20paper%20code-blue)](README.md)

**Groundwater-LAGNet** is a final paper-scope code package and evidence-table generator for **LAG-STGNet**, a station-adaptive lag spatio-temporal graph network for multi-site groundwater-level-change forecasting.

The repository is designed to be clean, lightweight, and honest about its boundaries: it contains public code, final configs, table-generation utilities, tests, and lightweight CSV snapshots. Raw datasets, checkpoints, full predictions, and the archived evidence package are intentionally kept outside the code repository.

## At A Glance

| Item | Setting |
| --- | --- |
| Task | Daily `90 d -> 30 d` multi-site sequence forecasting |
| Target | Future 30-day groundwater-level change from the prediction start level |
| Main model | `lag_stgnet` / LAG-STGNet |
| Lag candidates | `3, 7, 14, 30` days |
| Graph prior | Train-set Pearson correlation graph, `topk=15` |
| Adaptive graph mix | `alpha/lambda=0.3` in the main setting |
| Datasets | FrenchPiezo and BC PGOWN128 |
| Paper unit | `cm` for MAE/RMSE |

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/` | Model, data, graph, metric, training, and utility code |
| `configs/final/` | Final paper-scope configs for main, ablation, and sensitivity settings |
| `scripts/` | Prepare data, train, evaluate, generate tables, and release-check |
| `results/final_paper_tables/` | Lightweight paper-table CSV snapshots generated from evidence |
| `tests/` | Scope, metric, data, model-shape, and release-safety tests |
| `DATA.md` | Public data source links and expected local CSV paths |
| `EVIDENCE_PACKAGE.md` | External evidence package layout and lookup rules |

## Reproduction Scope

| Component | Included here | Source of paper numbers |
| --- | --- | --- |
| FrenchPiezo main experiment | Config and code | External evidence package |
| BC PGOWN128 external validation | Config and code | External evidence package |
| FrenchPiezo ablations | Final configs | External evidence package |
| FrenchPiezo top-k sensitivity | Final configs | External evidence package |
| FrenchPiezo lag-candidate sensitivity | Final configs | External evidence package |
| FrenchPiezo lambda sensitivity | Final configs | External evidence package |
| Lightweight final CSV snapshots | Yes | Generated from evidence package |
| Raw data, checkpoints, full predictions | No | Kept outside this repository |

Dataset source links and expected local CSV paths are documented in [DATA.md](DATA.md). Evidence package usage is documented in [EVIDENCE_PACKAGE.md](EVIDENCE_PACKAGE.md).

## Quick Start

Create the environment:

```powershell
conda env create -f environment.yml
conda activate groundwater-lagnet
```

Or install with pip:

```powershell
python -m pip install -r requirements.txt
```

Prepare data after placing raw CSV files at the paths declared in `configs/final/*.yaml`:

```powershell
python scripts\prepare_data.py --config configs\final\frenchpiezo_main.yaml
python scripts\prepare_data.py --config configs\final\bc_pgown128_main.yaml
```

Train final LAG-STGNet:

```powershell
python scripts\train.py --config configs\final\frenchpiezo_main.yaml --model lag_stgnet --horizon 30
python scripts\train.py --config configs\final\bc_pgown128_main.yaml --model lag_stgnet --horizon 30
```

Each run writes:

| File | Metric scale |
| --- | --- |
| `metrics.json` | Standardized target-delta scale |
| `metrics_paper.json` | Paper scale, with MAE/RMSE converted to `cm` |

Generate final paper tables from the external evidence package:

```powershell
python scripts\make_final_paper_tables.py --package ..\evidence_package_20260602 --out-dir results\final_paper_tables
```

If `--package` is omitted, the script checks `GROUNDWATER_EVIDENCE_PACKAGE`, then searches for a sibling `*20260602` directory containing `summary_csv`.

Run tests and release checks:

```powershell
python -m pytest tests -q
python scripts\check_release_clean.py
```

## Metric Scale

Model training uses standardized target deltas. Paper MAE/RMSE values are computed as:

```text
paper_error_scale_cm = std[target_index] * target_unit_to_cm
MAE_cm = MAE_standardized * paper_error_scale_cm
RMSE_cm = RMSE_standardized * paper_error_scale_cm
```

FrenchPiezo configs use `target_unit_to_cm: 100.0`; BC PGOWN128 uses `target_unit_to_cm: 1.0`.

## Preprocessing Modes

The final paper configs keep `fill_method: interpolate`. This is the archived paper preprocessing mode used by the evidence package; it fills missing values by interpolation with forward/backward fallback before train-statistics-only standardization.

An optional `fill_method: causal_ffill` is available for future strictly historical preprocessing. It forward-fills each station/feature using prior observations only and fills leading gaps with training-period statistics. Switching to `causal_ffill` changes the experimental protocol and requires rerunning all experiments and regenerating the paper tables; do not mix it with the current evidence package.

## Baseline Boundary

The repository includes API-compatible baseline classes so the final training interface remains inspectable and lightweight. Some baseline implementations are simplified starter/fallback versions rather than full official reproductions. The paper table values should therefore be cited from the archived evidence package, not inferred from newly training these lightweight baseline classes.

## Expected Paper Metrics

The generated final paper table should include these LAG-STGNet values:

| Dataset | RMSE |
| --- | ---: |
| FrenchPiezo | `34.06863021850586 cm` |
| BC PGOWN128 | `51.02039868039719 cm` |

## Citation

If you use this repository, please cite the accompanying paper and this code package. The machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).
