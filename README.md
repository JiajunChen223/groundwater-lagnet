# Groundwater-LAGNet

Final paper-scope lightweight code package and evidence-table generator for station-adaptive lag spatio-temporal graph forecasting of groundwater-level changes.

This repository is intentionally not a full raw-data-to-paper archive. It contains the public training/evaluation code, final-scope configs, lightweight result-table snapshots, and a script that regenerates the paper tables from a separate evidence package. Raw datasets, checkpoints, complete predictions, and large archived run directories are kept outside the code repository.

Dataset source links and expected local CSV paths are documented in `DATA.md`.

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
| Raw data, checkpoints, full predictions | No | Keep outside this repository |

Final task settings:

- Datasets: FrenchPiezo main experiment and BC PGOWN128 external validation.
- Task: daily `90 d -> 30 d` sequence forecasting.
- Target: future 30-day daily groundwater-level-change sequence, represented as change from the prediction start level.
- Lag candidates: `[3, 7, 14, 30]`.
- Graph setting: `topk=15`, `alpha/lambda=0.3`.
- Seed: `2022`.
- Paper tables: MAE and RMSE are reported in `cm`.

## Environment

Conda:

```powershell
conda env create -f environment.yml
conda activate groundwater-lagnet
```

Pip fallback:

```powershell
python -m pip install -r requirements.txt
```

## Main Commands

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

Each run writes `metrics.json` in the standardized target-delta scale and `metrics_paper.json` with MAE/RMSE converted to `cm`.

Generate final paper tables from the evidence package:

```powershell
python scripts\make_final_paper_tables.py --package ..\evidence_package_20260602 --out-dir results\final_paper_tables
```

If `--package` is omitted, the script checks `GROUNDWATER_EVIDENCE_PACKAGE`, then searches for a sibling `*20260602` directory containing `summary_csv`.

Run tests:

```powershell
python -m pytest tests -q
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

- FrenchPiezo RMSE: `34.06863021850586 cm` (`34.07 cm` in manuscript text).
- BC PGOWN128 RMSE: `51.02039868039719 cm` (`51.02 cm` in manuscript text).

See `DATA.md` for data boundaries and `EVIDENCE_PACKAGE.md` for evidence package layout.
