# Data Notes

Raw datasets are not bundled in this lightweight code repository. Public source links and expected local paths are listed below. Place the prepared CSV files at the declared paths before running preprocessing or training. Paper-table reproduction still works when a complete evidence package is supplied.

## FrenchPiezo

The source cited in the manuscript is the French national water-data piezometry dataset:

```text
https://www.data.gouv.fr/datasets/piezometrie
```

Expected local path:

```text
data/raw/frenchpiezo/dataset_2015_2021_nomissing_linear.csv
```

The final config expects:

- Date column: `time`
- Station column: `bss`
- Target column: `p`
- Features: `p`, `tp`, `e`, `month_sin`, `month_cos`
- Target: 30-day daily groundwater-level changes from the prediction start level
- Input window: `90`
- Horizon: `30`
- Unit conversion: `target_unit_to_cm: 100.0`

Until the raw CSV is supplied, FrenchPiezo preprocessing and training commands fail with `Raw data not found`.

## BC PGOWN128

Expected local file:

```text
data/raw/bc_pgown128/bc_pgown128_daily.csv
```

The source cited in the manuscript is the British Columbia Provincial Groundwater Observation Well Network daily mean observation well data:

```text
https://www.env.gov.bc.ca/wsd/data_searches/obswell/map/data/ObservationWellDataAll_DailyMean.csv
```

The final config expects:

- Date column: `date`
- Station column: `site_no`
- Target column: `gwl`
- Features: `gwl`, `prcp`, `pet_proxy`, `month_sin`, `month_cos`
- Target: 30-day daily groundwater-level changes from the prediction start level
- Input window: `90`
- Horizon: `30`
- Unit conversion: `target_unit_to_cm: 1.0`

Until this file is supplied, BC PGOWN128 preprocessing and training commands fail with `Raw data not found`.

## Shared Preprocessing

Both final tasks use daily data, 128 stations, chronological train/validation/test splitting, and train-statistics-only standardization. Processed files store the standardized array plus `target_unit_to_cm` and `paper_error_scale_cm`, so training and evaluation can write both standardized and paper-scale metric files.

The final paper configs use `fill_method: interpolate`, the archived paper preprocessing mode used by the evidence package. It interpolates missing values with forward/backward fallback before standardization. For future strictly historical preprocessing, `fill_method: causal_ffill` is available; changing to that mode requires a new full experimental run and updated tables.
