from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT.parent / "evidence_package_20260602"
DEFAULT_OUT = ROOT / "results" / "final_paper_tables"
EVIDENCE_ENV = "GROUNDWATER_EVIDENCE_PACKAGE"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def f(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"Missing numeric field {names}: {row}")


def text(row: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def model_key(value: object) -> str:
    return str(value).strip().lower()


def resolve_evidence_package(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_value = os.environ.get(EVIDENCE_ENV)
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(DEFAULT_PACKAGE)
    candidates.extend(
        sorted(
            [p for p in ROOT.parent.iterdir() if p.is_dir() and p.name.endswith("20260602")],
            key=lambda p: p.name,
        )
    )
    checked: list[str] = []
    for candidate in candidates:
        package = candidate.resolve()
        checked.append(str(package))
        if (package / "summary_csv").exists():
            return package
    raise FileNotFoundError(
        "Missing evidence package with summary_csv. Pass --package, set "
        f"{EVIDENCE_ENV}, or place a *20260602 package beside this repository. "
        f"Checked: {checked}"
    )


def package_run_dir(run_dir: str) -> str:
    normalized = run_dir.replace("\\", "/").strip()
    french_prefix = "outputs/top15_full_rerun/frenchpiezo/"
    bc_prefix = "outputs/bc_pgown128/seq30/"
    if normalized.startswith(french_prefix):
        return "01_FrenchPiezo_main/" + normalized[len(french_prefix) :]
    if normalized.startswith(bc_prefix):
        return "02_BC_PGOWN128_external/" + normalized[len(bc_prefix) :]
    return normalized


def metric_row(
    dataset: str,
    model: str,
    horizon: int,
    mae: float,
    rmse: float,
    r2: float,
    direction_acc: float,
    evidence: str,
    run_dir: str = "",
    **extra: object,
) -> dict[str, object]:
    return {
        "Dataset": dataset,
        "Model": model,
        "Horizon": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "Unit": "cm",
        "DirectionAcc": direction_acc,
        "R2": r2,
        "EvidenceFile": evidence,
        "run_dir": run_dir,
        **extra,
    }


def build_main_results(summary_dir: Path) -> list[dict[str, object]]:
    source = summary_dir / "two_dataset_main_results_for_paper.csv"
    run_lookup: dict[tuple[str, str], str] = {}
    for row in read_csv(summary_dir / "frenchpiezo_top15_summaries" / "frenchpiezo_main_summary.csv"):
        run_lookup[("FrenchPiezo", model_key(text(row, "model")))] = package_run_dir(text(row, "run_dir"))
    for row in read_csv(summary_dir / "external_dataset_summaries" / "bc_pgown128_seq30_main_summary.csv"):
        run_lookup[("BC_PGOWN128", model_key(text(row, "Model")))] = package_run_dir(text(row, "run_dir"))
    rows = []
    for row in read_csv(source):
        dataset = text(row, "Dataset")
        model = text(row, "Model")
        rows.append(
            metric_row(
                dataset=dataset,
                model=model,
                horizon=int(f(row, "Horizon")),
                mae=f(row, "MAE"),
                rmse=f(row, "RMSE"),
                r2=f(row, "R2"),
                direction_acc=f(row, "DirectionAcc"),
                evidence=text(row, "EvidenceFile", default=str(source.relative_to(summary_dir.parent))),
                run_dir=run_lookup.get((dataset, model_key(model)), ""),
            )
        )
    return rows


def build_recommended_metrics(summary_dir: Path) -> list[dict[str, object]]:
    rows = []
    source = summary_dir / "recommended_two_dataset_paper_metrics.csv"
    for row in read_csv(source):
        dataset = text(row, "Dataset")
        setting = text(row, "ModelOrSetting")
        model = text(row, "Model", default=model_key(setting or "lag_stgnet"))
        role = text(row, "Role")
        preferred = text(row, "PreferredUse")
        if "?" in preferred or not preferred.isascii():
            preferred = ""
        if dataset == "FrenchPiezo" and setting == "LAG-STGNet":
            role = role or "main"
            model = model or "lag_stgnet"
            preferred = preferred or "Main FrenchPiezo LAG-STGNet result; report MAE/RMSE/R2/DirectionAcc."
        elif dataset == "FrenchPiezo":
            role = role or "main baseline"
            model = model or model_key(setting)
            preferred = preferred or "Strongest FrenchPiezo baseline used for paper comparison."
        elif dataset == "BC_PGOWN128":
            role = role or "external validation"
            setting = setting or "LAG-STGNet k4"
            model = model or "lag_stgnet"
            preferred = preferred or "BC PGOWN128 external validation result; report MAE/RMSE/DirectionAcc."
        rows.append(
            {
                "Dataset": dataset,
                "Role": role,
                "ModelOrSetting": setting,
                "Model": model,
                "Horizon": int(f(row, "Horizon")),
                "MAE": f(row, "MAE"),
                "RMSE": f(row, "RMSE"),
                "Unit": "cm",
                "DirectionAcc": f(row, "DirectionAcc"),
                "R2": f(row, "R2"),
                "PreferredUse": preferred,
                "EvidenceFile": text(row, "EvidenceFile"),
            }
        )
    return rows


def build_paper_main_results(main_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    bc = {model_key(row["Model"]): row for row in main_rows if row["Dataset"] == "BC_PGOWN128"}
    rows = []
    for row in [r for r in main_rows if r["Dataset"] == "FrenchPiezo"]:
        key = model_key(row["Model"])
        if key not in bc:
            continue
        bc_row = bc[key]
        rows.append(
            {
                "Model": row["Model"],
                "Horizon": row["Horizon"],
                "FrenchPiezo_MAE": row["MAE"],
                "FrenchPiezo_RMSE": row["RMSE"],
                "FrenchPiezo_R2": row["R2"],
                "FrenchPiezo_DirectionAcc": row["DirectionAcc"],
                "BC_PGOWN128_MAE": bc_row["MAE"],
                "BC_PGOWN128_RMSE": bc_row["RMSE"],
                "BC_PGOWN128_DirectionAcc": bc_row["DirectionAcc"],
                "Unit": "cm",
                "FrenchPiezo_EvidenceFile": row["EvidenceFile"],
                "FrenchPiezo_run_dir": row["run_dir"],
                "BC_PGOWN128_EvidenceFile": bc_row["EvidenceFile"],
                "BC_PGOWN128_run_dir": bc_row["run_dir"],
            }
        )
    return rows


def build_multistep_rmse(summary_dir: Path) -> list[dict[str, object]]:
    french_rows = read_csv(summary_dir / "frenchpiezo_top15_summaries" / "frenchpiezo_main_summary.csv")
    bc_rows = read_csv(summary_dir / "external_dataset_summaries" / "bc_pgown128_seq30_main_summary.csv")
    bc = {model_key(text(row, "Model")): row for row in bc_rows}
    rows = []
    for row in french_rows:
        key = model_key(text(row, "model"))
        if key not in bc:
            continue
        bc_row = bc[key]
        rows.append(
            {
                "Model": text(row, "model"),
                "Horizon": int(f(row, "horizon")),
                "FrenchPiezo_RMSE_1_10": f(row, "RMSE_1_10_cm"),
                "FrenchPiezo_RMSE_11_20": f(row, "RMSE_11_20_cm"),
                "FrenchPiezo_RMSE_21_30": f(row, "RMSE_21_30_cm"),
                "BC_PGOWN128_RMSE_1_10": f(bc_row, "RMSE_1_10"),
                "BC_PGOWN128_RMSE_11_20": f(bc_row, "RMSE_11_20"),
                "BC_PGOWN128_RMSE_21_30": f(bc_row, "RMSE_21_30"),
                "Unit": "cm",
                "FrenchPiezo_EvidenceFile": "summary_csv/frenchpiezo_top15_summaries/frenchpiezo_main_summary.csv",
                "FrenchPiezo_run_dir": package_run_dir(text(row, "run_dir")),
                "BC_PGOWN128_EvidenceFile": text(bc_row, "EvidenceFile"),
                "BC_PGOWN128_run_dir": package_run_dir(text(bc_row, "run_dir")),
            }
        )
    return rows


def find_main_lag_row(main_rows: list[dict[str, object]], dataset: str) -> dict[str, object]:
    for row in main_rows:
        if row["Dataset"] == dataset and str(row["Model"]).lower() == "lag_stgnet":
            return row
    raise ValueError(f"Missing main LAG-STGNet row for {dataset}")


def build_french_ablation(summary_dir: Path, main_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    source = summary_dir / "frenchpiezo_top15_summaries" / "frenchpiezo_ablation_summary.csv"
    full = dict(find_main_lag_row(main_rows, "FrenchPiezo"))
    full["Ablation"] = "Full LAG-STGNet"
    full["EvidenceFile"] = "summary_csv/frenchpiezo_top15_summaries/frenchpiezo_main_summary.csv"
    full["run_dir"] = package_run_dir(str(full.get("run_dir", "")))
    rows = [full]
    for row in read_csv(source):
        rows.append(
            metric_row(
                dataset="FrenchPiezo",
                model=text(row, "model"),
                horizon=int(f(row, "horizon")),
                mae=f(row, "MAE_cm"),
                rmse=f(row, "RMSE_cm"),
                r2=f(row, "R2"),
                direction_acc=f(row, "DirectionAcc"),
                evidence="summary_csv/frenchpiezo_top15_summaries/frenchpiezo_ablation_summary.csv",
                run_dir=package_run_dir(text(row, "run_dir")),
                Ablation=text(row, "ablation"),
            )
        )
    return rows


def build_lambda_sensitivity(summary_dir: Path) -> list[dict[str, object]]:
    source = summary_dir / "frenchpiezo_top15_summaries" / "frenchpiezo_lambda_sensitivity_with_alpha_0p3.csv"
    lambda_source = summary_dir / "frenchpiezo_top15_summaries" / "frenchpiezo_lambda_sensitivity_summary.csv"
    main_source = summary_dir / "frenchpiezo_top15_summaries" / "frenchpiezo_main_summary.csv"
    run_lookup = {text(row, "lambda"): package_run_dir(text(row, "run_dir")) for row in read_csv(lambda_source)}
    main_run_dir = ""
    for row in read_csv(main_source):
        if model_key(text(row, "model")) == "lag_stgnet":
            main_run_dir = package_run_dir(text(row, "run_dir"))
            break
    rows = []
    for row in read_csv(source):
        lambda_value = text(row, "lambda")
        rows.append(
            {
                "Dataset": "FrenchPiezo",
                "Lambda": f(row, "lambda"),
                "MAE": f(row, "MAE_cm"),
                "RMSE": f(row, "RMSE_cm"),
                "Unit": "cm",
                "R2": f(row, "R2"),
                "DirectionAcc": f(row, "DirectionAcc"),
                "EvidenceFile": "summary_csv/frenchpiezo_top15_summaries/" + text(row, "source"),
                "note": text(row, "note"),
                "run_dir": run_lookup.get(lambda_value, main_run_dir if lambda_value == "0.3" else ""),
            }
        )
    return rows


def build_topk_sensitivity(summary_dir: Path, main_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    source = summary_dir / "frenchpiezo_top15_summaries" / "frenchpiezo_k_sensitivity_summary.csv"
    rows = []
    for row in read_csv(source):
        rows.append(
            {
                "Dataset": "FrenchPiezo",
                "TopK": int(f(row, "k")),
                "MAE": f(row, "MAE_cm"),
                "RMSE": f(row, "RMSE_cm"),
                "Unit": "cm",
                "R2": f(row, "R2"),
                "DirectionAcc": f(row, "DirectionAcc"),
                "EvidenceFile": "summary_csv/frenchpiezo_top15_summaries/frenchpiezo_k_sensitivity_summary.csv",
                "run_dir": package_run_dir(text(row, "run_dir")),
            }
        )
    full = find_main_lag_row(main_rows, "FrenchPiezo")
    rows.append(
        {
            "Dataset": "FrenchPiezo",
            "TopK": 15,
            "MAE": full["MAE"],
            "RMSE": full["RMSE"],
            "Unit": "cm",
            "R2": full["R2"],
            "DirectionAcc": full["DirectionAcc"],
            "EvidenceFile": "summary_csv/frenchpiezo_top15_summaries/frenchpiezo_main_summary.csv",
            "run_dir": package_run_dir(str(full.get("run_dir", ""))),
        }
    )
    return sorted(rows, key=lambda row: int(row["TopK"]))


def build_lag_sensitivity(summary_dir: Path, main_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    source = summary_dir / "frenchpiezo_top15_summaries" / "frenchpiezo_lag_candidates_summary.csv"
    rows = []
    for row in read_csv(source):
        rows.append(
            {
                "Dataset": "FrenchPiezo",
                "LagCandidates": text(row, "lag_candidates"),
                "LagKernels": text(row, "lag_kernels"),
                "MAE": f(row, "MAE_cm"),
                "RMSE": f(row, "RMSE_cm"),
                "Unit": "cm",
                "R2": f(row, "R2"),
                "DirectionAcc": f(row, "DirectionAcc"),
                "EvidenceFile": "summary_csv/frenchpiezo_top15_summaries/frenchpiezo_lag_candidates_summary.csv",
                "run_dir": package_run_dir(text(row, "run_dir")),
            }
        )
    full = find_main_lag_row(main_rows, "FrenchPiezo")
    rows.append(
        {
            "Dataset": "FrenchPiezo",
            "LagCandidates": "3/7/14/30 d",
            "LagKernels": "3/7/14/30",
            "MAE": full["MAE"],
            "RMSE": full["RMSE"],
            "Unit": "cm",
            "R2": full["R2"],
            "DirectionAcc": full["DirectionAcc"],
            "EvidenceFile": "summary_csv/frenchpiezo_top15_summaries/frenchpiezo_main_summary.csv",
            "run_dir": package_run_dir(str(full.get("run_dir", ""))),
        }
    )
    order = {"3/7 d": 0, "14/30 d": 1, "3/7/14 d": 2, "7/14/30 d": 3, "3/7/14/30 d": 4}
    return sorted(rows, key=lambda row: order.get(str(row["LagCandidates"]), 99))


def latest_metrics(package: Path, relative_root: str, prefix: str) -> Path:
    run_root = package / relative_root / "runs"
    runs = sorted(
        [p / "metrics.json" for p in run_root.glob(f"{prefix}_h30_s2022_*") if (p / "metrics.json").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No metrics found under {run_root} for prefix={prefix}")
    return runs[0]


def build_scale_checks(package: Path, main_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    checks = [
        (
            "FrenchPiezo",
            latest_metrics(package, "01_FrenchPiezo_main/main/lag_stgnet", "lag_stgnet"),
        ),
        (
            "BC_PGOWN128",
            latest_metrics(package, "02_BC_PGOWN128_external/lag_stgnet_k4", "lag_stgnet"),
        ),
    ]
    out = []
    for dataset, metrics_path in checks:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        main = find_main_lag_row(main_rows, dataset)
        rmse_std = float(metrics["RMSE"])
        mae_std = float(metrics["MAE"])
        rmse_cm = float(main["RMSE"])
        mae_cm = float(main["MAE"])
        out.append(
            {
                "Dataset": dataset,
                "MAE_standardized": mae_std,
                "RMSE_standardized": rmse_std,
                "MAE_cm": mae_cm,
                "RMSE_cm": rmse_cm,
                "MAE_cm_per_standard_unit": mae_cm / mae_std,
                "RMSE_cm_per_standard_unit": rmse_cm / rmse_std,
                "Unit": "cm",
                "metrics_json": str(metrics_path.relative_to(package)),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final paper-scope result tables from the evidence package.")
    parser.add_argument("--package", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    package = resolve_evidence_package(args.package)
    summary_dir = package / "summary_csv"
    out_dir = args.out_dir.resolve()
    if not summary_dir.exists():
        raise FileNotFoundError(f"Missing summary_csv directory: {summary_dir}")

    main_rows = build_main_results(summary_dir)
    outputs = {
        "recommended_metrics.csv": build_recommended_metrics(summary_dir),
        "table5_main_results.csv": build_paper_main_results(main_rows),
        "table6_multistep_rmse.csv": build_multistep_rmse(summary_dir),
        "table7_frenchpiezo_ablation.csv": build_french_ablation(summary_dir, main_rows),
        "table8_lambda_sensitivity.csv": build_lambda_sensitivity(summary_dir),
        "frenchpiezo_topk_sensitivity.csv": build_topk_sensitivity(summary_dir, main_rows),
        "frenchpiezo_lag_candidates_sensitivity.csv": build_lag_sensitivity(summary_dir, main_rows),
        "metric_scale_checks.csv": build_scale_checks(package, main_rows),
    }
    for obsolete in [
        "table4_main_results.csv",
        "table5_frenchpiezo_ablation.csv",
        "table6_frenchpiezo_lambda_sensitivity.csv",
        "bc_pgown128_external_ablation.csv",
    ]:
        stale = out_dir / obsolete
        if stale.exists():
            stale.unlink()
    for filename, rows in outputs.items():
        write_csv(out_dir / filename, rows)
    print(f"saved final paper tables to {out_dir}")


if __name__ == "__main__":
    main()
