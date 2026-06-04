from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def evidence_package() -> Path:
    env_value = os.environ.get("GROUNDWATER_EVIDENCE_PACKAGE")
    if env_value and (Path(env_value) / "summary_csv").exists():
        return Path(env_value)
    direct = ROOT.parent / "evidence_package_20260602"
    if (direct / "summary_csv").exists():
        return direct
    matches = [p for p in ROOT.parent.iterdir() if p.is_dir() and p.name.endswith("20260602") and (p / "summary_csv").exists()]
    if not matches:
        pytest.skip("Missing evidence package with summary_csv")
    return sorted(matches, key=lambda p: p.name)[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_final_configs_match_paper_scope() -> None:
    for path in (ROOT / "configs" / "final").glob("*.yaml"):
        name = path.name
        with (ROOT / "configs" / "final" / name).open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert config["seed"] == 2022
        assert config["data"]["input_window"] == 90
        assert config["data"]["horizons"] == [30]
        assert "target_unit_to_cm" in config["data"]
        assert config["model"]["name"] == "lag_stgnet"
        if name == "bc_pgown128_main.yaml":
            assert config["graph"]["topk"] == 15
            assert config["graph"]["alpha"] == 0.3
            assert config["model"]["lag_kernels"] == [3, 7, 14, 30]
            assert config["data"]["target_unit_to_cm"] == 1.0
        elif name == "frenchpiezo_main.yaml":
            assert config["graph"]["topk"] == 15
            assert config["graph"]["alpha"] == 0.3
            assert config["model"]["lag_kernels"] == [3, 7, 14, 30]
            assert config["data"]["target_unit_to_cm"] == 100.0
        elif name.startswith("frenchpiezo_"):
            assert config["data"]["target_unit_to_cm"] == 100.0


def test_public_entrypoints_are_final_horizon_only() -> None:
    train_script = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    eval_script = (ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--horizon", type=int, default=30)' in train_script
    assert 'parser.add_argument("--horizon", type=int, default=30)' in eval_script
    assert "supports the final paper horizon only" in train_script
    assert "supports the final paper horizon only" in eval_script
    assert "metrics_paper.json" in train_script
    assert "metrics_eval_paper.json" in eval_script
    assert not any((ROOT / "configs" / "final").glob("bc_pgown128_" + "ablation_*.yaml"))
    assert not (ROOT / "scripts" / "run_bc_final_experiments.py").exists()
    assert not (ROOT / "scripts" / "summarize_bc_final_results.py").exists()


def test_final_table_generation_reproduces_paper_metrics() -> None:
    out_dir = ROOT / "results" / "final_paper_tables"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_final_paper_tables.py",
            "--package",
            str(evidence_package()),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    rows = read_csv(out_dir / "table5_main_results.csv")
    by_model = {row["Model"]: row for row in rows}

    lag = by_model["lag_stgnet"]
    assert lag["Unit"] == "cm"
    assert "BC_PGOWN128_R2" not in lag
    assert abs(float(lag["FrenchPiezo_RMSE"]) - 34.06863021850586) < 1e-6
    assert abs(float(lag["BC_PGOWN128_RMSE"]) - 51.02039868039719) < 1e-6

    multistep = read_csv(out_dir / "table6_multistep_rmse.csv")
    lag_steps = {row["Model"]: row for row in multistep}["lag_stgnet"]
    assert abs(float(lag_steps["FrenchPiezo_RMSE_1_10"]) - 16.561452865600586) < 1e-6
    assert abs(float(lag_steps["BC_PGOWN128_RMSE_21_30"]) - 64.1218490600586) < 1e-6
    assert not (out_dir / "bc_pgown128_external_ablation.csv").exists()

    scale_rows = read_csv(out_dir / "metric_scale_checks.csv")
    assert all(row["Unit"] == "cm" for row in scale_rows)
    for row in scale_rows:
        assert float(row["RMSE_cm"]) > float(row["RMSE_standardized"])


def test_removed_non_paper_scope_terms_from_primary_text_files() -> None:
    table_script = (ROOT / "scripts" / "make_final_paper_tables.py").read_text(encoding="utf-8")
    assert 'DEFAULT_PACKAGE = ROOT.parent / "evidence_package_20260602"' in table_script
    assert 'DEFAULT_OUT = ROOT / "results" / "final_paper_tables"' in table_script
    assert "GROUNDWATER_EVIDENCE_PACKAGE" in table_script

    forbidden = [
        "GE" + "MS" + "-GER",
        "ge" + "ms" + "_ger",
        "configs" + "/" + "exp",
        "cum" + "30",
        "target_" + "mode: " + "cumulative_" + "delta",
        "target_" + "mode: " + "delta",
        "bc_pgown128_" + "candidate",
        "bc_pgown128_" + "ablation",
        "bc_external_ablation_results_" + "for_paper.csv",
        "label_" + "horizon",
        "return_" + "trend",
        "return_" + "weight",
        "dynamic_" + "weight_gamma",
        "event_" + "weight_gamma",
        "trend_" + "loss_weight",
        "use_" + "response_gate",
        "use_" + "regime_head",
        "use_" + "station_event_calibrator",
        "station_" + "metadata_path",
        "station_" + "static",
        "static_" + "alpha",
        "create_" + "demo_if_missing",
        "demo_" + "days",
        "demo_" + "stations",
        "cumu" + "lative_" + "delta",
        "drf_" + "net",
        "gru_" + "stlgraph",
        "tcn_" + "gr" + "u",
        '"' + "gr" + "u" + '"',
        "Multi" + "BranchTemporalConvBlock",
        "66." + "280481",
    ]
    suffixes = {".py", ".yaml", ".yml", ".md", ".json", ".csv"}
    roots = [ROOT]
    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.relative_to(root)}: {token}")
    assert offenders == []


def test_release_cleanliness_script_is_present() -> None:
    script = ROOT / "scripts" / "check_release_clean.py"
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "Release cleanliness check failed" in text
    assert "FORBIDDEN_SUFFIXES" in text
