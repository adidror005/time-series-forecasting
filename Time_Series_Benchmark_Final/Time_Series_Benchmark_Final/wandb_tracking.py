"""Optional Weights & Biases logging for saved time-series benchmark outputs.

One W&B run per model from an executed comparison CSV. No raw datasets,
credentials, checkpoints or full prediction arrays are uploaded.
"""
from __future__ import annotations

import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_PROJECT = "time-series-forecasting"


def _scalar(value):
    """Keep W&B configs small and JSON-serializable; omit missing values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value) if isinstance(value, float) else int(value)
    if isinstance(value, str) and len(value) <= 300:
        return value
    return None


def _settings(meta):
    """Whitelist simple, relevant configuration keys; never log secrets/data paths."""
    keep = (
        "seed", "lookback", "horizon", "max_epochs", "patience", "batch_size",
        "catboost_iterations", "catboost_patience", "catboost_depth",
        "neural_max_epochs", "catboost_max_trees", "n_series", "num_series",
        "sampled_series", "train_windows", "seasonal_period", "val_quarters",
        "validation_quarters", "context_days", "horizon_days", "smoke_test",
        "train_origin_step_days", "n_train_windows", "learning_rate",
        "nbeats_width", "nbeats_generic_blocks",
    )
    return {k: _scalar(meta[k]) for k in keep if k in meta and _scalar(meta[k]) is not None}


def _metrics(row):
    maps = {
        "rmse": "test/rmse", "test_rmse": "test/rmse",
        "mase": "test/mase", "test_mase": "test/mase",
        "rmsle": "test/rmsle", "test_rmsle": "test/rmsle",
        "mae": "test/mae", "smape": "test/smape",
        "observed_rmse": "test/observed_rmse",
        "val_rmse": "validation/rmse", "val_mase": "validation/mase",
        "val_rmsle": "validation/rmsle", "validation_rmse": "validation/rmse",
        "train_seconds": "train/seconds", "best_epoch": "train/best_epoch",
        "epochs_run": "train/epochs_run", "parameters": "model/parameters",
        "feature_count": "model/feature_count",
        "valid_mase_series": "evaluation/valid_mase_series",
        "n_test_values": "evaluation/n_test_values",
    }
    metrics = {}
    for key, val in row.items():
        target = maps.get(str(key))
        if target is not None and (v := _scalar(val)) is not None:
            if isinstance(v, (float, int)) and not isinstance(v, bool):
                metrics[target] = v
    return metrics


def _filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def _find_history(folder, name):
    candidates = [folder / f"{_filename(name)}_history.csv",
                  folder / f"{str(name).replace(' ', '_')}_history.csv"]
    for file in candidates:
        if file.is_file():
            try:
                return pd.read_csv(file)
            except Exception as exc:
                warnings.warn(f"Skipped unreadable history {file}: {exc}")
    merged = folder / "neural_training_log.csv"
    if merged.is_file():
        try:
            data = pd.read_csv(merged)
            if "model" in data.columns:
                return data.loc[data["model"].astype(str).eq(str(name))].copy()
        except Exception as exc:
            warnings.warn(f"Skipped unreadable history {merged}: {exc}")
    return None


def _log_history(run, frame):
    if frame is None or frame.empty:
        return
    step_key = next((k for k in ("epoch", "step", "iteration") if k in frame.columns), None)
    for i, (_, row) in enumerate(frame.iterrows(), 1):
        payload = {"epoch": int(_scalar(row.get(step_key)) or i)}
        for k, val in row.items():
            if k in ("model", "family", "epoch", "step", "iteration"):
                continue
            v = _scalar(val)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                payload["history/" + _filename(k)] = v
        if len(payload) > 1:
            run.log(payload)


def resolve_dir(relative: str | Path) -> Path:
    """Find outputs whether Jupyter started from repo root or notebook folder."""
    relative = Path(relative)
    if relative.is_absolute():
        return relative
    cwd = Path.cwd()
    choices = [cwd / relative,
               cwd / "Time_Series_Benchmark_Final" / relative,
               cwd.parent / relative,
               cwd.parent / "Time_Series_Benchmark_Final" / relative]
    for choice in choices:
        if choice.exists() and any((choice / f).exists() for f in ("comparison.csv", "ranking.csv", "feature_comparison.csv")):
            return choice.resolve()
    return choices[0]


def summarize_comparison(output_dir, dataset=None, csv_name=None) -> str:
    """Generate conclusions from CSV values, not from hardcoded notebook prose."""
    folder = resolve_dir(output_dir)
    path = folder / (csv_name or "comparison.csv")
    if not path.exists():
        return f"No current comparison CSV at {path}; do not infer a winner."
    frame = pd.read_csv(path)
    model_col = next((c for c in ("model", "Model", "name") if c in frame.columns), None)
    metric = next((c for c in ("mase", "rmse", "rmsle", "test_rmse", "test_mase", "test_rmsle") if c in frame.columns), None)
    if not (model_col and metric):
        return f"No standard model/test-metric pair in {path.name}; inspect manually."
    valid = frame.copy()
    valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
    valid = valid.loc[valid[metric].notna() & ~valid[model_col].astype(str).str.contains("naive|last observed|prior-year", case=False, regex=True)]
    if valid.empty:
        return f"No trained-model scores in {path}."
    best = valid.sort_values(metric).iloc[0]
    valkey = next((c for c in ("val_mase", "val_rmse", "val_rmsle") if c in valid.columns), None)
    result = f"{dataset or folder.name}: lowest held-out {metric.upper()} is {best[model_col]} ({best[metric]:.4f})."
    if valkey:
        subset = valid.copy()
        subset[valkey] = pd.to_numeric(subset[valkey], errors="coerce")
        subset = subset.dropna(subset=[valkey])
        if not subset.empty:
            chosen = subset.sort_values(valkey).iloc[0]
            result += f" Validation selects {chosen[model_col]} (test {metric.upper()} {chosen[metric]:.4f})."
    if "family" in valid.columns:
        neural = valid.loc[valid.family.astype(str).str.lower().eq("neural")]
        trees = valid.loc[valid.family.astype(str).str.lower().eq("tree")]
        if not neural.empty and not trees.empty:
            n, t = neural.sort_values(metric).iloc[0], trees.sort_values(metric).iloc[0]
            better = "neural" if n[metric] < t[metric] else "CatBoost"
            gap = 100 * abs(float(n[metric]) - float(t[metric])) / max(float(n[metric]), float(t[metric]), 1e-10)
            result += f" Best neural: {n[model_col]} ({n[metric]:.4f}); best tree: {t[model_col]} ({t[metric]:.4f}); {better} lower by {gap:.1f}% relative to the higher error."
    return result + " These are descriptive single-run comparisons, not proof of a universal ranking."


def log_comparison_csv(output_dir, dataset=None, enabled=False, project=None,
                       csv_name="comparison.csv", source="notebook_executed") -> int:
    """Log one W&B run per model, only from the CSV actually on disk.

    Disabled by default. Can be run *after* training without retraining. Logs
    validation and test metrics as such; does not hide exploratory test selection.
    """
    if not enabled:
        print("W&B logging disabled. Set USE_WANDB = True in the notebook's final cell to upload existing CSV results.")
        return 0
    folder = resolve_dir(output_dir)
    path = folder / csv_name
    if not path.is_file():
        warnings.warn(f"No executed comparison data at {path}; no W&B runs created.")
        return 0
    try:
        import wandb
    except ImportError:
        warnings.warn("W&B unavailable. Install with `pip install wandb` and run `wandb login`.")
        return 0
    data = pd.read_csv(path)
    if "model" not in data.columns:
        warnings.warn(f"No model column in {path}; no W&B runs created.")
        return 0
    metadata = {}
    if (folder / "run_info.json").exists():
        try:
            metadata = json.loads((folder / "run_info.json").read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            warnings.warn(f"Could not read run_info.json: {exc}")
    group = str(dataset or metadata.get("dataset") or folder.name)
    project = project or os.getenv("WANDB_PROJECT", DEFAULT_PROJECT)
    count = 0
    for _, row in data.iterrows():
        model = str(row["model"])
        config = {**_settings(metadata), "dataset": group, "model": model,
                  "family": str(row.get("family", "unknown")),
                  "source_type": source, "metric_source_file": path.name}
        try:
            with wandb.init(project=project, group=group, name=f"{group} | {model}",
                            config=config, tags=["timeseries", source],
                            job_type="benchmark") as run:
                _log_history(run, _find_history(folder, model))
                result = _metrics(row.to_dict())
                if result:
                    run.log(result)
                run.summary["result_source"] = str(path)
                run.summary["test_results_are_descriptive"] = True
            count += 1
        except Exception as exc:
            warnings.warn(f"W&B failed for {group}/{model}; local CSV unchanged: {exc}")
    print(f"Logged {count}/{len(data)} saved model rows to W&B project {project!r} (group={group!r}).")
    return count
