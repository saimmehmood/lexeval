"""End-to-end run: prepare data -> (optimize) -> batch inference -> metrics -> MLflow."""

from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import dspy
import mlflow
import pandas as pd

from lexeval.config import dump_config, flatten, loads_config
from lexeval.data import prepare_splits
from lexeval.errors import export_for_annotation
from lexeval.inference import batch_predict
from lexeval.metrics import classification_metrics, confusion_pairs, per_label_f1
from lexeval.optimize import compile_program
from lexeval.program import ClauseClassifier, to_examples

DEFAULT_EXPERIMENT = os.environ.get("LEXEVAL_EXPERIMENT", "lexeval")


def configure_lm(cfg: dict) -> dspy.LM:
    m = cfg["model"]
    lm = dspy.LM(m["name"], temperature=m.get("temperature", 0.0), max_tokens=m.get("max_tokens", 512))
    dspy.configure(lm=lm)
    return lm


def run(cfg: dict, experiment: str | None = None, splits: dict[str, pd.DataFrame] | None = None,
        labels: list[str] | None = None, tags: dict[str, str] | None = None) -> str:
    """Execute one configured run and return its MLflow run_id.

    `splits`/`labels` can be passed in (e.g. read from a Delta table on Databricks);
    otherwise they are built from the config.
    """
    random.seed(cfg["seed"])
    if splits is None or labels is None:
        splits, labels = prepare_splits(cfg)

    mlflow.set_experiment(experiment or DEFAULT_EXPERIMENT)
    with mlflow.start_run(run_name=cfg["name"]) as active:
        run_id = active.info.run_id
        mlflow.set_tags({"optimizer": cfg["optimizer"]["type"], "dspy_version": dspy.__version__,
                         "git_commit": _git_commit(), **(tags or {})})
        mlflow.log_params(flatten(cfg))
        mlflow.log_text(dump_config(cfg), "config.yaml")
        mlflow.log_dict({"labels": labels}, "labels.json")
        for name, df in splits.items():
            mlflow.log_param(f"split.{name}.rows", len(df))
            mlflow.log_text(df["row_id"].to_json(orient="values"), f"splits/{name}_row_ids.json")

        lm = configure_lm(cfg)
        program = ClauseClassifier(labels, module=cfg["program"]["module"])
        program = compile_program(program, cfg, to_examples(splits["train"]), to_examples(splits["val"]))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "program.json"
            program.save(str(path))
            mlflow.log_artifact(str(path), "program")
        mlflow.log_dict(_describe_prompt(program), "prompt/program_prompt.json")

        n_calls_before = len(lm.history)
        preds = batch_predict(program, splits["test"], labels, cfg.get("inference", {}).get("num_threads", 8))
        if len(lm.history) > n_calls_before:
            mlflow.log_dict({"messages": lm.history[-1].get("messages")}, "prompt/sample_rendered_prompt.json")

        metrics = classification_metrics(preds["label"], preds["pred"])
        metrics["format_failure_rate"] = float((preds["pred"] == "").mean())
        metrics["mean_latency_s"] = float(preds["latency_s"].mean())
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "n"})
        mlflow.log_metric("n_test", metrics["n"])

        _log_table(preds, "predictions/test_predictions.parquet")
        _log_table(per_label_f1(preds["label"], preds["pred"]).reset_index(), "eval/per_label_f1.csv")
        _log_table(confusion_pairs(preds["label"], preds["pred"]), "eval/top_confusions.csv")
        _log_table(export_for_annotation(preds, run_id, cfg["name"]), "annotation/misclassified.csv")

        print(json.dumps({"run_id": run_id, "name": cfg["name"], **metrics}, indent=2))
        return run_id


def rerun(run_id: str, experiment: str | None = None) -> str:
    """Re-execute a past run from the exact config it logged."""
    text = mlflow.artifacts.load_text(f"runs:/{run_id}/config.yaml")
    cfg = loads_config(text)
    return run(cfg, experiment=experiment, tags={"rerun_of": run_id})


def _describe_prompt(program: ClauseClassifier) -> dict:
    out = {}
    for name, predictor in program.named_predictors():
        out[name] = {
            "instructions": predictor.signature.instructions,
            "demos": [dict(d) for d in predictor.demos],
        }
    return out


def _log_table(df: pd.DataFrame, artifact_path: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(artifact_path).name
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        mlflow.log_artifact(str(path), str(Path(artifact_path).parent))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
