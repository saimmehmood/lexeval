"""Export misclassifications for manual annotation and summarize annotated files."""

from __future__ import annotations

import pandas as pd

# Annotation taxonomy (see annotations/README.md for definitions).
ERROR_CATEGORIES = [
    "near_synonym_label",      # gold and pred are overlapping LEDGAR types (e.g. Terminations vs Term)
    "multi_topic_clause",      # clause genuinely covers several provision types
    "heading_mismatch",        # clause heading points one way, body another
    "truncation",              # signal lost because the clause was cut at max_chars
    "label_noise",             # gold label looks wrong on inspection
    "model_reasoning",         # clear-cut clause, model simply wrong
    "format_failure",          # empty/unparseable output
]


def export_for_annotation(predictions: pd.DataFrame, run_id: str, config_name: str) -> pd.DataFrame:
    wrong = predictions[~predictions["correct"]].copy()
    wrong.insert(0, "run_id", run_id)
    wrong.insert(1, "config", config_name)
    wrong["error_category"] = ""
    wrong.loc[wrong["pred"] == "", "error_category"] = "format_failure"
    wrong["notes"] = ""
    cols = ["run_id", "config", "row_id", "label", "pred", "pred_raw", "error_category", "notes", "text"]
    return wrong[cols].rename(columns={"label": "gold"}).sort_values(["gold", "pred"]).reset_index(drop=True)


def summarize_annotations(annotated: pd.DataFrame) -> pd.DataFrame:
    unknown = set(annotated["error_category"].dropna()) - set(ERROR_CATEGORIES) - {""}
    if unknown:
        raise ValueError(f"unknown error categories: {sorted(unknown)}")
    labelled = annotated[annotated["error_category"].fillna("") != ""]
    summary = labelled.groupby(["config", "error_category"]).size().rename("count").reset_index()
    summary["share"] = summary["count"] / summary.groupby("config")["count"].transform("sum")
    return summary.sort_values(["config", "count"], ascending=[True, False]).reset_index(drop=True)
