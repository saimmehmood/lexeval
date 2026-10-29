"""Accuracy and F1 computed directly from gold/pred columns (no sklearn dependency)."""

from __future__ import annotations

import pandas as pd


def classification_metrics(gold: pd.Series, pred: pd.Series) -> dict[str, float]:
    gold = gold.astype(str).reset_index(drop=True)
    pred = pred.fillna("").astype(str).reset_index(drop=True)
    n = len(gold)
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "weighted_f1": 0.0, "n": 0}

    correct = gold == pred
    labels = sorted(set(gold))  # macro/weighted F1 are averaged over gold labels
    per_label = per_label_f1(gold, pred, labels)
    support = gold.value_counts()

    macro = per_label["f1"].mean()
    weighted = (per_label["f1"] * support.reindex(per_label.index)).sum() / n
    accuracy = correct.mean()
    # In single-label classification where every row gets one prediction, micro-F1 == accuracy.
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro),
        "micro_f1": float(accuracy),
        "weighted_f1": float(weighted),
        "n": n,
    }


def per_label_f1(gold: pd.Series, pred: pd.Series, labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels if labels is not None else sorted(set(gold))
    rows = []
    for label in labels:
        tp = int(((gold == label) & (pred == label)).sum())
        fp = int(((gold != label) & (pred == label)).sum())
        fn = int(((gold == label) & (pred != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": label, "precision": precision, "recall": recall, "f1": f1, "support": tp + fn})
    return pd.DataFrame(rows).set_index("label")


def confusion_pairs(gold: pd.Series, pred: pd.Series, top: int = 25) -> pd.DataFrame:
    """Most frequent (gold -> pred) mistakes."""
    df = pd.DataFrame({"gold": gold.values, "pred": pred.values})
    wrong = df[df["gold"] != df["pred"]]
    return (
        wrong.value_counts(["gold", "pred"])
        .rename("count")
        .reset_index()
        .head(top)
    )
