"""Threaded batch inference that never drops a row: failures become empty predictions."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import dspy
import pandas as pd

from lexeval.program import normalize_label


def batch_predict(program: dspy.Module, df: pd.DataFrame, labels: list[str], num_threads: int = 8) -> pd.DataFrame:
    canonical = {normalize_label(l): l for l in labels}

    def run_one(row) -> dict:
        start = time.perf_counter()
        try:
            pred = program(clause=row.text)
            raw = getattr(pred, "label", "")
            error = None
        except Exception as exc:  # parse failures, rate limits after retries, etc.
            raw, error = "", f"{type(exc).__name__}: {exc}"
        return {
            "row_id": int(row.row_id),
            "pred_raw": str(raw),
            "pred": canonical.get(normalize_label(str(raw)), ""),
            "error": error,
            "latency_s": time.perf_counter() - start,
        }

    rows = list(df.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(pool.map(run_one, rows))

    out = df[["row_id", "text", "label"]].merge(pd.DataFrame(results), on="row_id", how="left")
    out["correct"] = out["label"] == out["pred"]
    return out
