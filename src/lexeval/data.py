"""Load LEDGAR from Hugging Face and prepare clean, stratified splits with pandas."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

_WS = re.compile(r"\s+")
# LEDGAR clauses often keep section numbering / headings, e.g. "12.3 Governing Law."
_LEADING_NUMBERING = re.compile(r"^\(?[0-9ivxlcIVXLC]+(\.[0-9]+)*[.)]?\s+")


def load_ledgar(dataset: str = "coastalcph/lex_glue", subset: str = "ledgar") -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Return {split: DataFrame[text, label_id, label]} and the ordered label names."""
    from datasets import load_dataset

    ds = load_dataset(dataset, subset)
    label_names: list[str] = ds["train"].features["label"].names
    frames = {}
    for split in ds:
        df = ds[split].to_pandas().rename(columns={"label": "label_id"})
        df["label"] = df["label_id"].map(lambda i: label_names[i])
        frames[split] = df
    return frames, label_names


def clean_text(text: str, max_chars: int | None = None) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace(" ", " ")
    text = _WS.sub(" ", text).strip()
    text = _LEADING_NUMBERING.sub("", text)
    if max_chars and len(text) > max_chars:
        # cut on a word boundary so the model doesn't see half a token
        text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
    return text


def clean_frame(df: pd.DataFrame, max_chars: int | None = None) -> pd.DataFrame:
    """Normalize text, drop empty/duplicate clauses, and keep a stable row id."""
    out = df.copy()
    out["raw_len"] = out["text"].str.len()
    out["text"] = out["text"].map(lambda t: clean_text(t, max_chars=max_chars))
    out = out[out["text"].str.len() > 0]
    out = out.drop_duplicates(subset=["text", "label"])
    # A clause that appears with two different labels is ambiguous ground truth.
    conflicted = out.groupby("text")["label"].transform("nunique") > 1
    out = out[~conflicted]
    out = out.reset_index(drop=True)
    out.insert(0, "row_id", out.index.astype(int))
    return out


def stratified_sample(df: pd.DataFrame, n: int, seed: int, label_col: str = "label") -> pd.DataFrame:
    """Sample ~n rows keeping label proportions, with at least one row per label when possible."""
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    frac = n / len(df)
    parts = []
    for _, group in df.groupby(label_col, sort=True):
        k = max(1, round(len(group) * frac))
        parts.append(group.sample(n=min(k, len(group)), random_state=seed))
    sampled = pd.concat(parts)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prepare_splits(cfg: dict) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load, clean, and sample train/val/test according to a run config."""
    d = cfg["data"]
    seed = cfg["seed"]
    frames, labels = load_ledgar(d["dataset"], d["subset"])
    train = clean_frame(frames["train"], d.get("max_chars"))
    val = clean_frame(frames["validation"], d.get("max_chars"))
    test = clean_frame(frames["test"], d.get("max_chars"))
    return {
        "train": stratified_sample(train, d["train_size"], seed),
        "val": stratified_sample(val, d["val_size"], seed),
        "test": stratified_sample(test, d["test_size"], seed),
    }, labels
