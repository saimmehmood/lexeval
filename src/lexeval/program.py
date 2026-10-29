"""DSPy signature and program for LEDGAR clause classification."""

# No `from __future__ import annotations` here: the runtime-built Literal label type must stay resolvable.
from typing import Literal

import dspy
import pandas as pd


def make_signature(labels: list[str]) -> type[dspy.Signature]:
    Label = Literal[tuple(labels)]  # constrains the output to the 100 LEDGAR provision types

    class ClassifyClause(dspy.Signature):
        """Classify a provision from a commercial contract (SEC filing) into its LEDGAR provision type.
        Choose the single label that best describes the legal function of the clause."""

        clause: str = dspy.InputField(desc="contract provision text")
        label: Label = dspy.OutputField(desc="LEDGAR provision type")

    return ClassifyClause


class ClauseClassifier(dspy.Module):
    def __init__(self, labels: list[str], module: str = "predict"):
        super().__init__()
        sig = make_signature(labels)
        self.classify = dspy.ChainOfThought(sig) if module == "cot" else dspy.Predict(sig)

    def forward(self, clause: str) -> dspy.Prediction:
        return self.classify(clause=clause)


def to_examples(df: pd.DataFrame) -> list[dspy.Example]:
    return [
        dspy.Example(clause=row.text, label=row.label, row_id=int(row.row_id)).with_inputs("clause")
        for row in df.itertuples(index=False)
    ]


def exact_match(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    return normalize_label(getattr(pred, "label", "")) == normalize_label(example.label)


def normalize_label(label: str | None) -> str:
    return (label or "").strip().strip('"').strip("'").lower()
