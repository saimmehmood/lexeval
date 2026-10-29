import dspy
import pandas as pd
from dspy.utils import DummyLM

from lexeval.inference import batch_predict
from lexeval.optimize import compile_program
from lexeval.program import ClauseClassifier, to_examples

LABELS = ["Governing Laws", "Notices", "Terminations"]
DF = pd.DataFrame({
    "row_id": [0, 1],
    "text": ["Notices shall be in writing.", "This Agreement is governed by New York law."],
    "label": ["Notices", "Governing Laws"],
})


def test_batch_predict_with_dummy_lm():
    dspy.configure(lm=DummyLM([{"label": "Notices"}] * 10))
    out = batch_predict(ClauseClassifier(LABELS), DF, LABELS, num_threads=2)
    assert out["error"].isna().all()
    assert out["pred"].tolist() == ["Notices", "Notices"]
    assert out["correct"].tolist() == [True, False]


def test_bootstrap_fewshot_collects_passing_demo():
    dspy.configure(lm=DummyLM([{"label": "Notices"}] * 10))
    cfg = {"seed": 0, "optimizer": {"type": "bootstrap_fewshot", "max_bootstrapped_demos": 1, "max_labeled_demos": 1}}
    program = compile_program(ClauseClassifier(LABELS), cfg, to_examples(DF), to_examples(DF))
    assert program.classify.demos and program.classify.demos[0]["label"] == "Notices"
