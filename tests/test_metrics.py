import pandas as pd
import pytest

from lexeval.metrics import classification_metrics, confusion_pairs, per_label_f1


def test_metrics_hand_computed():
    gold = pd.Series(["a", "a", "b", "b"])
    pred = pd.Series(["a", "b", "b", "b"])
    m = classification_metrics(gold, pred)
    assert m["accuracy"] == pytest.approx(0.75)
    # a: P=1, R=0.5, F1=2/3 ; b: P=2/3, R=1, F1=0.8
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m["weighted_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m["n"] == 4


def test_empty_predictions_count_as_wrong():
    m = classification_metrics(pd.Series(["a", "b"]), pd.Series(["a", None]))
    assert m["accuracy"] == pytest.approx(0.5)


def test_per_label_and_confusions():
    gold = pd.Series(["x", "x", "y"])
    pred = pd.Series(["y", "y", "y"])
    table = per_label_f1(gold, pred)
    assert table.loc["x", "f1"] == 0.0
    assert table.loc["y", "precision"] == pytest.approx(1 / 3)
    pairs = confusion_pairs(gold, pred)
    assert pairs.iloc[0].tolist() == ["x", "y", 2]
