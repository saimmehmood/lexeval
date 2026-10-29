from pathlib import Path

import pandas as pd
import pytest

from lexeval.config import flatten, load_config, with_overrides
from lexeval.errors import export_for_annotation, summarize_annotations

CONFIGS = sorted(Path(__file__).parents[1].joinpath("configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_are_valid(path):
    cfg = load_config(path)
    assert "model.name" in flatten(cfg)


def test_overrides():
    cfg = with_overrides(load_config(CONFIGS[0]), ["data.test_size=50", "model.temperature=0.3"])
    assert cfg["data"]["test_size"] == 50 and cfg["model"]["temperature"] == 0.3
    with pytest.raises(ValueError):
        with_overrides(cfg, ["optimizer.type=sgd"])


def test_annotation_roundtrip():
    preds = pd.DataFrame({"row_id": [1, 2, 3], "text": ["t1", "t2", "t3"], "label": ["a", "b", "c"],
                          "pred": ["a", "", "a"], "pred_raw": ["a", "???", "a"], "correct": [True, False, False]})
    sheet = export_for_annotation(preds, "run123", "baseline")
    assert len(sheet) == 2
    assert sheet.set_index("row_id").loc[2, "error_category"] == "format_failure"
    sheet.loc[sheet["row_id"] == 3, "error_category"] = "near_synonym_label"
    summary = summarize_annotations(sheet)
    assert summary["count"].sum() == 2
    sheet.loc[0, "error_category"] = "bogus"
    with pytest.raises(ValueError):
        summarize_annotations(sheet)
