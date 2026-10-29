import pandas as pd

from lexeval.data import clean_frame, clean_text, stratified_sample


def test_clean_text_normalizes_and_strips_numbering():
    assert clean_text("12.3  Governing Law.\n This Agreement") == "Governing Law. This Agreement"
    assert clean_text("(iv) Notices shall be in writing.") == "Notices shall be in writing."


def test_clean_text_truncates_on_word_boundary():
    out = clean_text("alpha beta gamma delta", max_chars=12)
    assert out == "alpha beta ..."


def test_clean_frame_drops_empty_duplicates_and_conflicts():
    df = pd.DataFrame({
        "text": ["Notices shall be sent.", "Notices shall be sent.", "   ", "Ambiguous.", "Ambiguous.", "Term is one year."],
        "label": ["Notices", "Notices", "Notices", "Terms", "Terminations", "Term"],
    })
    out = clean_frame(df)
    assert out["text"].tolist() == ["Notices shall be sent.", "Term is one year."]
    assert out["row_id"].tolist() == [0, 1]


def test_stratified_sample_keeps_every_label_and_is_deterministic():
    df = pd.DataFrame({"row_id": range(1000), "text": [f"t{i}" for i in range(1000)],
                       "label": ["a"] * 900 + ["b"] * 90 + ["c"] * 10})
    s1 = stratified_sample(df, 100, seed=7)
    s2 = stratified_sample(df, 100, seed=7)
    assert len(s1) == 100
    assert set(s1["label"]) == {"a", "b", "c"}
    assert s1["row_id"].tolist() == s2["row_id"].tolist()
