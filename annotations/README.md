# Error annotations

Each run logs `annotation/misclassified.csv` to MLflow: one row per wrong prediction with `gold`, `pred`,
the raw model output, and empty `error_category` / `notes` columns. Download it, label each row, and save it
here as `<config>_<run_id[:8]>.csv`. Rows with an empty prediction are pre-filled as `format_failure`.

| category | use when |
|---|---|
| `near_synonym_label` | gold and pred are overlapping LEDGAR types (e.g. *Terminations* vs *Term*, *Notices* vs *Notice*) |
| `multi_topic_clause` | the clause genuinely covers more than one provision type |
| `heading_mismatch` | the clause heading suggests one type but the body is another |
| `truncation` | the deciding text was cut off by `data.max_chars` |
| `label_noise` | on inspection, the gold label looks wrong |
| `model_reasoning` | the clause is clear-cut and the model is simply wrong |
| `format_failure` | the model returned nothing parseable |

Summarize one or more annotated files:

```bash
lexeval summarize-annotations annotations/*.csv
```
