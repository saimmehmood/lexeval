# LexEval

**DSPy legal-text inference & evaluation on Databricks.**

LexEval classifies contract provisions from [LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue)
(the LexGLUE subset of ~80k SEC-filing clauses across 100 provision types) with a DSPy program, and compares a
zero-shot baseline against DSPy-optimized programs (**BootstrapFewShot** and **MIPROv2**). Every run is a batch
inference job that logs its config, compiled prompt, predictions, accuracy/F1, and misclassifications to
**MLflow**, so any result can be traced back to, and re-executed from, the exact config that produced it.

```
 Hugging Face LEDGAR ──► pandas cleaning ──► stratified train/val/test ──► Delta tables
                                                                              │
          configs/{baseline,bootstrap,mipro}.yaml ──► DSPy program ──► (optimizer) ──► batch inference
                                                                                          │
                MLflow run: params · config.yaml · program.json · prompt · predictions · accuracy/F1
                                                                                          │
                           misclassified.csv ──► manual annotation ──► error-category summary
```

## What's in the repo

| path | purpose |
|---|---|
| `src/lexeval/data.py` | load LEDGAR, clean text with pandas (NFKC, whitespace, section numbering, truncation, duplicate and conflicting-label removal), stratified sampling |
| `src/lexeval/program.py` | DSPy signature with the output constrained to the 100 LEDGAR labels; `Predict` or `ChainOfThought` module |
| `src/lexeval/optimize.py` | `none` (baseline), `BootstrapFewShot`, `MIPROv2`, all driven by config |
| `src/lexeval/inference.py` | threaded batch inference; failures become empty predictions instead of dropped rows |
| `src/lexeval/metrics.py` | accuracy, macro/micro/weighted F1, per-label F1, top confusions |
| `src/lexeval/runner.py` | end-to-end run + MLflow logging; `rerun(run_id)` re-executes from the logged config |
| `src/lexeval/errors.py` | exports misclassifications for annotation and summarizes annotated sheets |
| `configs/` | one YAML per experiment: model, data sizes, seed, module, optimizer settings |
| `notebooks/` | Databricks notebooks: prepare data → run inference → compare & analyze errors → rerun |
| `databricks.yml` | Asset Bundle job: prepare data, run the three configs in parallel, then compare |
| `annotations/` | error taxonomy and annotated misclassification sheets |

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export OPENAI_API_KEY=...        # any LiteLLM-supported provider works; set model.name in the config

pytest                            # offline tests (use a dummy LM, no API calls)

# small smoke run, then the full comparison
lexeval run configs/baseline.yaml --set data.test_size=50
lexeval run configs/baseline.yaml configs/bootstrap.yaml configs/mipro.yaml

mlflow ui                         # browse runs in ./mlruns
lexeval rerun <run_id>            # re-execute a run from its logged config.yaml
```

## Running on Databricks

1. Add this repo as a Git folder in your workspace (or deploy it as a bundle, below).
2. Store the model API key as a secret: `databricks secrets create-scope lexeval` then
   `databricks secrets put-secret lexeval openai_api_key`.
3. Either run the notebooks in order, or deploy the job:

```bash
databricks bundle deploy -t dev
databricks bundle run lexeval_eval -t dev
```

The job runs `01_prepare_data` once (writing `main.lexeval.ledgar_{train,val,test,labels}` Delta tables),
then runs `02_run_inference` for the baseline, BootstrapFewShot, and MIPROv2 configs in parallel against those
same tables, and finishes with `03_error_analysis`. Catalog, schema, and experiment path are bundle variables.

## What each MLflow run records

- **params**: every config value flattened (`model.name`, `optimizer.type`, `data.test_size`, …) plus split sizes
- **tags**: optimizer, DSPy version, git commit, data source, `rerun_of` for re-executions
- **metrics**: `accuracy`, `macro_f1`, `micro_f1`, `weighted_f1`, `format_failure_rate`, `mean_latency_s`, `n_test`
- **artifacts**:
  - `config.yaml`: the exact config, used by `rerun`
  - `labels.json`, `splits/*_row_ids.json`: the label order and the rows in each split
  - `program/program.json`: the compiled DSPy program (instructions + demos)
  - `prompt/program_prompt.json`, `prompt/sample_rendered_prompt.json`: optimized instructions/demos and one fully rendered prompt
  - `predictions/test_predictions.parquet`: gold, prediction, raw output, error, latency for every test row
  - `eval/per_label_f1.csv`, `eval/top_confusions.csv`
  - `annotation/misclassified.csv`: the error-analysis sheet

## Error analysis

Each run exports its misclassifications with empty `error_category` and `notes` columns. Label them with the
taxonomy in [`annotations/README.md`](annotations/README.md) (near-synonym labels, multi-topic clauses, heading
mismatch, truncation, label noise, model reasoning, format failure), save the sheet under `annotations/`, and run:

```bash
lexeval summarize-annotations annotations/*.csv
```

## Design notes

- **Fair comparison.** All configs share the same data block and seed, and on Databricks they read the same Delta
  tables, so only the program and optimizer differ between runs.
- **Constrained output.** The output field is a `Literal` over the 100 LEDGAR labels, so DSPy rejects off-list
  answers. Anything unparseable counts as wrong and is tracked in `format_failure_rate`.
- **Cost control.** LEDGAR's test split has 10k clauses. Configs sample 1,000 test rows (stratified) by default.
  Use `--set data.test_size=...` to scale up or down.
- **Metric choice.** LEDGAR is heavily imbalanced, so macro-F1 is reported alongside accuracy. Micro-F1 equals
  accuracy for single-label classification and is logged for comparison with LexGLUE.

## License

MIT
