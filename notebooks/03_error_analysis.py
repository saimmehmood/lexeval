# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Compare runs & error analysis
# MAGIC Pulls every run in the experiment, compares baseline vs. optimized programs, and surfaces the
# MAGIC misclassifications exported for annotation (`annotation/misclassified.csv`).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import mlflow
import pandas as pd

from lexeval.errors import ERROR_CATEGORIES, summarize_annotations

dbutils.widgets.text("experiment", "/Shared/lexeval")
experiment = dbutils.widgets.get("experiment")

# COMMAND ----------

runs = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time DESC"])
cols = ["run_id", "tags.mlflow.runName", "tags.optimizer", "metrics.accuracy", "metrics.macro_f1",
        "metrics.weighted_f1", "metrics.format_failure_rate", "metrics.n_test", "start_time"]
summary = runs[[c for c in cols if c in runs.columns]]
display(summary)

# COMMAND ----------

# MAGIC %md ### Latest run per config

# COMMAND ----------

latest = summary.drop_duplicates("tags.mlflow.runName").rename(columns={"tags.mlflow.runName": "config"})
display(latest)

# COMMAND ----------

# MAGIC %md ### Most frequent confusions

# COMMAND ----------

confusions = []
for row in latest.itertuples():
    df = pd.read_csv(mlflow.artifacts.download_artifacts(f"runs:/{row.run_id}/eval/top_confusions.csv"))
    df.insert(0, "config", row.config)
    confusions.append(df)
display(pd.concat(confusions))

# COMMAND ----------

# MAGIC %md ### Misclassifications to annotate
# MAGIC Download a run's `annotation/misclassified.csv`, fill `error_category` with one of the categories below
# MAGIC (definitions in `annotations/README.md`), save it under `annotations/`, and summarize it in the next cell.

# COMMAND ----------

ERROR_CATEGORIES

# COMMAND ----------

dbutils.widgets.text("annotated_csv", "")
path = dbutils.widgets.get("annotated_csv")
if path:
    annotated = pd.read_csv(path, keep_default_na=False)
    display(summarize_annotations(annotated))
