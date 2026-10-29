# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Prepare LEDGAR
# MAGIC Load LEDGAR (LexGLUE) from Hugging Face, clean it with pandas, draw stratified train/val/test samples,
# MAGIC and persist them as Delta tables so every inference run reads exactly the same rows.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from lexeval.config import load_config
from lexeval.data import clean_frame, load_ledgar, stratified_sample

dbutils.widgets.text("config", "../configs/baseline.yaml")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "lexeval")

cfg = load_config(dbutils.widgets.get("config"))
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

frames, labels = load_ledgar(cfg["data"]["dataset"], cfg["data"]["subset"])
{split: len(df) for split, df in frames.items()}

# COMMAND ----------

# MAGIC %md ### Clean
# MAGIC NFKC-normalize, collapse whitespace, strip leading section numbers, truncate long clauses,
# MAGIC drop empty rows, exact duplicates, and clauses that appear under conflicting labels.

# COMMAND ----------

max_chars = cfg["data"].get("max_chars")
cleaned = {
    "train": clean_frame(frames["train"], max_chars),
    "val": clean_frame(frames["validation"], max_chars),
    "test": clean_frame(frames["test"], max_chars),
}
report = {split: {"raw": len(frames[src]), "clean": len(cleaned[split])}
          for split, src in [("train", "train"), ("val", "validation"), ("test", "test")]}
report

# COMMAND ----------

display(cleaned["train"]["label"].value_counts().rename("rows").reset_index())

# COMMAND ----------

# MAGIC %md ### Stratified samples → Delta

# COMMAND ----------

d, seed = cfg["data"], cfg["seed"]
samples = {
    "train": stratified_sample(cleaned["train"], d["train_size"], seed),
    "val": stratified_sample(cleaned["val"], d["val_size"], seed),
    "test": stratified_sample(cleaned["test"], d["test_size"], seed),
}

for split, df in samples.items():
    (spark.createDataFrame(df[["row_id", "text", "label_id", "label", "raw_len"]])
        .write.mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{catalog}.{schema}.ledgar_{split}"))

import pandas as pd

spark.createDataFrame(pd.DataFrame({"label_id": range(len(labels)), "label": labels})) \
    .write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.ledgar_labels")

{split: len(df) for split, df in samples.items()}
