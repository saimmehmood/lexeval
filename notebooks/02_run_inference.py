# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Batch inference + evaluation
# MAGIC Runs one config (baseline, BootstrapFewShot, or MIPROv2) against the Delta splits from `01_prepare_data`
# MAGIC and logs config, compiled prompt, predictions, accuracy/F1 and misclassifications to MLflow.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from lexeval.config import load_config
from lexeval.runner import run

dbutils.widgets.text("config", "../configs/baseline.yaml")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "lexeval")
dbutils.widgets.text("experiment", "/Shared/lexeval")
dbutils.widgets.text("secret_scope", "lexeval")
dbutils.widgets.text("secret_key", "openai_api_key")

os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(dbutils.widgets.get("secret_scope"), dbutils.widgets.get("secret_key"))

cfg = load_config(dbutils.widgets.get("config"))
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
cfg

# COMMAND ----------

splits = {s: spark.table(f"{catalog}.{schema}.ledgar_{s}").toPandas() for s in ("train", "val", "test")}
labels = spark.table(f"{catalog}.{schema}.ledgar_labels").orderBy("label_id").toPandas()["label"].tolist()
{s: len(df) for s, df in splits.items()}

# COMMAND ----------

run_id = run(
    cfg,
    experiment=dbutils.widgets.get("experiment"),
    splits=splits,
    labels=labels,
    tags={"data_source": f"{catalog}.{schema}", "platform": "databricks"},
)

# COMMAND ----------

dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.notebook.exit(run_id)
