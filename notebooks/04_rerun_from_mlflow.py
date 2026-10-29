# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Re-execute a logged run
# MAGIC Every run logs its full `config.yaml`. Given a run id, this notebook reloads that config and the same
# MAGIC Delta splits and executes the run again, tagging the new run with `rerun_of`.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import mlflow

from lexeval.config import loads_config
from lexeval.runner import run

dbutils.widgets.text("run_id", "")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "lexeval")
dbutils.widgets.text("experiment", "/Shared/lexeval")
dbutils.widgets.text("secret_scope", "lexeval")
dbutils.widgets.text("secret_key", "openai_api_key")

os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(dbutils.widgets.get("secret_scope"), dbutils.widgets.get("secret_key"))
source_run = dbutils.widgets.get("run_id")
cfg = loads_config(mlflow.artifacts.load_text(f"runs:/{source_run}/config.yaml"))
cfg

# COMMAND ----------

catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
splits = {s: spark.table(f"{catalog}.{schema}.ledgar_{s}").toPandas() for s in ("train", "val", "test")}
labels = spark.table(f"{catalog}.{schema}.ledgar_labels").orderBy("label_id").toPandas()["label"].tolist()

new_run = run(cfg, experiment=dbutils.widgets.get("experiment"), splits=splits, labels=labels,
              tags={"rerun_of": source_run, "data_source": f"{catalog}.{schema}", "platform": "databricks"})
new_run
