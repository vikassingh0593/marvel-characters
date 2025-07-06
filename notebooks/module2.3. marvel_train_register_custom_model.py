# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.custom_model import CustomModel
from marvel_characters import __version__ as marvel_characters_v
from dotenv import load_dotenv

# Set up Databricks or local MLflow tracking
def is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------
# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    import os
    os.environ["PROFILE"] = "marvelous"
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "module2"})
code_paths=[f"../dist/marvel_characters-{marvel_characters_v}-py3-none-any.whl"]

# COMMAND ----------
# Initialize model with the config path
custom_model = CustomModel(config=config, tags=tags, spark=spark, code_paths=code_paths)

# COMMAND ----------
custom_model.load_data()
custom_model.prepare_features()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
custom_model.train()
custom_model.log_model()

# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/marvel-characters-custom"], filter_string="tags.branch='module2'"
).run_id[0]

model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-marvel-character-model")

# COMMAND ----------
# Register model
custom_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = custom_model.load_latest_model_and_predict(X_test)
# COMMAND ---------- 