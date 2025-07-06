# Databricks notebook source
# !pip install house_price-1.1.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
from pyspark.sql import SparkSession
from loguru import logger
from marvel_characters.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Synthetic Data

# COMMAND ----------
from marvel_characters.data_processor import generate_synthetic_data

inference_data_skewed = generate_synthetic_data(train_set, drift= True, num_rows=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Tables and Update marvel_features_online

# COMMAND ----------

inference_data_skewed_spark = spark.createDataFrame(inference_data_skewed).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

inference_data_skewed_spark.write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.inference_data_skewed"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send Data to the Endpoint

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set") \
                        .withColumn("Id", col("Id").cast("string")) \
                        .toPandas()


inference_data_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed") \
                        .withColumn("Id", col("Id").cast("string")) \
                        .toPandas()


# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------


from databricks.sdk import WorkspaceClient
import requests
import time

workspace = WorkspaceClient()

# Required columns for inference
required_columns = [
    "Height",
    "Weight",
    "Universe",
    "Identity",
    "Gender",
    "Marital_Status",
    "Teams",
    "Origin",
    "Creators",
]

# COMMAND ----------

def send_request_https(dataframe_record):
    """
    Sends a request to the model serving endpoint using HTTPS.
    """
    serving_endpoint = f"https://{host}/serving-endpoints/marvel-characters-model-serving/invocations"
    
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": dataframe_record},
    )
    return response.status_code, response.text

def send_request_workspace(dataframe_record):
    """
    Sends a request to the model serving endpoint using workspace client.
    """
    serving_endpoint = f"https://{host}/serving-endpoints/marvel-characters-model-serving/invocations"
    
    response = workspace.serving_endpoints.query(
        name="marvel-characters-model-serving",
        dataframe_records=dataframe_record
    )
    return response

# COMMAND ----------

# Sample records for testing
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

# Test the endpoint
for i in range(len(dataframe_records)):
    status_code, response_text = send_request_https(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2) 