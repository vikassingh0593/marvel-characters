import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.custom_model import CustomModel
from marvelous.common import create_parser

args = create_parser()

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize Marvel custom model
marvel_model = CustomModel(config=config, tags=tags, spark=spark)
logger.info("Marvel CustomModel initialized.")

# Load Marvel data
marvel_model.load_data()
logger.info("Marvel data loaded.")

# Perform feature engineering
marvel_model.feature_engineering()

# Train the Marvel model
marvel_model.train()
logger.info("Marvel model training completed.")

# Evaluate Marvel model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)

model_improved = marvel_model.model_improved(test_set=test_set)
logger.info("Marvel model evaluation completed, model improved: %s", model_improved)

is_test = args.is_test

# when running test, always register and deploy
if is_test==1:
    model_improved = True

if model_improved:
    # Register the model
    latest_version = fe_model.register_model()
    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
