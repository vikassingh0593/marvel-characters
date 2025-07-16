import argparse

from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig
from marvel_characters.serving.model_serving import ModelServing
from databricks.sdk import WorkspaceClient
from marvel_characters.utils import get_dbr_host

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--is_test",
    action="store",
    default=0,
    type=int,
    required=True,
)


root_path = args.root_path
is_test = args.is_test
config_path = f"{root_path}/files/project_config_marvel.yml"

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = f"marvel-characters-model-serving-{args.env}"

# Initialize Marvel Model Serving Manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.marvel_character_model_basic",
    endpoint_name=endpoint_name
)

# Deploy the Marvel model serving endpoint
model_serving.deploy_or_update_serving_endpoint(version=model_version)
logger.info("Started deployment/update of the Marvel serving endpoint.")

# Delete endpoint if test
if is_test==1:
    workspace = WorkspaceClient()
    workspace.serving_endpoints.delete(name=endpoint_name)
    logger.info("Deleting serving endpoint.")
