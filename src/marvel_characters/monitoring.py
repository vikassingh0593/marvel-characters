"""Model monitoring module for Marvel characters."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType

from marvel_characters.config import ProjectConfig


def create_or_refresh_monitoring(config: ProjectConfig, spark: SparkSession, workspace: WorkspaceClient) -> None:
    """Create or refresh a monitoring table for Marvel character model serving data.

    This function processes the inference data from a Delta table,
    parses the request and response JSON fields, joins with test and inference sets,
    and writes the resulting DataFrame to a Delta table for monitoring purposes.

    :param config: Configuration object containing catalog and schema names.
    :param spark: Spark session used for executing SQL queries and transformations.
    :param workspace: Workspace object used for managing quality monitors.
    """
    inf_table = spark.sql(f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`model-serving-fe_payload`")

    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("Height", DoubleType(), True),
                            StructField("Weight", DoubleType(), True),
                            StructField("Universe", StringType(), True),
                            StructField("Identity", StringType(), True),
                            StructField("Gender", StringType(), True),
                            StructField("Marital_Status", StringType(), True),
                            StructField("Teams", StringType(), True),
                            StructField("Origin", StringType(), True),
                            StructField("Creators", StringType(), True),
                            StructField("Id", StringType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    response_schema = StructType(
        [
            StructField("predictions", ArrayType(IntegerType()), True),
            StructField(
                "databricks_output",
                StructType(
                    [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
                ),
                True,
            ),
        ]
    )

    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

    inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

    df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))

    df_final = df_exploded.withColumn("timestamp_ms", (F.col("request_time").cast("long") * 1000)).select(
        F.col("request_time").alias("timestamp"),  # Use request_time as the timestamp
        F.col("timestamp_ms"),  # Select the newly created timestamp_ms column
        "databricks_request_id",
        "execution_duration_ms",
        F.col("record.Id").alias("Id"),
        F.col("record.Height").alias("Height"),
        F.col("record.Weight").alias("Weight"),
        F.col("record.Universe").alias("Universe"),
        F.col("record.Identity").alias("Identity"),
        F.col("record.Gender").alias("Gender"),
        F.col("record.Marital_Status").alias("Marital_Status"),
        F.col("record.Teams").alias("Teams"),
        F.col("record.Origin").alias("Origin"),
        F.col("record.Creators").alias("Creators"),
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("marvel-characters-model-fe").alias("model_name"),
    )

    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
    inference_set_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")

    df_final_with_status = (
        df_final.join(test_set.select("Id", "Alive"), on="Id", how="left")
        .withColumnRenamed("Alive", "alive_test")
        .join(inference_set_skewed.select("Id", "Alive"), on="Id", how="left")
        .withColumnRenamed("Alive", "alive_inference")
        .select("*", F.coalesce(F.col("alive_test"), F.col("alive_inference")).alias("alive"))
        .drop("alive_test", "alive_inference")
        .withColumn("alive", F.col("alive").cast("int"))
        .withColumn("prediction", F.col("prediction").cast("int"))
        .dropna(subset=["alive", "prediction"])
    )

    marvel_features = spark.table(f"{config.catalog_name}.{config.schema_name}.marvel_features")

    df_final_with_features = df_final_with_status.join(marvel_features, on="Id", how="left")

    df_final_with_features = df_final_with_features.withColumn("Height", F.col("Height").cast("double"))
    df_final_with_features = df_final_with_features.withColumn("Weight", F.col("Weight").cast("double"))

    df_final_with_features.write.format("delta").mode("append").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.model_monitoring"
    )

    try:
        workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
        workspace.quality_monitors.run_refresh(
            table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
        )
        logger.info("Lakehouse monitoring table exist, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table is created.")


def create_monitoring_table(config: ProjectConfig, spark: SparkSession, workspace: WorkspaceClient) -> None:
    """Create a new monitoring table for Marvel character model monitoring.

    This function sets up a monitoring table using the provided configuration,
    SparkSession, and workspace. It also enables Change Data Feed for the table.

    :param config: Configuration object containing catalog and schema names
    :param spark: SparkSession object for executing SQL commands
    :param workspace: Workspace object for creating quality monitors
    """
    logger.info("Creating new monitoring table..")

    monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=f"{config.catalog_name}.{config.schema_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="alive",
        ),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
