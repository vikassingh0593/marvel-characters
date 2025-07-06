"""Custom model implementation for Marvel character classification.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict (Alive).
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from marvel_characters.config import ProjectConfig, Tags


class MarvelCharacterModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for machine learning models to be used with MLflow.

    This class wraps a machine learning model for predicting Marvel character survival.
    """

    def __init__(self, model: object) -> None:
        """Initialize the MarvelCharacterModelWrapper.

        :param model: The underlying machine learning model.
        """
        self.model = model

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame | np.ndarray
    ) -> dict[str, int]:
        """Make predictions using the wrapped model.

        :param context: The MLflow context (unused in this implementation).
        :param model_input: Input data for making predictions.
        :return: A dictionary containing the prediction (0 for dead, 1 for alive).
        """
        logger.info(f"model_input:{model_input}")
        predictions = self.model.predict(model_input)
        logger.info(f"predictions: {predictions}")
        # Return prediction as {"Prediction": 0 or 1}
        return {"Prediction": int(predictions[0]) if len(predictions) == 1 else predictions.tolist()}


class CustomModel:
    """Custom model class for Marvel character survival prediction.

    This class encapsulates the entire workflow of loading data, preparing features,
    training the model, and making predictions.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: list[str]) -> None:
        """Initialize the CustomModel.

        :param config: Configuration object containing model settings.
        :param tags: Tags for MLflow logging.
        :param spark: SparkSession object.
        :param code_paths: List of paths to additional code dependencies.
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_custom
        self.tags = tags.dict()
        self.code_paths = code_paths

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        This method loads data from Databricks tables and splits it into features and target variables.
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.data_version = "0"  # describe history -> retrieve

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """Prepare features for model training.

        This method sets up a preprocessing pipeline including one-hot encoding for categorical
        features and LightGBM classification model.
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("classifier", LGBMClassifier(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model using the prepared pipeline."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self, dataset_type: Literal["PandasDataset", "SparkDataset"] = "SparkDataset") -> None:
        """Log the trained model and its metrics to MLflow.

        This method evaluates the model, logs parameters and metrics, and saves the model in MLflow.
        """
        mlflow.set_experiment(self.experiment_name)
        additional_pip_deps = ["pyspark==3.5.0"]
        for package in self.code_paths:
            whl_name = package.split("/")[-1]
            additional_pip_deps.append(f"./code/{whl_name}")

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            y_pred = self.pipeline.predict(self.X_test)

            # Evaluate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="weighted")
            recall = recall_score(self.y_test, y_pred, average="weighted")
            f1 = f1_score(self.y_test, y_pred, average="weighted")

            logger.info(f"📊 Accuracy: {accuracy}")
            logger.info(f"📊 Precision: {precision}")
            logger.info(f"📊 Recall: {recall}")
            logger.info(f"📊 F1 Score: {f1}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LightGBM Classifier with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=self.pipeline.predict(self.X_train))
            if dataset_type == "PandasDataset":
                dataset = mlflow.data.from_pandas(
                    self.train_set,
                    name="train_set",
                )
            elif dataset_type == "SparkDataset":
                dataset = mlflow.data.from_spark(
                    self.train_set_spark,
                    table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                    version=self.data_version,
                )
            else:
                raise ValueError("Unsupported dataset type.")

            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            mlflow.pyfunc.log_model(
                python_model=MarvelCharacterModelWrapper(self.pipeline),
                artifact_path="pyfunc-marvel-character-model",
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
                input_example=self.X_train.iloc[0:1],
            )

    def register_model(self) -> None:
        """Register the trained model in MLflow Model Registry.

        This method registers the model and sets an alias for the latest version.
        """
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pyfunc-marvel-character-model",
            name=f"{self.catalog_name}.{self.schema_name}.marvel_character_model_custom",
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.marvel_character_model_custom",
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve MLflow run dataset.

        :return: Loaded dataset source
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("✅ Dataset source loaded.")
        return dataset_source.load()

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve MLflow run metadata.

        :return: Tuple containing metrics and parameters dictionaries
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Dataset metadata loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model from MLflow (alias=latest-model) and make predictions.

        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Numpy array with predictions.
        """
        logger.info("🔄 Loading model from MLflow alias 'latest-model'...")

        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.marvel_character_model_custom@latest-model"
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions
        return predictions

    def model_improved(self, test_set: pd.DataFrame) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using accuracy and F1-score.
        :param test_set: DataFrame containing the test data.
        :return: True if the current model performs better, False otherwise.
        """
        from sklearn.metrics import accuracy_score, f1_score

        X_test = test_set[self.num_features + self.cat_features]
        y_test = test_set[self.target]

        # Predictions from the latest registered model
        latest_model_uri = f"models:/{self.catalog_name}.{self.schema_name}.marvel_character_model_custom@latest-model"
        latest_model = mlflow.pyfunc.load_model(latest_model_uri)
        y_pred_latest = latest_model.predict(X_test)

        # Predictions from the current (just trained) model
        y_pred_current = self.pipeline.predict(X_test)

        # Compute metrics
        acc_latest = accuracy_score(y_test, y_pred_latest)
        acc_current = accuracy_score(y_test, y_pred_current)
        f1_latest = f1_score(y_test, y_pred_latest, average="weighted")
        f1_current = f1_score(y_test, y_pred_current, average="weighted")

        logger.info(f"Accuracy (Current): {acc_current}, F1 (Current): {f1_current}")
        logger.info(f"Accuracy (Latest): {acc_latest}, F1 (Latest): {f1_latest}")

        # You may choose your preferred metric for comparison
        if f1_current > f1_latest:
            logger.info("Current model performs better. Returning True.")
            return True
        else:
            logger.info("Current model does not improve over latest. Returning False.")
            return False
