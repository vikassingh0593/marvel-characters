"""Utility class."""

import os


def is_databricks() -> str:
    """Check if the code is running in a Databricks environment."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ
