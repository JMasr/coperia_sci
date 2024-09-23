import logging
from datetime import datetime

import mlflow
import requests


class MlFlowService:
    def __init__(
            self,
            app_logger: logging.Logger,
            uri: str = "http://127.0.0.1",
            port: int = 5000,
    ):
        self.port = port
        self.tracking_uri = f"{uri}:{port}"

        self.app_logger = app_logger

    def is_up(self):
        try:
            uri = f"{self.tracking_uri}/health"
            response = requests.get(uri)
            if response.status_code == 200:
                self.app_logger.info("MLFlow is running")
                return True
            else:
                self.app_logger.error(f"MLFlow is not running: {response.text}")
                return False
        except Exception as e:
            self.app_logger.error(f"Error connecting to MLFlow: {e}")
            return False

    def record_a_experiment(
            self,
            dataset_config: dict,
            all_scores: dict,
            model_config: dict,
            fold_recorded: int,
            seed: int,
    ):

        mlflow.set_tracking_uri(self.tracking_uri)

        # Check if the experiment is already running
        if mlflow.active_run():
            mlflow.end_run()

        filters = dataset_config.get("filters", {})
        filter_as_str = "_".join([f"{k}:{v}" for k, v in filters.items()])
        for char in [" ", "/", "\\", "[", "]", "{", "}", "(", ")", ",", ".", "'"]:
            filter_as_str = filter_as_str.replace(char, "")
        dataset_config.pop("filters", None)

        feature_config = dataset_config.get("feature_config", {})
        feature_name = feature_config.get("feature_type", "default")
        dataset_config.pop("feature_config", None)

        model_name = model_config["model_name"]

        mlflow_experiment_name = f"{filter_as_str}_seed:{seed}_date:{datetime.now().strftime('%Y-%m-%d')}"
        mlflow.set_experiment(experiment_name=mlflow_experiment_name)
        with mlflow.start_run(
                run_name=f"{model_name}_{feature_name}_{fold_recorded}_{seed}"
        ):
            # Log the parameters
            mlflow.log_params(model_config)
            mlflow.log_params(dataset_config)
            mlflow.log_params(filters)
            mlflow.log_params(feature_config)
            mlflow.log_param("fold", fold_recorded + 1)
            mlflow.log_param("random_state", seed)

            # Log the metrics
            mlflow_metric = {
                k: v for k, v in all_scores.items() if isinstance(v, (int, float))
            }
            mlflow.log_metrics(mlflow_metric)
