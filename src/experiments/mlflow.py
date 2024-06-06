import mlflow
import requests

from logger import app_logger


class MlFlowService:
    def __init__(self, uri: str = "http://127.0.0.1", port: int = 5000):
        self.port = port
        self.tracking_uri = f"{uri}:{port}"

    def is_up(self):
        try:
            uri = f"{self.tracking_uri}/health"
            response = requests.get(uri)
            if response.status_code == 200:
                app_logger.info("MLFlow is running")
                return True
            else:
                app_logger.error(f"MLFlow is not running: {response.text}")
                return False
        except Exception as e:
            app_logger.error(f"Error connecting to MLFlow: {e}")
            return False

    def record_a_experiment(
            self,
            filters: dict,
            all_scores: dict,
            model_config: dict,
            feature_config: dict,
            num_fold_to_record: int,
            seed: int,
    ):

        mlflow.set_tracking_uri(self.tracking_uri)

        # Check if the experiment is already running
        if mlflow.active_run():
            mlflow.end_run()

        filter_as_str = "_".join([f"{k}:{v}" for k, v in filters.items()])
        for char in [" ", "/", "\\", "[", "]", "{", "}", "(", ")", ",", ".", "'"]:
            filter_as_str = filter_as_str.replace(char, "")

        mlflow.set_experiment(f"{filter_as_str}_seed:{seed}")

        model_name = model_config["model_name"]
        feature_name = feature_config["feature_type"]

        with mlflow.start_run(
                run_name=f"{model_name}_{feature_name}_{num_fold_to_record}_{seed}"
        ):
            # Log the parameters
            mlflow.log_params(filters)
            mlflow.log_params(feature_config)
            mlflow.log_params(model_config)
            mlflow.log_param("fold", num_fold_to_record + 1)
            mlflow.log_param("random_state", seed)

            # Log the metrics
            mlflow_metric = {
                k: v for k, v in all_scores.items() if isinstance(v, (int, float))
            }
            mlflow.log_metrics(mlflow_metric)
