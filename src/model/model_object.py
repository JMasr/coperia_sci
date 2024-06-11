import os.path
import pickle
import types

from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from exceptions import ModelError
from src.logger import app_logger

DEFAULT_CONFIG = types.MappingProxyType(
    {
        "LogisticRegression": {
            "C": 0.01,
            "max_iter": 40,
            "penalty": "l2",
            "random_state": 42,
            "solver": "liblinear",
            "class_weight": "balanced",
            "verbose": True,
        },
        "RandomForest": {
            "n_estimators": 20,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": True,
        },
        "LinearSVM": {
            "C": 0.1,
            "tol": 0.001,
            "max_iter": 1000,
            "class_weight": None,
            "verbose": True,
            "random_state": 42,
        },
        "MLP": {
            "alpha": 0.001,
            "max_iter": 1500,
            "solver": "adam",
            "activation": "tanh",
            "learning_rate_init": 0.001,
            "hidden_layer_sizes": [20, 20],
            "verbose": True,
            "random_state": 42,
        },
    }
)

SUPPORTED_MODELS = types.MappingProxyType(
    {
        "LogisticRegression": LogisticRegression(
            **DEFAULT_CONFIG["LogisticRegression"]
        ),
        "RandomForest": RandomForestClassifier(**DEFAULT_CONFIG["RandomForest"]),
        "LinearSVM": SVC(**DEFAULT_CONFIG["LinearSVM"]),
        "MLP": MLPClassifier(**DEFAULT_CONFIG["MLP"]),
    }
)


class ModelBuilder(BaseModel):
    name: str
    path_to_model: str

    seed: int = 42
    model: object = None
    is_trained: bool = False
    parameters: dict = None

    def _check_model_parameters(self):
        if self.name not in DEFAULT_CONFIG.keys():
            raise ValueError(f"Model {self.name} is not supported.")

        if self.parameters:
            # Check that all parameters in the default configuration are present
            for key in DEFAULT_CONFIG[self.name].keys():
                if key not in self.parameters.keys():
                    app_logger.warning(
                        f"Parameter {key} not found in the model configuration."
                        f" Using default value: {DEFAULT_CONFIG[self.name][key]}"
                    )
                    self.parameters[key] = DEFAULT_CONFIG[self.name][key]

            # Check that all parameters in the model configuration are valid
            for key in self.parameters.keys():
                if key not in DEFAULT_CONFIG[self.name].keys():
                    app_logger.warning(
                        f"Parameter {key} not found in the default configuration."
                        f" Removing parameter from the model configuration."
                    )
                    self.parameters.pop(key)

    def save_as_a_serialized_object(self, path_to_save: str = None):
        if path_to_save:
            path_to_save = os.path.join(path_to_save, self.name + ".pkl")
        else:
            path_to_save = os.path.join(self.path_to_model, self.name + ".pkl")

        try:
            with open(path_to_save, "wb") as file:
                pickle.dump(self.model, file)
        except Exception as e:
            message = f"An error occurred while saving the model: {e}"
            app_logger.error(message)
            raise IOError(message)

        app_logger.info(f"Model saved in {path_to_save}")

    def load_model_from_a_serialized_object(self, path_to_load: str = None):
        if path_to_load:
            path_to_load = os.path.join(path_to_load, self.name + ".pkl")
        else:
            path_to_load = os.path.join(self.path_to_model, self.name + ".pkl")

        try:
            with open(path_to_load, "rb") as file:
                self.model = pickle.load(file)
        except Exception as e:
            message = f"An error occurred while loading the model: {e}"
            app_logger.error(message)
            raise IOError(message)

        app_logger.info(f"Model loaded from {path_to_load}")
        return self.model

    def build_model(self):

        if self.name not in SUPPORTED_MODELS.keys():
            raise ValueError(f"Model {self.name} is not supported.")

        try:
            if self.parameters:
                self._check_model_parameters()

                model = SUPPORTED_MODELS[self.name]
                model.set_params(**self.parameters)
            else:
                app_logger.warning(
                    f"No parameters found for the model. Using default configuration."
                )
                model = SUPPORTED_MODELS[self.name]
        except Exception as e:
            message = f"ModelBuilder - An error occurred while loading the model: {e}"
            app_logger.error(message)
            raise ModelError(message)

        self.model = model
        app_logger.info(f"ModelBuilder - {self.name} build successfully.")

    def train_model(self, x, y):
        if self.model is None:
            raise ValueError(
                "ModelBuilder - Model is not built yet. Please build the model first."
            )

        try:
            self.model.fit(x, y)
            self.is_trained = True
            app_logger.info(f"ModelBuilder - Model trained successfully.")
        except Exception as e:
            app_logger.error(
                f"ModelBuilder - An error occurred while training the model: {e}"
            )
            raise ModelError(e)

        return self.model
