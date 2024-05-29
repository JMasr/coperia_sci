import os.path
import pickle

from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.logger import app_logger

DEFAULT_CONFIG = {
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
        "max_iter": 500,
        "solver": "adam",
        "activation": "tanh",
        "learning_rate_init": 0.001,
        "hidden_layer_sizes": [20, 20],
        "verbose": True,
        "random_state": 42,
    },
}

SUPPORTED_MODELS = {
    "LogisticRegression": LogisticRegression(**DEFAULT_CONFIG["LogisticRegression"]),
    "RandomForest": RandomForestClassifier(**DEFAULT_CONFIG["RandomForest"]),
    "LinearSVM": SVC(**DEFAULT_CONFIG["LinearSVM"]),
    "MLP": MLPClassifier(**DEFAULT_CONFIG["MLP"]),
}


class ModelBuilder(BaseModel):
    name: str
    path_to_model: str

    seed: int = 42
    model: object = None
    parameters: dict = None

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
                model = SUPPORTED_MODELS[self.name]
                model.set_params(**self.parameters)
            else:
                model = SUPPORTED_MODELS[self.name]
        except Exception as e:
            message = f"ModelBuilder - An error occurred while loading the model: {e}"
            app_logger.error(message)
            raise Exception(message)

        self.model = model
        app_logger.info(f"ModelBuilder - {self.name} build successfully.")
        return model
