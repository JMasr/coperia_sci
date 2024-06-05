import json
import os.path
import pickle

import numpy as np
import torch
from pydantic import BaseModel
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)
from tqdm import tqdm

from logger import app_logger
from model.model_object import ModelBuilder


class BasicExperiment(BaseModel):
    name: str
    seed: int
    description: str
    path_to_save_experiment: str

    folds_test: dict
    folds_train: dict
    feature_name: str

    name_model: str
    parameters_model: dict

    def save_as_a_serialized_object(self):
        path_to_save = os.path.join(self.path_to_save_experiment, f"{self.name}.pkl")
        pickle.dump(self, open(path_to_save, "wb"))

    def make_prediction(self, model, y_feats: list) -> list:
        # Predict the scores
        y_score = []
        for feat_ in tqdm(y_feats, total=len(y_feats)):
            # Predict
            if self.name_model.lower() == "lstmclassifier":
                with torch.no_grad():
                    feat_ = feat_.to("cpu")
                    output_score = model.predict_proba(feat_)
                    output_score = sum(output_score)[0].item() / len(output_score)
            else:
                # Print a warning if the number of features is not the same
                if feat_.shape[1] != model.n_features_in_:
                    app_logger.warning(
                        f"Warning: The number of features is not the same. "
                        f"Expected {model.n_features_in_} but got {feat_.shape[1]}"
                    )
                    if feat_.shape[1] < model.n_features_in_:
                        feat_ = np.concatenate(
                            (
                                feat_,
                                np.zeros(
                                    (
                                        feat_.shape[0],
                                        model.n_features_in_ - feat_.shape[1],
                                    )
                                ),
                            ),
                            axis=1,
                        )
                    elif feat_.shape[1] > model.n_features_in_:
                        feat_ = feat_[:, : model.n_features_in_]

                # Predict
                output_score = model.predict(feat_)
                output_score = float(np.mean(output_score))

            # Average the scores of all segments from the input file
            y_score.append(output_score)
        return y_score

    def score_sklearn(self, model_trained, test_feats: list, test_label: list) -> dict:
        """
        Calculate a set of performance metrics using sklearn
        :param model_trained: a model trained using for inference
        :param test_feats: a list of test feats
        :param test_label: a list of test labels
        :return: a set of performance metrics: confusion matrix, f1 score, f-beta score, precision, recall, and auc score
        """
        # Start testing
        y_feats, y_true = test_feats, test_label
        y_score = self.make_prediction(model_trained, y_feats)

        # Calculate the auc_score, FP-rate, and TP-rate
        sklearn_roc_auc_score = roc_auc_score(y_true, y_score)
        sklear_fpr, sklearn_tpr, n_thresholds = roc_curve(y_true, y_score)

        # Make prediction using a threshold that maximizes the difference between TPR and FPR
        optimal_idx = np.argmax(sklearn_tpr - sklear_fpr)
        optimal_threshold = n_thresholds[optimal_idx]
        y_pred = [1 if scr > optimal_threshold else 0 for scr in y_score]

        # Calculate Precision, Recall, F1, and F-beta scores
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f_beta, support = precision_recall_fscore_support(
            y_true, y_pred
        )
        f1_scr = f1_score(y_true, y_pred)

        # Calculate Confusion Matrix
        confusion_mx = confusion_matrix(y_true, y_pred)

        # Calculate the specificity and sensitivity
        tn, fp, fn, tp = confusion_mx.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # Create a dictionary of scores
        dict_scores = {
            "model_name": self.name_model,
            "acc_score": float(acc),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "tpr": sklearn_tpr.tolist(),
            "tnr": sklear_fpr.tolist(),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "decision_thresholds": n_thresholds.tolist(),
            "optimal_threshold": float(optimal_threshold),
            "auc_score": float(sklearn_roc_auc_score),
            "confusion_matrix": confusion_mx.tolist(),
            "f1_scr": float(f1_scr),
            "f_beta": f_beta.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        }

        print("--------------------------------------------")
        print(
            "Scoring: Accuracy = {:.2f}, AUC = {:.2f}".format(
                acc, sklearn_roc_auc_score
            )
        )
        print(
            "Scoring: Sensitivity = {:.2f}, specificity = {:.2f}".format(
                sensitivity, specificity
            )
        )
        print("============================================\n")
        return dict_scores

    def training_phase(self):
        if self.parameters_model is None or self.name_model is None:
            model_builder = ModelBuilder(name="LogisticRegression")
        else:
            model_builder = ModelBuilder(
                name=self.name_model,
                parameters=self.parameters_model,
                path_to_model=self.path_to_save_experiment,
            )

        for fold in self.folds_train.keys():
            train_data = self.folds_train[fold]

            model_builder.build_model()
            trained_model = model_builder.train_model(train_data["X"], train_data["y"])

            # Save the model
            path_to_save_model = os.path.join(
                self.path_to_save_experiment, f"{self.name}_{fold}.pkl"
            )
            pickle.dump(trained_model, open(path_to_save_model, "wb"))

            # Score the model
            fold_scores_train = self.score_sklearn(
                trained_model,
                self.folds_test[fold]["X"],
                self.folds_test[fold]["y"],
            )
            pretty_score = json.dumps(fold_scores_train, indent=4)
