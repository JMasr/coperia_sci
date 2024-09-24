import logging
import os.path
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import permutation_test_score
from tqdm import tqdm

from src.dataset.basic_dataset import LocalDataset
from src.exceptions import ExperimentError
from src.experiments.mlflow import MlFlowService
from src.model.model_object import ModelBuilder
from src.model.voting import ExpertVotingSystem


class BasicExperiment:
    def __init__(
            self,
            seed: int,
            name: str,
            dataset: LocalDataset,
            feature_name: str,
            target_class: str,
            target_label: str,
            k_fold: int,
            test_size: float,
            name_model: str = None,
            parameters_model: dict = None,
            path_to_save_experiment: str = None,
            app_logger: logging.Logger = None,
    ):
        self.name = name
        self.seed = seed
        self.k_fold = k_fold
        self.test_size = test_size
        self.path_to_save_experiment = path_to_save_experiment

        self.dataset = dataset
        self.target_class = target_class
        self.target_label = target_label
        self.feature_name = feature_name
        self.parameters_dataset = None

        self.name_model = name_model
        self.parameters_model = parameters_model
        self.trained_model = None

        self.experiment_performance = None
        self.experiment_predictions = None

        self.app_logger = app_logger
        self.mlflow_service = MlFlowService(
            uri="http://127.0.0.1", port=5000, app_logger=app_logger
        )

    class Config:
        arbitrary_types_allowed = True

    def save_as_a_serialized_object(self):
        file_name = f"{self.name}_{self.name_model}_{self.feature_name}_{self.k_fold}_{self.seed}.pkl"
        path_to_save = os.path.join(self.path_to_save_experiment, file_name)
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
                    self.app_logger.warning(
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
                # Average the scores of all segments from the input file
                output_score = float(np.mean(output_score))

            y_score.append(output_score)
        return y_score

    def score_a_prediction_sklearn(self, y_true, y_pred, score_suffix: str = "") -> dict:
        """
        Calculate a set of performance metrics of a prediction set using sklearn
        :param y_true: a list of test labels
        :param y_pred: a list of predicted labels
        :param score_suffix: a suffix to add to the score name (optional)
        :return: set of performance metrics: acc, confusion matrix, f1 score, f-beta score, precision, and recall
        """
        self.app_logger.info("Experiment - Scoring a prediction...")
        score_suffix = f"{score_suffix.upper()}_" if score_suffix else ""

        try:
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f_beta, support = precision_recall_fscore_support(
                y_true, y_pred
            )
            f1 = f1_score(y_true, y_pred)

            self.app_logger.info("Experiment - Calculating the confusion matrix.")
            confusion_mx = confusion_matrix(y_true, y_pred)

            self.app_logger.info(
                "Experiment - Calculating the sensitivity and specificity."
            )
            tn, fp, fn, tp = confusion_mx.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            # Create a dictionary of scores
            dict_scores = {
                f"{score_suffix}TN": int(tn),
                f"{score_suffix}FP": int(fp),
                f"{score_suffix}FN": int(fn),
                f"{score_suffix}TP": int(tp),
                f"{score_suffix}F1": float(f1),
                f"{score_suffix}Acc": float(acc),
                f"{score_suffix}Sensitivity": float(sensitivity),
                f"{score_suffix}Specificity": float(specificity),
                f"{score_suffix}Confusion Matrix": confusion_mx.tolist()
            }

            # Scores for classes
            num_classes = len(np.unique(y_true))
            for i in range(num_classes):
                dict_scores[f"{score_suffix}Precision_Class_{i}"] = precision[i]
                dict_scores[f"{score_suffix}Recall_Class_{i}"] = recall[i]
                dict_scores[f"{score_suffix}F_Beta_Class_{i}"] = f_beta[i]

            self.app_logger.info(
                f"Experiment - {score_suffix}Scoring: Accuracy = {acc:.2f}, F1-Score = {f1:.2f}"
            )

            self.app_logger.info(
                f"Experiment - {score_suffix}Scoring: Sensitivity = {sensitivity:.2f}, Specificity = {specificity:.2f}"
            )
        except Exception as e:
            self.app_logger.error(f"Error calculating the performance metrics: {e}")
            raise ExperimentError(e)

        return dict_scores

    def score_a_model_using_sklearn(
            self, model_trained, test_feats: list, test_label: list, score_suffix: str = "Audio-Level"
    ) -> Tuple[dict, dict]:
        """
        Calculate a set of performance metrics of a trained model using sklearn
        :param model_trained: a model trained using for inference
        :param test_feats: a list of test feats
        :param test_label: a list of test labels
        :param score_suffix: a suffix to add to the score name (optional)
        :return: set of performance metrics: confusion matrix, f1 score, f-beta score, precision, recall, and auc score
        """
        self.app_logger.info("Experiment - Scoring the model...")
        score_suffix = f"{score_suffix.upper()}_" if score_suffix else ""
        try:
            y_feats, y_true = test_feats, test_label
            y_score = self.make_prediction(model_trained, y_feats)

            # Calculate the auc_score, FP-rate, and TP-rate
            auc_score = roc_auc_score(y_true, y_score)
            fpr, tpr, n_thresholds = roc_curve(y_true, y_score)

            # Make prediction using a threshold that maximizes the difference between TPR and FPR
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = n_thresholds[optimal_idx]
            y_pred = [1 if scr > optimal_threshold else 0 for scr in y_score]
        except Exception as e:
            self.app_logger.error(f"Error making prediction over the test set: {e}")
            raise ExperimentError(e)

        try:
            dict_scores = self.score_a_prediction_sklearn(y_true, y_pred, score_suffix)
        except Exception as e:
            self.app_logger.error(f"Error calculating the performance metrics: {e}")
            raise ExperimentError(e)

        try:
            self.app_logger.info(
                "Experiment - Calculating the P-Value using permutation test."
            )
            estimator = model_trained.__class__(**model_trained.get_params())

            matrix_feats = np.empty((0, y_feats[0].shape[1]))
            matrix_labels = np.empty((0, 1))
            for feat, label in zip(y_feats, y_true):
                matrix_feats = np.vstack((matrix_feats, feat))
                label = np.array([label] * feat.shape[0])
                matrix_labels = np.vstack((matrix_labels, label))
            matrix_labels = matrix_labels.ravel()

            score, permutation_scores, pvalue = permutation_test_score(
                estimator,
                matrix_feats,
                matrix_labels,
                random_state=self.seed,
                n_jobs=-1,
            )
            dict_scores[f"{score_suffix}P-Value"] = pvalue
            dict_scores[f"{score_suffix}Permutation-Score"] = score

        except Exception as e:
            self.app_logger.error(f"Error calculating the Permutation Metrics: {e}")
            raise ExperimentError(e)

        dict_scores[f"{score_suffix}AUC"] = auc_score
        dict_scores[f"{score_suffix}Threshold"] = optimal_threshold

        dict_predictions = {
            "y_true": y_true,
            "y_scores": y_score,
            "y_pred": y_pred,
        }

        self.app_logger.info(
            f"Experiment - Scoring: Accuracy = {dict_scores.get('Acc', -1)},"
            f" AUC = {dict_scores.get('AUC', -1)},"
            f" F1-Score = {dict_scores.get('F1', -1)}"
            f" P-Value = {dict_scores.get('P-Value', -1)}"
        )
        self.app_logger.info(
            f"Experiment - Scoring: Sensitivity = {dict_scores.get('Sensitivity', -1)},"
            f"Specificity = {dict_scores.get('Specificity', -1)},"
        )

        return dict_scores, dict_predictions

    def run_experiment(self):
        if self.parameters_model is None or self.name_model is None:
            model_builder = ModelBuilder(
                name="LogisticRegression", path_to_model=self.path_to_save_experiment
            )
        else:
            model_builder = ModelBuilder(
                name=self.name_model,
                parameters=self.parameters_model.copy(),
                path_to_model=self.path_to_save_experiment,
                app_logger=self.app_logger,
            )

        folds_train, folds_test = self.dataset.get_k_audio_subsets_multiprocess(
            acoustics_feat_name=self.feature_name,
            test_size=self.test_size,
            k_folds=self.k_fold,
            seed=self.seed,
        )

        model_performance_for_each_fold, predictions_for_each_fold = {}, {}
        for fold in folds_train.keys():
            # Get the training set
            train_data = folds_train[fold]

            # Build and train the model
            model_builder.build_model()
            trained_model = model_builder.train_model(train_data["X"], train_data["y"])
            self.trained_model = trained_model

            # Get the test set
            test_data = folds_test[fold]
            test_label = test_data["y"]
            test_samples = test_data["X"]

            # Score the model using the audio-level predictions on the test set
            fold_audio_scores_test_set, fold_predictions_test_set = self.score_a_model_using_sklearn(
                trained_model,
                test_samples,
                test_label,
            )

            # Get the audio-level predictions
            fold_predictions_test_set["audio_id"] = [
                Path(audio_id).stem
                for audio_id in test_data[self.dataset.column_with_ids]
            ]
            fold_predictions_test_set["spk_id"] = [
                spk_id
                for spk_id in test_data["spk_id"]
            ]
            df_fold_predictions_test_set = pd.DataFrame(fold_predictions_test_set)

            # Get a speaker-level prediction using the audio-level predictions
            voting_model = ExpertVotingSystem(voting="soft")
            df_prediction_labels_for_speaker_test_set = voting_model.predict(
                df=df_fold_predictions_test_set,
                id_col="spk_id",
                class_col="y_pred",
                prob_col="y_scores"
            )

            # Get the true labels for the speaker-level predictions
            df_true_labels_for_speaker_test_set = df_fold_predictions_test_set.groupby("spk_id")[
                "y_true"].first().reset_index()
            # Convert the true labels to integers
            df_true_labels_for_speaker_test_set["y_true"] = df_true_labels_for_speaker_test_set["y_true"].astype(int)

            # Score the speaker-level predictions
            fold_speaker_scores_test_set = self.score_a_prediction_sklearn(
                df_true_labels_for_speaker_test_set["y_true"],
                df_prediction_labels_for_speaker_test_set["y_pred"],
                score_suffix="Speaker-Level"
            )

            # Union the audio-level and speaker-level scores
            fold_speaker_scores_test_set.update(fold_audio_scores_test_set)

            # Save the performance metrics and predictions for each fold
            model_performance_for_each_fold[fold] = fold_speaker_scores_test_set
            predictions_for_each_fold[fold] = df_prediction_labels_for_speaker_test_set

        self.experiment_performance = model_performance_for_each_fold
        self.experiment_predictions = predictions_for_each_fold
        self.app_logger.info(f"Experiment - Training phase completed.")

        self.save_as_a_serialized_object()
        return model_performance_for_each_fold

    def record_experiment(self):
        if not self.mlflow_service.is_up():
            self.app_logger.error(
                "MLFlow is not running. The experiment will not be recorded."
            )
            raise ConnectionError("MLFlow is not running.")

        for fold in self.experiment_performance.keys():
            model_config = self.parameters_model.copy()
            model_config["model_name"] = self.name_model
            model_performance = self.experiment_performance[fold]

            dataset_info = self.dataset.get_dataset_info()

            try:
                self.mlflow_service.record_a_experiment(
                    dataset_config=dataset_info,
                    all_scores=model_performance,
                    model_config=model_config,
                    fold_recorded=fold,
                    seed=self.seed,
                )
            except Exception as e:
                self.app_logger.error(f"Error recording the experiment: {e}")
                raise ExperimentError(e)
