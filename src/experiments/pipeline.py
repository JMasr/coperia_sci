import json
import logging
import os
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from src.dataset.basic_dataset import AudioDataset, LocalDataset
from src.experiments.basic_exp import BasicExperiment
from src.features.audio_processor import SUPPORTED_FEATS
from src.logger import BasicLogger
from src.model.model_object import SUPPORTED_MODELS


class Pipeline(BaseModel):
    name: str
    root_path: Path
    config_file_path: Path

    app_logger: logging.Logger = None
    configurations_as_dict: dict = None

    dataset: LocalDataset = None

    supported_models: ClassVar = SUPPORTED_MODELS
    supported_feats: ClassVar = SUPPORTED_FEATS

    class Config:
        arbitrary_types_allowed = True

    def _logger_setup(self):
        if self.app_logger is None:
            try:
                self.app_logger = BasicLogger(
                    log_file=os.path.join(self.root_path, "logs", "experiment.log")
                ).get_logger()
                self.app_logger.warning(
                    "Pipeline - No logger was passed. Using a default one."
                )
            except Exception as e:
                raise ValueError("Pipeline - Error setting up the logger.") from e
        else:
            try:
                self.app_logger.info("Pipeline - Logger already set up.")
            except Exception as e:
                raise ValueError("Pipeline - Error on the Logger provided.") from e

    def _load_config_from_a_json(self) -> dict:
        str_with_path = self.config_file_path
        try:
            path = Path(str_with_path)
            with path.open("r", encoding="utf-8") as file:
                json_as_dict = json.loads(file.read())

            if json_as_dict == {}:
                raise ValueError(
                    "Pipeline - The specified file is empty or has an empty dictionary."
                )

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Pipeline - File '{str_with_path}' does not exist."
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Pipeline - An error occurred while decoding the JSON file: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Pipeline - An error occurred while reading the JSON file: {e}"
            ) from e

        self.app_logger.info("Pipeline - Configurations loaded.")
        return json_as_dict

    def setup_using_config(self):
        self._logger_setup()

        try:
            self.app_logger.info("Pipeline - Initialization of a COPERIA's experiment")
            self.configurations_as_dict = self._load_config_from_a_json()
        except Exception as e:
            self.app_logger.error(
                f"Pipline - Error loading the config file: {self.config_file_path}"
            )
            raise IOError(e)

        self.app_logger.info("Pipeline - Configurations loaded successfully.")

    def process_dataset(self, function_to_apply_over_metadata: callable):
        try:
            config_audio = self.configurations_as_dict.get("audio")
            feat_name = config_audio.get("feature_type")

            config_run_experiment = self.configurations_as_dict.get("run")
            debug = config_run_experiment.get("debug")

            config_dataset_experiment = self.configurations_as_dict.get("dataset")
            dataset_name = config_dataset_experiment.get("name")
            column_with_ids = config_dataset_experiment.get("column_with_ids")
            column_with_target_class = config_dataset_experiment.get(
                "column_with_target_class"
            )
            column_with_label_of_class = config_dataset_experiment.get(
                "column_with_label_of_class"
            )
            metadata_path = config_dataset_experiment.get("path_to_csv")
            dataset_object_path = config_dataset_experiment.get("path_to_object", False)
            dataset_raw_data_path = config_dataset_experiment.get("raw_data_path")
            filters = config_dataset_experiment.get("filters")
            raw_data_path = config_dataset_experiment.get("raw_data_path")

            self.app_logger.info("Pipeline - Processing the dataset")
            dataset = AudioDataset(
                name=dataset_name,
                storage_path=os.path.dirname(dataset_object_path),
                app_logger=self.app_logger,
                column_with_ids=column_with_ids,
                column_with_target_class=column_with_target_class,
                column_with_label_of_class=column_with_label_of_class,
                filters=filters,
                config_audio=config_audio,
                dataset_raw_data_path=raw_data_path,
            )
            if os.path.exists(dataset_object_path):
                self.app_logger.info(
                    "Pipeline -Loading the dataset from a serialized object."
                )
                dataset = dataset.load_dataset_from_a_serialized_object(
                    dataset_object_path
                )
            else:
                self.app_logger.info("Pipeline - Processing the dataset from scratch.")
                dataset.load_metadata_from_csv(metadata_path, decimal=".")
                if debug:
                    dataset.sample_metadata(fraction=0.1)

                dataset.transform_metadata([function_to_apply_over_metadata])
                dataset.transform_column_id_2_data_path(
                    column_name=column_with_ids,
                    path=dataset_raw_data_path,
                    extension=".wav",
                )

                dataset.load_raw_data()
                dataset.extract_acoustic_features(feat_name)
                dataset.save_dataset_as_a_serialized_object()
        except Exception as e:
            self.app_logger.error("Pipeline - Error processing the dataset.")
            raise RuntimeError(e)

        self.dataset = dataset
        self.app_logger.info("Pipeline - Dataset processed successfully.")

    @staticmethod
    def run_an_experiment(
            configurations_as_dict: dict, dataset: LocalDataset, app_logger: logging.Logger
    ) -> BasicExperiment:
        try:
            config_run_experiment = configurations_as_dict.get("run")
            seed = config_run_experiment.get("seed")
            k_fold = config_run_experiment.get("k_folds")
            test_size = config_run_experiment.get("test_size")
            run_name = config_run_experiment.get("run_name")
            path_to_save_experiment = config_run_experiment.get(
                "path_to_save_experiment"
            )

            config_dataset_experiment = configurations_as_dict.get("dataset")
            target_class = config_dataset_experiment.get("column_with_label_of_class")
            target_label = config_dataset_experiment.get("target_label")

            config_audio = configurations_as_dict.get("audio")
            feat_name = config_audio.get("feature_type")

            config_model_experiment = configurations_as_dict.get("model")
            model_name = config_model_experiment.get("name")
            model_parameters = config_model_experiment.get("parameters")
            model_parameters["random_state"] = seed

            if dataset is None:
                raise ValueError("Pipeline - No dataset was processed.")

            experiment = BasicExperiment(
                seed=seed,
                name=run_name,
                dataset=dataset,
                k_fold=k_fold,
                test_size=test_size,
                feature_name=feat_name,
                target_class=target_class,
                target_label=target_label,
                name_model=model_name,
                parameters_model=model_parameters,
                path_to_save_experiment=path_to_save_experiment,
                app_logger=app_logger,
            )

            experiment.run_experiment()
            experiment.record_experiment()
        except Exception as e:
            app_logger.error("Pipeline - Error setting up the experiment.")
            raise RuntimeError(e)

        return experiment

    def run_an_experiment_from_config(self):
        self.app_logger.info("Pipeline - Running an experiment.")
        experiment = self.run_an_experiment(
            configurations_as_dict=self.configurations_as_dict,
            dataset=self.dataset,
            app_logger=self.app_logger,
        )

        self.app_logger.info("Pipeline - Experiment executed successfully.")
        return experiment

    def run_pipeline_with_an_experiment_from_config(
            self, function_to_apply_over_metadata: callable
    ):
        self.setup_using_config()
        self.process_dataset(
            function_to_apply_over_metadata=function_to_apply_over_metadata
        )
        self.run_an_experiment_from_config()
        self.app_logger.info("Pipeline - Pipeline executed successfully.")

    def run_pipeline_with_experiments_all_models(
            self, function_to_apply_over_metadata: callable
    ):
        self.setup_using_config()
        self.process_dataset(
            function_to_apply_over_metadata=function_to_apply_over_metadata
        )

        self.app_logger.info("Pipeline - Running experiments for all models.")
        for model_name in self.supported_models.keys():
            self.configurations_as_dict["model"]["name"] = model_name
            self.run_an_experiment_from_config()

    def run_pipeline_with_experiments_all_feats(
            self, function_to_apply_over_metadata: callable
    ):

        self.setup_using_config()
        self.process_dataset(
            function_to_apply_over_metadata=function_to_apply_over_metadata
        )

        self.app_logger.info("Pipeline - Running experiments for all feature types.")
        for feat_name in self.supported_feats:
            self.configurations_as_dict["audio"]["feature_type"] = feat_name
            self.run_an_experiment_from_config()
        self.app_logger.info("Pipeline - Pipeline executed successfully.")

    def run_pipeline_for_all_model_and_feats(self, function_to_apply_over_metadata: callable):
        self.setup_using_config()
        self.process_dataset(
            function_to_apply_over_metadata=function_to_apply_over_metadata
        )

        self.app_logger.info("Pipeline - Running experiments for all models and feature types.")
        for model_name in self.supported_models.keys():
            self.configurations_as_dict["model"]["name"] = model_name
            for feat_name in self.supported_feats:
                self.configurations_as_dict["audio"]["feature_type"] = feat_name
                self.run_an_experiment_from_config()
        self.app_logger.info("Pipeline - Pipeline executed successfully.")

    def run_pipeline_for_one_model_and_all_feats(self,
                                                 model_name: str,
                                                 function_to_apply_over_metadata: callable):
        self.setup_using_config()
        self.process_dataset(
            function_to_apply_over_metadata=function_to_apply_over_metadata
        )

        self.app_logger.info(f"Pipeline - Running experiments for {model_name.upper()} and all feature types.")
        self.configurations_as_dict["model"]["name"] = model_name
        for feat_name in self.supported_feats:
            self.configurations_as_dict["audio"]["feature_type"] = feat_name
            self.run_an_experiment_from_config()
        self.app_logger.info("Pipeline - Pipeline executed successfully.")

    def run_pipeline_for_one_feat_all_models(self,
                                             feat_name: str,
                                             function_to_apply_over_metadata: callable):
        self.setup_using_config()
        self.process_dataset(
            function_to_apply_over_metadata=function_to_apply_over_metadata
        )

        self.app_logger.info(f"Pipeline - Running experiments for all models and {feat_name.upper()} feature.")
        self.configurations_as_dict["audio"]["feature_type"] = feat_name
        for model_name in self.supported_models.keys():
            self.configurations_as_dict["model"]["name"] = model_name
            self.run_an_experiment_from_config()
        self.app_logger.info("Pipeline - Pipeline executed successfully.")
