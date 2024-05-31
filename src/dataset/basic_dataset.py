import multiprocessing
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import KFold, train_test_split

import src.features.audio_processor
from src.exceptions import MetadataError, AudioProcessingError
from src.features.audio_processor import AudioProcessor
from src.logger import app_logger


class LocalDataset(BaseModel):
    """
    The LocalDataset class is a Python class that represents a local dataset.
    It provides methods for loading metadata from a CSV file, transforming the metadata, and creating train-test subsets
    for machine learning tasks.

    Main functionalities:
        * Loading metadata from a CSV file
        * Transforming the metadata using a list of transformations
        * Creating train-test subsets based on a target column
        * Creating k-fold subsets based on a target column

    Example Usage:
        # Create an instance of the LocalDataset class
        dataset = LocalDataset(name="TEST-DATASET")

        # Load metadata from a CSV file
        dataset.load_metadata_from_csv("metadata.csv", decimal=",")

        # Transform the metadata using a list of transformations
        dataset.transform_metadata(transformations=[transformation1, transformation2])

        # Create train-test subsets using a target column
        subsets = dataset.make_1_fold_subsets(target_class_for_fold="patient_id",
                                              target_label_for_fold="label"
                                              test_size=0.2, seed=42)

        # Create k-fold subsets using a target column
        subsets = dataset.make_k_fold_subsets(target_class_for_fold="patient_id", k_fold=5, seed=42)

    """

    name: str
    storage_path: str

    raw_metadata: pd.DataFrame = pd.DataFrame()
    post_processed_metadata: pd.DataFrame = pd.DataFrame()

    folds_data: dict = {}

    class Config:
        arbitrary_types_allowed = True

    def save_dataset_as_a_serialized_object(self, path_to_save_the_dataset: str = None):
        if path_to_save_the_dataset is None:
            path_to_save_the_dataset = os.path.join(
                self.storage_path, f"{self.name}.pkl"
            )

        try:
            with open(path_to_save_the_dataset, "wb") as file:
                pickle.dump(self, file)

            app_logger.info(
                f"LocalDataset - The object was saved to {path_to_save_the_dataset}"
            )
        except Exception as e:
            app_logger.error(f"LocalDataset - Saving the dataset fails. Error: {e}")
            raise MetadataError(e)

    def load_dataset_from_a_serialized_object(self, path_to_object: str = None):
        # Deserialize the object from a file
        try:
            if path_to_object is None:
                path_to_object = os.path.join(self.storage_path, f"{self.name}.pkl")
            with open(path_to_object, "rb") as file:
                dataset = pickle.load(file)
        except Exception as e:
            app_logger.error(f"LocalDataset - Loading the dataset fails. Error: {e}")
            raise MetadataError(e)

        return dataset

    def load_metadata_from_csv(self, path_to_metadata: str, **kwargs):
        try:
            path_to_metadata = Path(path_to_metadata)
            with open(path_to_metadata, "r") as file:
                self.raw_metadata = pd.read_csv(file, **kwargs)
                self.raw_metadata.drop_duplicates(inplace=True)
            app_logger.info("LocalDataset - The CSV file was successful read")
        except ValueError as e:
            app_logger.error(f"LocalDataset - Pandas failed reading the CSV: {e}")
            raise MetadataError(f"{e}")
        except FileNotFoundError as e:
            app_logger.error(f"LocalDataset - The file wasn´t found: {e}")
            raise MetadataError(f"{e}")

    def transform_metadata(self, transformations: list):
        if self.raw_metadata.empty:
            message = "LocalDataset - Metadata is empty. Please, load a metadata first."
            app_logger.error(message)
            raise ValueError(message)
        if not transformations:
            message = "LocalDataset - No transformations were provided."
            app_logger.error(message)
            raise ValueError(message)
        else:
            try:
                app_logger.info(
                    f"LocalDataset - Starting {len(transformations)} transformations over metadata"
                )
                df = self.raw_metadata.copy(deep=False)
                for index, transformation in enumerate(transformations):
                    df = transformation(df)
                    app_logger.debug(
                        f"LocalDataset - Transformation #{index + 1} finished successfully"
                    )

                self.post_processed_metadata = df
            except RuntimeError as e:
                raise MetadataError(
                    f"An runtime error occurred during the transformation process: {e}"
                )
            except Exception as e:
                raise MetadataError(
                    f"An error occurred during transforming metadata: {e}"
                )

    def transform_column_id_2_data_path(
            self, column_name: str, path: str, extension: str
    ):
        # Check if string path is a valid path
        if os.path.exists(path) and os.path.isdir(path):
            path = os.path.abspath(path)
        # Check if extension has a dot
        if not extension.startswith("."):
            extension = "." + extension

        # Transform the column values to data paths
        self.post_processed_metadata[column_name] = self.post_processed_metadata[
            column_name
        ].apply(lambda x: os.path.join(path, x + extension))

    @staticmethod
    def get_index_for_1_fold(
            exp_metadata: pd.DataFrame,
            target_class_for_fold: str,
            target_label_for_fold: str,
            test_size: float = 0.2,
            seed: int = 42,
    ) -> tuple:
        target_class_data = exp_metadata[
            [target_class_for_fold, target_label_for_fold]
        ].drop_duplicates()
        target_ids = target_class_data[target_class_for_fold]
        labels_of_class = target_class_data[target_label_for_fold]

        # Split the data
        pat_train, pat_test, _, _ = train_test_split(
            target_ids,
            labels_of_class,
            test_size=test_size,
            stratify=labels_of_class,
            random_state=seed,
            shuffle=True,
        )
        app_logger.debug(
            f"LocalDataset - Subset creation-"
            f" Results: {pat_train.shape[0]} training objects & {pat_test.shape[0]} test objects"
        )

        return pat_train, pat_test

    @staticmethod
    def get_a_series_by_index_and_a_target_class(
            exp_metadata: pd.DataFrame, index_to_get: np.ndarray, target_class: str
    ) -> pd.Series:
        """
        Get a subset of the metadata based on the index of the samples and the target class.
        Example:
        # Using the target subsets to select the samples
            samples_train = self.get_a_series_by_index_and_a_target_class(exp_metadata,
                                                                          index_train_fold,
                                                                          target_data_for_fold)
            labels_train = self.get_a_series_by_index_and_a_target_class(exp_metadata,
                                                                         index_train_fold,
                                                                         target_label_for_fold)
            samples_test = self.get_a_series_by_index_and_a_target_class(exp_metadata,
                                                                         index_test_fold,
                                                                         target_data_for_fold)
            labels_test = self.get_a_series_by_index_and_a_target_class(exp_metadata,
                                                                        index_test_fold,
                                                                        target_label_for_fold)
        :param exp_metadata: DataFrame with the metadata
        :param index_to_get: List with the index of the samples to get
        :param target_class: Column name with the target class
        :return:
        """

        # Using the target subsets to select the samples
        sample_data = exp_metadata.loc[exp_metadata[target_class].isin(index_to_get)]
        samples_filtered = sample_data[target_class]

        # Log the lengths of the subsets
        app_logger.debug(
            f"LocalDataset - Filtered the Dataset."
            f" Results: {index_to_get.shape[0]} instance & {samples_filtered.shape[0]} samples"
        )

        return samples_filtered

    def make_1_fold_subsets(
            self,
            target_class_for_fold: str,
            target_label_for_fold: str,
            test_size: float = 0.2,
            seed: int = 42,
    ) -> dict:
        """
        Split the data into train and test subsets. The split is done using a target column.
        For example, patients ids.
        :param target_class_for_fold: Column with the target class to split the data. For example, patient_id
        :param target_label_for_fold: Column with the labels of the data.
        :param test_size: Size of the test set
        :param seed: Random seed for the split
        :return: A dictionary with the train and test subsets
        """
        app_logger.info(
            f"LocalDataset - Subsets creation- Starting the creation of train and test subsets"
        )

        # Prepare the metadata
        if self.post_processed_metadata.empty:
            app_logger.warning(
                "LocalDataset - Not transformation detected over the metadata."
                "Making the subsets using the original metadata."
            )
            self.post_processed_metadata = self.raw_metadata.copy()

        try:
            exp_metadata = self.post_processed_metadata.copy(deep=False)
            index_train_fold, index_test_fold = self.get_index_for_1_fold(
                exp_metadata,
                target_class_for_fold,
                target_label_for_fold,
                test_size,
                seed,
            )
            # Create a new column to identify the subsets
            exp_metadata["subset"] = "train"
            exp_metadata.loc[
                exp_metadata[target_class_for_fold].isin(index_test_fold), "subset"
            ] = "test"

            self.folds_data[0] = exp_metadata

            # Log the lengths of the subsets
            app_logger.info(
                "LocalDataset - Subsets creation- Train and test subsets creation successful."
                f" Train subset: {exp_metadata[exp_metadata['subset'] == 'train'].shape[0]} &"
                f" Test subset: {exp_metadata[exp_metadata['subset'] == 'test'].shape[0]}"
            )
        except Exception as e:
            app_logger.error(
                f"LocalDataset - Subsets creation - The subsets creation fails. Error: {e}"
            )
            raise MetadataError(e)

        return self.folds_data

    def make_k_fold_subsets(
            self, target_class_for_fold: str, k_fold: int, seed: int
    ) -> dict:
        app_logger.info(
            f"LocalDataset - Subsets creation- Starting the creation of {k_fold}-folds"
        )
        # Prepare the metadata
        if self.post_processed_metadata.empty:
            app_logger.warning(
                "LocalDataset - Not transformation detected over the metadata."
                "Making the subsets using the original metadata."
            )
            self.post_processed_metadata = self.raw_metadata.copy()

        try:
            # Create the KFold object
            sklearn_k_fold_operator = KFold(
                n_splits=k_fold, shuffle=True, random_state=seed
            )
            k_folds_index_generator = sklearn_k_fold_operator.split(
                self.post_processed_metadata[target_class_for_fold]
            )

            for fold_index, (train_index, test_index) in enumerate(
                    k_folds_index_generator
            ):
                fold_metadata = self.post_processed_metadata.copy(deep=False)
                fold_metadata["subset"] = "train"

                # Update the subset column using the index of the samples for each fold
                fold_metadata.iloc[
                    test_index, fold_metadata.columns.get_loc("subset")
                ] = "test"
                self.folds_data[fold_index] = fold_metadata
                app_logger.info(
                    f"LocalDataset - Subsets creation- {fold_index + 1} fold created."
                    f"Train subset: {fold_metadata[fold_metadata['subset'] == 'train'].shape[0]} &"
                    f"Test subset: {fold_metadata[fold_metadata['subset'] == 'test'].shape[0]}"
                )

            app_logger.info(
                f"LocalDataset - Subsets creation- All folds created successfully."
            )
        except Exception as e:
            app_logger.error(
                f"LocalDataset - Subsets creation- K-folds process fails. Error: {e}"
            )
            raise MetadataError(e)

        return self.folds_data


# Create a class child class of LocalDataset with the name "AudioDataset"
class AudioDataset(LocalDataset):
    config_audio: dict
    raw_audio_data: dict = {}
    acoustic_feat_data: dict = {}

    def _create_an_audio_processor(self):
        return AudioProcessor(arguments=self.config_audio)

    def load_raw_data(
            self,
            column_with_path: str = "audio_id",
            num_cores: int = multiprocessing.cpu_count(),
    ):
        if self.raw_audio_data and self.raw_audio_data is not None:
            message = "AudioDataset - The raw audio data is already loaded."
            app_logger.warning(message)
            raise ValueError(message)

        try:
            feature_extractor = self._create_an_audio_processor()
            dict_ids_to_raw_data = feature_extractor.load_all_wav_files_from_dataset(
                dataset=self.post_processed_metadata,
                name_column_with_path=column_with_path,
                num_cores=num_cores,
            )

            # Create a dictionary of dictionaries with metadata and numpy arrays
            self.raw_audio_data = dict_ids_to_raw_data

            app_logger.info(
                f"AudioDataset - Loading raw data successful: {len(self.raw_audio_data)} raw examples."
            )
        except Exception as e:
            app_logger.error(
                f"AudioDataset - Loading raw data fails. Error: {e}",
            )
            raise AudioProcessingError(e)

        return dict_ids_to_raw_data

    def extract_acoustic_features(
            self,
            feat_name: str = None,
            num_cores: int = multiprocessing.cpu_count(),
    ):

        if not self.raw_audio_data:
            app_logger.warning("AudioDataset - No raw audio data loaded.")
            app_logger.info("AudioDataset - Loading raw data using a path.")
            self.load_raw_data_using_a_path()

        if feat_name is not None:
            self.config_audio["feature_type"] = feat_name

        try:
            feature_extractor = self._create_an_audio_processor()
            dict_ids_to_feats = feature_extractor.extract_features_from_raw_data(
                raw_data_matrix=self.raw_audio_data,
                num_cores=num_cores,
            )
            # Create a dictionary of dictionaries with metadata and numpy arrays
            self.acoustic_feat_data = {
                id_: {self.config_audio.get("feature_type"): feat}
                for id_, feat in dict_ids_to_feats.items()
            }

            # Save the new feat set
            self.save_dataset_as_a_serialized_object()
        except Exception as e:
            app_logger.error(
                f"AudioDataset - Extracting acoustic features fails. Error: {e}",
            )
            raise AudioProcessingError(e)

        return dict_ids_to_feats

    def extract_all_acoustic_features_supported(
            self,
            num_cores: int = multiprocessing.cpu_count(),
    ):

        if not self.raw_audio_data:
            app_logger.warning("AudioDataset - No raw audio data loaded.")
            app_logger.info("AudioDataset - Loading raw data using a path.")
            self.load_raw_data_using_a_path()

        supported_feats = src.features.audio_processor.SUPPORTED_FEATS

        for feat in supported_feats:
            try:
                self.config_audio["feature_type"] = feat
                feature_extractor = self._create_an_audio_processor()
                dict_ids_to_feats = feature_extractor.extract_features_from_raw_data(
                    raw_data_matrix=self.raw_audio_data,
                    num_cores=num_cores,
                )

                # Create a dictionary of dictionaries with metadata and numpy arrays
                self.acoustic_feat_data = {
                    id_: {self.config_audio.get("feature_type"): feat}
                    for id_, feat in dict_ids_to_feats.items()
                }

                # Save the new feat set
                self.save_dataset_as_a_serialized_object()

            except Exception as e:
                app_logger.error(
                    f"AudioDataset - Extracting acoustic features fails. Error: {e}",
                )
                raise AudioProcessingError(e)
        return self.acoustic_feat_data
