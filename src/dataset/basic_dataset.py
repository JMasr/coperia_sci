import logging
import multiprocessing
import os
import pickle
from abc import abstractmethod, ABC
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from sys import platform
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import KFold, train_test_split

import src.features.audio_processor
from src.exceptions import MetadataError, AudioProcessingError
from src.features.audio_processor import AudioProcessor


def increase_open_file_limit(new_limit=100000):
    if platform == "linux" or platform == "linux2":
        from resource import getrlimit, setrlimit, RLIMIT_NOFILE

        soft_limit, hard_limit = getrlimit(RLIMIT_NOFILE)
        setrlimit(RLIMIT_NOFILE, (new_limit, hard_limit))


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
    app_logger: logging.Logger

    column_with_ids: str
    column_with_target_class: str
    column_with_label_of_class: str

    filters: dict = {}
    raw_metadata: pd.DataFrame = pd.DataFrame()
    post_processed_metadata: pd.DataFrame = pd.DataFrame()

    num_folders: int = None
    size_test_folder: int = None
    size_train_folder: int = None

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def check_if_dataset_exists(path_to_dataset: str) -> bool:
        return os.path.exists(path_to_dataset)

    def save_dataset_as_a_serialized_object(self):
        path_to_save_the_dataset = f"{self.storage_path}.pkl"

        try:
            if os.path.exists(path_to_save_the_dataset):
                os.remove(path_to_save_the_dataset)
            else:
                os.makedirs(os.path.dirname(path_to_save_the_dataset), exist_ok=True)

            with open(path_to_save_the_dataset, "wb") as file:
                pickle.dump(self, file)

            self.app_logger.info(
                f"LocalDataset - The object was saved to {path_to_save_the_dataset}"
            )
        except Exception as e:
            self.app_logger.error(
                f"LocalDataset - Saving the dataset fails. Error: {e}"
            )
            raise MetadataError(e)

    def load_dataset_from_a_serialized_object(self, path_to_object: str = None):
        # Deserialize the object from a file
        try:
            if path_to_object is None:
                path_to_object = f"{self.storage_path}.pkl"

            with open(path_to_object, "rb") as file:
                dataset = pickle.load(file)
        except Exception as e:
            self.app_logger.error(
                f"LocalDataset - Loading the dataset fails. Error: {e}"
            )
            raise MetadataError(e)

        return dataset

    def load_metadata_from_csv(self, path_to_metadata: str, **kwargs):
        try:
            path_to_metadata = Path(path_to_metadata)
            with open(path_to_metadata, "r") as file:
                self.raw_metadata = pd.read_csv(file, **kwargs)
                self.raw_metadata.drop_duplicates(inplace=True)
            self.app_logger.info("LocalDataset - The CSV file was successful read")
        except ValueError as e:
            self.app_logger.error(f"LocalDataset - Pandas failed reading the CSV: {e}")
            raise MetadataError(f"{e}")
        except FileNotFoundError as e:
            self.app_logger.error(f"LocalDataset - The file wasn´t found: {e}")
            raise MetadataError(f"{e}")

    def sample_metadata(self, fraction: float = 0.1, seed: int = 42):
        self.raw_metadata = self.raw_metadata.sample(frac=fraction, random_state=seed)

    def transform_metadata(self, transformations: list):
        if self.raw_metadata.empty:
            message = "LocalDataset - Metadata is empty. Please, load a metadata first."
            self.app_logger.error(message)
            raise ValueError(message)
        if not transformations:
            message = "LocalDataset - No transformations were provided."
            self.app_logger.error(message)
            raise ValueError(message)
        else:
            try:
                self.app_logger.info(
                    f"LocalDataset - Starting {len(transformations)} transformations over metadata"
                )
                df = self.raw_metadata.copy(deep=False)
                for index, transformation in enumerate(transformations):
                    df = transformation(df)
                    self.app_logger.debug(
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

    def _make_conditions_for_transform_column_id_2_data_path(
            self, column_name: str, path: str, extension: str
    ) -> Tuple[str, str]:
        # Check if the column exists in the metadata
        if column_name not in self.post_processed_metadata.columns:
            raise MetadataError(f"The column {column_name} is not in the metadata.")

        # Check if string path is a valid path
        if os.path.exists(path) and os.path.isdir(path):
            path = os.path.abspath(path)
        else:
            raise MetadataError(f"The path {path} is not a valid directory.")

        # Check if extension has a dot
        if not extension.startswith("."):
            extension = "." + extension

        # Check if the column values end with the extension
        if self.post_processed_metadata[column_name].str.endswith(extension).all():
            extension = ""
        return path, extension

    def transform_column_id_2_data_path(
            self, column_name: str, path: str, extension: str
    ):
        path, extension = self._make_conditions_for_transform_column_id_2_data_path(
            column_name, path, extension
        )

        # Transform the column values to data paths
        self.post_processed_metadata[column_name] = self.post_processed_metadata[
            column_name
        ].apply(lambda x: os.path.join(path, x + extension))

    def get_index_for_1_fold(
            self,
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
        self.app_logger.debug(
            f"LocalDataset - Subset creation-"
            f" Results: {pat_train.shape[0]} training objects & {pat_test.shape[0]} test objects"
        )

        return pat_train, pat_test

    def _make_1_fold_subsets(
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
        self.app_logger.info(
            f"LocalDataset - Subsets creation- Starting the creation of train and test subsets"
        )

        # Prepare the metadata
        if self.post_processed_metadata.empty:
            self.app_logger.warning(
                "LocalDataset - Not transformation detected over the metadata."
                "Making the subsets using the original metadata."
            )
            self.post_processed_metadata = self.raw_metadata.copy()

        folds_data = {}
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

            folds_data[0] = exp_metadata

            # Log the lengths of the subsets
            self.app_logger.info(
                "LocalDataset - Subsets creation- Train and test subsets creation successful. "
                f"Train subset: {exp_metadata[exp_metadata['subset'] == 'train'].shape[0]} & "
                f"Test subset: {exp_metadata[exp_metadata['subset'] == 'test'].shape[0]}"
            )
        except Exception as e:
            self.app_logger.error(
                f"LocalDataset - Subsets creation - The subsets creation fails. Error: {e}"
            )
            raise MetadataError(e)

        return folds_data

    def _make_k_fold_subsets(
            self, target_class_for_fold: str, k_fold: int, seed: int
    ) -> dict:
        self.app_logger.info(
            f"LocalDataset - Subsets creation- Starting the creation of {k_fold}-folds"
        )
        # Prepare the metadata
        if self.post_processed_metadata.empty:
            self.app_logger.warning(
                "LocalDataset - Not transformation detected over the metadata."
                "Making the subsets using the original metadata."
            )
            self.post_processed_metadata = self.raw_metadata.copy()

        folds_data = {}
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
                folds_data[fold_index] = fold_metadata
                self.app_logger.info(
                    f"LocalDataset - Subsets creation- {fold_index + 1} fold created. "
                    f"Train subset: {fold_metadata[fold_metadata['subset'] == 'train'].shape[0]} & "
                    f"Test subset: {fold_metadata[fold_metadata['subset'] == 'test'].shape[0]}"
                )

            self.app_logger.info(
                f"LocalDataset - Subsets creation- All folds created successfully."
            )
        except Exception as e:
            self.app_logger.error(
                f"LocalDataset - Subsets creation- K-folds process fails. Error: {e}"
            )
            raise MetadataError(e)

        return folds_data

    def get_k_subsets(
            self,
            k_folds: int,
            test_size: float,
            seed: int,
    ) -> dict:
        target_class_for_fold = self.column_with_target_class
        target_label_for_fold = self.column_with_label_of_class

        # Record the number of folders
        self.num_folders = k_folds
        if k_folds <= 1:
            folds_data = self._make_1_fold_subsets(
                target_class_for_fold, target_label_for_fold, test_size, seed
            )

        elif k_folds >= 2:
            folds_data = self._make_k_fold_subsets(target_class_for_fold, k_folds, seed)
        else:
            raise MetadataError(
                f"The value for the number of folds is invalid: {k_folds}"
            )

        # Record the size of the folders
        self.size_test_folder = folds_data[0][folds_data[0]["subset"] == "test"].shape[0]
        self.size_train_folder = folds_data[0][folds_data[0]["subset"] == "train"].shape[0]

        return folds_data

    @abstractmethod
    def get_dataset_info(self):
        pass


# Create a class child class of LocalDataset with the name "AudioDataset"
class AudioDataset(LocalDataset, ABC):
    config_audio: dict
    dataset_raw_data_path: str

    raw_audio_data: dict = {}
    acoustic_feat_data: dict = {}

    class Config:
        arbitrary_types_allowed = True

    def _create_an_audio_processor(self) -> AudioProcessor:
        return AudioProcessor(arguments=self.config_audio)

    def load_raw_data(
            self,
            num_cores: int = multiprocessing.cpu_count(),
    ) -> Dict[str, np.ndarray]:
        if len(self.raw_audio_data) != 0 and self.raw_audio_data is not None:
            message = "AudioDataset - The raw audio data is already loaded."
            self.app_logger.warning(message)
            raise ValueError(message)

        if self.post_processed_metadata.empty:
            self.app_logger.warning(
                "AudioDataset - No post-processed metadata found. Loading raw data using raw metadata."
            )
            self.post_processed_metadata = self.raw_metadata

        self.transform_column_id_2_data_path(
            column_name=self.column_with_ids,
            path=self.dataset_raw_data_path,
            extension=".wav",
        )

        try:
            feature_extractor = self._create_an_audio_processor()
            dict_ids_to_raw_data = feature_extractor.load_all_wav_files_from_dataset(
                dataset=self.post_processed_metadata,
                name_column_with_path=self.column_with_ids,
                num_cores=num_cores,
            )

            # Create a dictionary of dictionaries with metadata and numpy arrays
            self.raw_audio_data = dict_ids_to_raw_data

            self.app_logger.info(
                f"AudioDataset - Loading raw data successful: {len(self.raw_audio_data)} raw examples."
            )
        except Exception as e:
            self.app_logger.error(
                f"AudioDataset - Loading raw data fails. Error: {e}",
            )
            raise AudioProcessingError(e)

        return dict_ids_to_raw_data

    def _check_if_feat_is_already_extracted(self, feat_name: str) -> bool:
        is_acoustic_feat_data_valid = len(self.acoustic_feat_data) == len(
            self.raw_audio_data
        )
        has_each_audio_the_feat_requested = all(
            [
                feat_name in feat_dict.keys()
                for feat_dict in self.acoustic_feat_data.values()
            ]
        )

        feat_is_storage_on_memory = (
                has_each_audio_the_feat_requested and is_acoustic_feat_data_valid
        )
        return feat_is_storage_on_memory

    def extract_acoustic_features(
            self,
            feat_name: str = None,
            num_cores: int = multiprocessing.cpu_count(),
    ) -> Dict[str, np.ndarray]:

        if not self.raw_audio_data:
            self.app_logger.warning("AudioDataset - No raw audio data loaded.")
            self.app_logger.info("AudioDataset - Loading raw data using a path.")
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
        except Exception as e:
            self.app_logger.error(
                f"AudioDataset - Extracting acoustic features fails. Error: {e}",
            )
            raise AudioProcessingError(e)

        return dict_ids_to_feats

    def extract_all_acoustic_features_supported(
            self,
            num_cores: int = multiprocessing.cpu_count(),
    ) -> Dict[str, dict]:
        if not self.raw_audio_data:
            self.app_logger.warning("AudioDataset - No raw audio data loaded.")
            self.app_logger.info("AudioDataset - Loading raw data using a path.")
            self.load_raw_data_using_a_path()

        supported_feats = src.features.audio_processor.SUPPORTED_FEATS
        acoustic_feat_data = {
            audio_id: {feat_name: None for feat_name in supported_feats}
            for audio_id in self.raw_audio_data.keys()
        }

        for feat_name in supported_feats:
            try:
                self.config_audio["feature_type"] = feat_name
                feature_extractor = self._create_an_audio_processor()

                dict_ids_to_feats: dict
                dict_ids_to_feats = feature_extractor.extract_features_from_raw_data(
                    raw_data_matrix=self.raw_audio_data,
                    num_cores=num_cores,
                )

                # Create a dictionary of dictionaries with metadata and numpy arrays
                for id_, feat in dict_ids_to_feats.items():
                    acoustic_feat_data[id_][feat_name] = feat

                self.save_dataset_as_a_serialized_object()

            except Exception as e:
                self.app_logger.error(
                    f"AudioDataset - Extracting acoustic features fails. Error: {e}",
                )
                raise AudioProcessingError(e)

        return self.acoustic_feat_data

    @staticmethod
    def _process_fold(
            dataframe: pd.DataFrame,
            fold: int,
            column_with_ids: str,
            column_with_labels_for_fold: str,
            acoustics_feat_name: str,
            acoustic_feat_data: dict,
            subset_at_sample_lv: bool = False,
    ) -> Tuple[int, dict]:
        try:
            # Get the column with target_label_for_fold and subset
            test_metadata = dataframe[dataframe["subset"] == "test"][
                [column_with_ids, column_with_labels_for_fold]
            ]
            train_metadata = dataframe[dataframe["subset"] == "train"][
                [column_with_ids, column_with_labels_for_fold]
            ]

            test_ids = set(test_metadata[column_with_ids])
            train_ids = set(train_metadata[column_with_ids])

            train_feats, train_labels, train_audio_id, train_spk_id = [], [], [], []
            test_feats, test_labels, test_audio_id, test_spk_id = [], [], [], []
            for id_, feats_of_id in acoustic_feat_data.items():

                spk_id = dataframe[dataframe[column_with_ids] == id_]["patient_id"].item()

                feat_of_id_sample = np.array(feats_of_id[acoustics_feat_name])

                label_of_id_sample = dataframe[dataframe[column_with_ids] == id_][
                    column_with_labels_for_fold
                ]
                label_of_id_sample = np.array(
                    [label_of_id_sample] * feat_of_id_sample.shape[0]
                )
                label_of_id_sample = label_of_id_sample.reshape(
                    feat_of_id_sample.shape[0], 1
                )

                if id_ in set(train_ids):
                    train_feats.append(feat_of_id_sample)
                    train_labels.append(label_of_id_sample)
                    for _ in range(feat_of_id_sample.shape[0]):
                        train_spk_id.append(spk_id)
                        train_audio_id.append(id_)

                elif id_ in set(test_ids):
                    test_feats.append(feat_of_id_sample)

                    if subset_at_sample_lv:
                        test_spk_id.append(spk_id * feat_of_id_sample.shape[0])
                        test_labels.append(label_of_id_sample)
                        test_audio_id.append(
                            np.array([id_] * feat_of_id_sample.shape[0])
                        )
                    else:
                        test_spk_id.append(spk_id)
                        test_labels.append(np.array([np.mean(label_of_id_sample)]))
                        test_audio_id.append(id_)

            train_feats = np.vstack(train_feats).astype(np.float32)
            train_labels = np.vstack(train_labels).astype(int)

            train_labels = train_labels.ravel()

            if subset_at_sample_lv:
                test_feats = np.vstack(test_feats, dtype=np.float32)
                test_labels = np.vstack(test_labels, dtype=int)

            result = {
                "train": {
                    "X": train_feats,
                    "y": train_labels,
                    column_with_ids: train_audio_id,
                    "spk_id": train_spk_id,
                },
                "test": {
                    "X": test_feats,
                    "y": test_labels,
                    column_with_ids: test_audio_id,
                    "spk_id": test_spk_id,
                },
            }

            return fold, result
        except Exception as e:
            raise MetadataError(e)

    def get_k_audio_subsets(
            self,
            acoustics_feat_name: str,
            k_folds: int = 5,
            seed: int = 42,
            test_size: float = 0.2,
            subset_at_sample_lv: bool = False,
    ) -> Tuple[dict, dict]:

        fold_data: dict = self.get_k_subsets(
            seed=seed,
            k_folds=k_folds,
            test_size=test_size,
        )

        folds_test_ids_to_feats_and_labels = {fold: {} for fold in fold_data.keys()}
        folds_train_ids_to_feats_and_labels = {fold: {} for fold in fold_data.keys()}

        try:
            for fold, dataframe in fold_data.items():
                _, result = self._process_fold(
                    dataframe,
                    fold,
                    self.column_with_ids,
                    self.column_with_label_of_class,
                    acoustics_feat_name,
                    self.acoustic_feat_data,
                    subset_at_sample_lv,
                )

                folds_train_ids_to_feats_and_labels[fold] = result["train"]
                folds_test_ids_to_feats_and_labels[fold] = result["test"]

        except Exception as e:
            self.app_logger.error(
                f"AudioDataset - Error while creating the folds. Error: {e}"
            )
            raise MetadataError(e)

        return folds_train_ids_to_feats_and_labels, folds_test_ids_to_feats_and_labels

    def get_k_audio_subsets_multiprocess(
            self,
            acoustics_feat_name: str,
            k_folds: int = 5,
            seed: int = 42,
            test_size: float = 0.2,
            subset_at_sample_lv: bool = False,
    ) -> Tuple[dict, dict]:

        fold_data: dict = self.get_k_subsets(
            seed=seed, k_folds=k_folds, test_size=test_size
        )

        self.config_audio["feature_type"] = acoustics_feat_name
        if not self._check_if_feat_is_already_extracted(acoustics_feat_name):
            self.app_logger.info(
                f"AudioDataset - Feats not found. Extracting the feature {acoustics_feat_name}"
            )
            self.extract_acoustic_features(acoustics_feat_name)

        folds_train_ids_to_feats_and_labels = {}
        folds_test_ids_to_feats_and_labels = {}

        try:
            increase_open_file_limit()
            max_workers = min(4, k_folds)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_fold = {
                    executor.submit(
                        self._process_fold,
                        dataframe,
                        fold,
                        self.column_with_ids,
                        self.column_with_label_of_class,
                        acoustics_feat_name,
                        self.acoustic_feat_data,
                        subset_at_sample_lv,
                    ): fold
                    for fold, dataframe in fold_data.items()
                }

                for future in as_completed(future_to_fold):
                    fold, result = future.result()
                    folds_train_ids_to_feats_and_labels[fold] = result["train"]
                    folds_test_ids_to_feats_and_labels[fold] = result["test"]
        except Exception as e:
            self.app_logger.error(
                f"AudioDataset - Error while creating the folds. Error: {e}"
            )
            raise MetadataError(e)

        return folds_train_ids_to_feats_and_labels, folds_test_ids_to_feats_and_labels

    def get_dataset_info(self):
        return {
            "name": self.name,
            "storage_path": self.storage_path,
            "column_with_ids": self.column_with_ids,
            "column_with_target_class": self.column_with_target_class,
            "column_with_label_of_class": self.column_with_label_of_class,
            "filters": self.filters,
            "raw_metadata_size": len(self.raw_metadata),
            "post_processed_metadata_size": len(self.post_processed_metadata),
            "num_folders": self.num_folders,
            "size_test_folder": self.size_test_folder,
            "size_train_folder": self.size_train_folder,
            "feature_config": self.config_audio,
            "dataset_raw_data_path": self.dataset_raw_data_path,
            "raw_audio_data_size": len(self.raw_audio_data),
            "acoustic_feat_data": self.acoustic_feat_data,
        }
