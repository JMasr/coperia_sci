import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from faker import Faker
from pydantic import ValidationError

from src.dataset.basic_dataset import LocalDataset, AudioDataset
from src.features.audio_processor import SUPPORTED_FEATS
from src.logger import BasicLogger
from test import ROOT_PATH


def mock_a_dataframe_with_metadata(number_of_rows: int = 100):
    fake = Faker()
    ids = [fake.uuid4() for _ in range(number_of_rows)]
    labels = [fake.boolean() for _ in range(number_of_rows)]
    data = np.random.randint(0, 100, size=number_of_rows)
    genders = [fake.random_element(["Male", "Female"]) for _ in range(number_of_rows)]
    return pd.DataFrame({"id": ids, "label": labels, "data": data, "gender": genders})


def mock_a_dataframe_with_metadata_and_audio(
        path_to_temp_folder: str, sample_rate: int, number_of_rows: int = 100
):
    df_with_metadata = mock_a_dataframe_with_metadata(number_of_rows)

    # Create a temporary audio file
    os.makedirs(path_to_temp_folder, exist_ok=True)

    paths_to_dummy_audio_files = []
    for i in df_with_metadata["id"]:
        path_to_dummy_valid_signal = os.path.join(
            path_to_temp_folder, f"{i}.wav"
        )
        valid_dummy_signal = np.random.uniform(-1, 1, size=(sample_rate * 10, 2))
        sf.write(
            path_to_dummy_valid_signal,
            valid_dummy_signal,
            sample_rate,
            subtype="PCM_24",
        )
        paths_to_dummy_audio_files.append(path_to_dummy_valid_signal)
    df_with_metadata["path"] = paths_to_dummy_audio_files

    return df_with_metadata


class TestLocalDatasetShould:

    @classmethod
    def setup_class(cls):
        # Create a temporary resources for testing
        cls.str_path_temp_folder = os.path.join(ROOT_PATH, "test", "temp_folder")
        os.makedirs(cls.str_path_temp_folder, exist_ok=True)

        cls.str_path_temp_file = os.path.join(cls.str_path_temp_folder, "temp_file.csv")
        cls.temp_file = Path(cls.str_path_temp_file)

        cls.num_rows = 50
        cls.mock_dataframe = mock_a_dataframe_with_metadata(cls.num_rows)
        cls.mock_dataframe.to_csv(cls.temp_file, index=False, sep=",")

        # Create a logger
        cls.app_logger = BasicLogger(
            log_file=os.path.join(cls.str_path_temp_folder, "logs", "experiment.log")
        ).get_logger()

    @classmethod
    def teardown_class(cls):
        # Remove the temporary CSV file after testing
        cls.temp_file.unlink()

    def test_load_metadata_of_dataset_from_a_csv(self):
        # Arrange
        dataset = LocalDataset(
            name="TEST-DATASET",
            column_with_ids="id",
            column_with_target_class="data",
            column_with_label_of_class="label",
            storage_path=self.str_path_temp_folder,
            app_logger=self.app_logger,
        )

        # Act
        dataset.load_metadata_from_csv(self.str_path_temp_file, decimal=",")

        # Assert
        assert isinstance(dataset.raw_metadata, pd.DataFrame)

    def test_valid_initialization_of_localdataset(self):
        # Arrange
        name = "TEST-DATASET"

        # Act
        dataset = LocalDataset(
            name=name,
            column_with_ids="id",
            column_with_target_class="data",
            column_with_label_of_class="label",
            storage_path=self.str_path_temp_folder,
            app_logger=self.app_logger,
        )

        # Assert
        assert dataset.name == name
        assert dataset.raw_metadata.empty
        assert dataset.post_processed_metadata.empty

    def test_invalid_initialization_of_localdataset(self):
        # Arrange
        name = 123

        # Act & Assert
        with pytest.raises(ValidationError):
            LocalDataset(
                name=name,
                column_with_ids="id",
                column_with_target_class="data",
                column_with_label_of_class="label",
                storage_path=self.str_path_temp_folder,
                app_logger=self.app_logger,
            )

    def test_valid_transformations_over_metadata_in_a_localdataset(self):
        # Arrange
        dataset = LocalDataset(
            name="TEST-DATASET",
            column_with_ids="id",
            column_with_target_class="data",
            column_with_label_of_class="label",
            storage_path=self.str_path_temp_folder,
            app_logger=self.app_logger,
        )
        dataset.load_metadata_from_csv(self.str_path_temp_file)

        def transformation(df: pd.DataFrame) -> pd.DataFrame:
            # Multiply all values by 2
            df["data"] = df["data"].astype(int) * 2
            return df

        # Act
        dataset.transform_metadata([transformation])

        # Assert
        assert not dataset.post_processed_metadata.empty
        assert (
                dataset.post_processed_metadata["data"]
                == self.mock_dataframe["data"].astype(int) * 2
        ).all()
        assert (dataset.raw_metadata["data"] == self.mock_dataframe["data"]).all()

    def test_invalid_empty_transformations_provided_for_localdataset(self):
        # Arrange
        dataset = LocalDataset(
            name="TEST-DATASET",
            column_with_ids="id",
            column_with_target_class="data",
            column_with_label_of_class="label",
            storage_path=self.str_path_temp_folder,
            app_logger=self.app_logger,
        )

        # Act & Assert
        with pytest.raises(ValueError):
            dataset.transform_metadata([])

    def test_valid_subset_making_in_a_localdataset(self):
        # Arrange
        dataset = LocalDataset(
            name="TEST-DATASET",
            column_with_ids="id",
            column_with_target_class="data",
            column_with_label_of_class="label",
            storage_path=self.str_path_temp_folder,
            app_logger=self.app_logger,
        )
        dataset.load_metadata_from_csv(self.str_path_temp_file)

        # Act
        dataset_ready = dataset._make_1_fold_subsets(
            target_class_for_fold="id",
            target_label_for_fold="label",
            test_size=0.2,
            seed=42,
        )

        fold: pd.DataFrame = dataset_ready.get(0)

        # Assert
        assert isinstance(dataset_ready, dict)
        assert len(dataset_ready) == 1

        assert (
                fold.loc[fold["subset"] == "train"].shape[0]
                == dataset.post_processed_metadata.shape[0] * 0.8
        )
        assert fold.loc[fold["subset"] == "test"].shape[0] == int(
            dataset.post_processed_metadata.shape[0] * 0.2
        )

    @pytest.mark.parametrize(
        "k_fold", [2, 4, 6]
    )  # Folds must be multiples of 2 when the dataset has a pair #s of rows
    def test_valid_k_fold_subset_making_in_a_localdataset(self, k_fold: int):
        # Arrange
        dataset = LocalDataset(
            name="TEST-DATASET",
            column_with_ids="id",
            column_with_target_class="data",
            column_with_label_of_class="label",
            storage_path=self.str_path_temp_folder,
            app_logger=self.app_logger,
        )
        dataset.load_metadata_from_csv(self.str_path_temp_file)

        # Act
        dataset_ready = dataset._make_k_fold_subsets(
            target_class_for_fold="gender", k_fold=k_fold, seed=42
        )

        # Assert
        assert isinstance(dataset_ready, dict)
        assert len(dataset_ready) == k_fold
        assert all(isinstance(fold, pd.DataFrame) for fold in dataset_ready.values())

        for fold in dataset_ready.values():
            assert (
                           fold.loc[fold["subset"] == "train"].shape[0]
                           - dataset.post_processed_metadata.shape[0] * (1 - 1 / k_fold)
                   ) < 1
            assert (
                           fold.loc[fold["subset"] == "test"].shape[0]
                           - dataset.post_processed_metadata.shape[0] * (1 / k_fold)
                   ) < 1


class TestAudioDatasetShould:
    @classmethod
    def setup_class(cls):
        # Create a temporary resources for testing
        cls.str_path_temp_folder = os.path.join(ROOT_PATH, "test", "temp_folder")
        os.makedirs(cls.str_path_temp_folder, exist_ok=True)

        cls.num_rows = 50
        cls.sample_rate = 16000
        cls.mock_dataframe = mock_a_dataframe_with_metadata_and_audio(
            cls.str_path_temp_folder, cls.sample_rate, cls.num_rows
        )

        cls.temp_file = Path(os.path.join(cls.str_path_temp_folder, "temp_file.csv"))
        cls.mock_dataframe.to_csv(cls.temp_file, index=False, sep=",")

        # Create defaults configurations for audio processing
        cls.seed = 42
        cls.k_fold = 2
        cls.test_size = 0.2

        cls.dataset_name = "TEST-DATASET"
        cls.filter_testing = {"audio_type": ["ALL"], "audio_moment": ["ALL"]}
        cls.config_audio = {
            "feature_type": "compare_2016_energy",
            "top_db": 30,
            "pre_emphasis_coefficient": 0.97,
            "resampling_rate": 44100,
            "n_mels": 64,
            "n_mfcc": 32,
            "plp_order": 13,
            "conversion_approach": "Wang",
            "f_max": 22050,
            "f_min": 100,
            "window_size": 25.0,
            "hop_length": 10.0,
            "window_type": "hamming",
            "normalize": "mvn",
            "use_energy": True,
            "apply_mean_norm": True,
            "apply_vari_norm": True,
            "compute_deltas_feats": True,
            "compute_deltas_deltas_feats": True,
            "compute_opensmile_extra_features": True,
        }
        cls.feature_name = cls.config_audio["feature_type"]

        # Create a logger
        cls.app_logger = BasicLogger(
            log_file=os.path.join(cls.str_path_temp_folder, "logs", "experiment.log")
        ).get_logger()

    @classmethod
    def teardown_class(cls):
        # Remove the temporary CSV file after testing
        cls.temp_file.unlink()

    @pytest.mark.parametrize("feat_type", SUPPORTED_FEATS)
    def test_valid_initialization_of_audiodataset(self, feat_type: str):
        # Arrange
        config_audio = self.config_audio.copy()
        config_audio["feature_type"] = feat_type

        # Act
        dataset = AudioDataset(
            name=self.dataset_name,
            storage_path=self.str_path_temp_folder,
            app_logger=self.app_logger,
            column_with_ids="id",
            column_with_target_class="data",
            column_with_label_of_class="label",
            filters=self.filter_testing,
            config_audio=config_audio,
            dataset_raw_data_path=self.str_path_temp_folder,
        )

        # Assert
        assert dataset.name == self.dataset_name
        assert dataset.raw_metadata.empty
        assert dataset.post_processed_metadata.empty

    def test_valid_load_metadata_of_audiodataset_from_a_csv(self):
        # Arrange
        dataset = AudioDataset(
            name=self.dataset_name,
            storage_path=self.str_path_temp_folder,
            app_logger=self.app_logger,
            column_with_ids="id",
            column_with_target_class="data",
            column_with_label_of_class="label",
            filters=self.filter_testing,
            config_audio=self.config_audio,
            dataset_raw_data_path=self.str_path_temp_folder,
        )

        # Act
        dataset.load_metadata_from_csv(str(self.temp_file), decimal=",")

        # Assert
        assert isinstance(dataset.raw_metadata, pd.DataFrame)
        assert dataset.raw_metadata.shape[0] == self.num_rows
        assert dataset.post_processed_metadata.empty

    def test_valid_appalling_transformations_over_metadata_in_a_audiodataset(self):
        # Arrange
        dataset = AudioDataset(
            name=self.dataset_name,
            storage_path=self.str_path_temp_folder,
            app_logger=self.app_logger,
            column_with_ids="id",
            column_with_target_class="data",
            column_with_label_of_class="label",
            filters=self.filter_testing,
            config_audio=self.config_audio,
            dataset_raw_data_path=self.str_path_temp_folder,
        )

        def transformation_bool_to_int(df: pd.DataFrame) -> pd.DataFrame:
            df["label"] = df["label"].map({True: 1, False: 0}).astype(int)
            return df

        def transformation_gender_to_tabular(df: pd.DataFrame) -> pd.DataFrame:
            # Transform the unique values on column gender to integers
            df["gender"] = df["gender"].map({"Male": 0, "Female": 1}).astype(int)
            return df

        # Act
        dataset.load_metadata_from_csv(str(self.temp_file), decimal=",")
        dataset.transform_metadata(
            [transformation_bool_to_int, transformation_gender_to_tabular]
        )

        # Assert
        assert not dataset.post_processed_metadata.empty
        assert dataset.post_processed_metadata["label"].dtype == int
        assert dataset.post_processed_metadata["gender"].dtype == int

    def test_valid_load_raw_data_of_audiodataset(self):
        # Arrange
        dataset = AudioDataset(
            name=self.dataset_name,
            storage_path=self.str_path_temp_folder,
            app_logger=self.app_logger,
            column_with_ids="id",
            column_with_target_class="data",
            column_with_label_of_class="label",
            filters=self.filter_testing,
            config_audio=self.config_audio,
            dataset_raw_data_path=self.str_path_temp_folder,
        )
        dataset.load_metadata_from_csv(str(self.temp_file), decimal=",")

        # Act
        dataset.load_raw_data()

        # Assert
        assert isinstance(dataset.raw_metadata, pd.DataFrame)
        assert dataset.raw_metadata.shape[0] == self.num_rows
        assert len(dataset.raw_audio_data) == self.num_rows

    def test_valid_k_fold_of_feats(self):
        # Arrange
        dataset = AudioDataset(
            name=self.dataset_name,
            storage_path=self.str_path_temp_folder,
            app_logger=self.app_logger,
            column_with_ids="id",
            column_with_target_class="data",
            column_with_label_of_class="label",
            filters=self.filter_testing,
            config_audio=self.config_audio,
            dataset_raw_data_path=self.str_path_temp_folder,
        )
        dataset.load_metadata_from_csv(str(self.temp_file), decimal=",")
        dataset.load_raw_data()
        dataset.extract_acoustic_features(self.feature_name)

        folds_train, folds_test = dataset.get_k_audio_subsets_multiprocess(
            acoustics_feat_name=self.feature_name,
            test_size=self.test_size,
            k_folds=self.k_fold,
            seed=self.seed,
        )

        # Assert
        assert isinstance(folds_train, dict)
        assert isinstance(folds_test, dict)
        assert len(folds_train) == self.k_fold
        assert len(folds_test) == self.k_fold

    # Testing all Supported Features takes a long time, so I will test only two
    @pytest.mark.parametrize("feat_type", ["compare_2016_energy", "compare_2016_llds"])
    def test_valid_extract_acoustic_feats(self, feat_type: str):
        # Arrange
        config_audio = self.config_audio.copy()
        config_audio["feature_type"] = feat_type

        dataset = AudioDataset(
            name=self.dataset_name,
            storage_path=self.str_path_temp_folder,
            app_logger=self.app_logger,
            column_with_ids="id",
            column_with_target_class="data",
            column_with_label_of_class="label",
            filters=self.filter_testing,
            config_audio=config_audio,
            dataset_raw_data_path=self.str_path_temp_folder,
        )
        dataset.load_metadata_from_csv(str(self.temp_file), decimal=",")
        dataset.load_raw_data()

        # Act
        dataset.extract_acoustic_features(feat_type)

        # Assert
        assert isinstance(dataset.raw_metadata, pd.DataFrame)
        assert dataset.raw_metadata.shape[0] == self.num_rows
        assert len(dataset.raw_audio_data) == self.num_rows
        assert len(dataset.acoustic_feat_data) == self.num_rows


if __name__ == "__main__":
    # Run all tests in the module
    pytest.main()
