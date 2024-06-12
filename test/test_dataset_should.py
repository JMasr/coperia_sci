import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from faker import Faker
from pydantic import ValidationError

from src.dataset.basic_dataset import LocalDataset
from test import ROOT_PATH


def mock_a_dataframe_with_metadata(number_of_rows: int = 100):
    fake = Faker()
    ids = [fake.uuid4() for _ in range(number_of_rows)]
    labels = [fake.boolean() for _ in range(number_of_rows)]
    data = np.random.randint(0, 100, size=number_of_rows)
    genders = [fake.random_element(["Male", "Female"]) for _ in range(number_of_rows)]
    return pd.DataFrame({"id": ids, "label": labels, "data": data, "gender": genders})


def mock_a_dataframe_with_metadata_and_audio(path_to_temp_folder: str, sample_rate: int, number_of_rows: int = 100,
                                             sf=None):
    df_with_metadata = mock_a_dataframe_with_metadata(number_of_rows)

    # Create a temporary audio file
    os.makedirs(path_to_temp_folder, exist_ok=True)

    paths_to_dummy_audio_files = []
    for i in range(number_of_rows):
        path_to_dummy_valid_signal = os.path.join(path_to_temp_folder, f"valid_dummy_wav_{i}.wav")
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

        cls.num_rows = 300
        cls.fake_dataframe = mock_a_dataframe_with_metadata(cls.num_rows)
        cls.fake_dataframe.to_csv(cls.temp_file, index=False, sep=",")

    @classmethod
    def teardown_class(cls):
        # Remove the temporary CSV file after testing
        cls.temp_file.unlink()

    def test_load_metadata_of_dataset_from_a_csv(self):
        # Arrange
        dataset = LocalDataset(
            name="TEST-DATASET", storage_path=self.str_path_temp_folder
        )

        # Act
        dataset.load_metadata_from_csv(self.str_path_temp_file, decimal=",")

        # Assert
        assert isinstance(dataset.raw_metadata, pd.DataFrame)

    def test_valid_initialization_of_localdataset(self):
        # Arrange
        name = "TEST-DATASET"

        # Act
        dataset = LocalDataset(name=name, storage_path=self.str_path_temp_folder)

        # Assert
        assert dataset.name == name
        assert dataset.raw_metadata.empty
        assert dataset.post_processed_metadata.empty

    def test_invalid_initialization_of_localdataset(self):
        # Arrange
        name = 123

        # Act & Assert
        with pytest.raises(ValidationError):
            dataset = LocalDataset(name=name, storage_path=self.str_path_temp_folder)
            print(dataset.name)

    def test_valid_transformations_over_metadata_in_a_localdataset(self):
        # Arrange
        dataset = LocalDataset(
            name="TEST-DATASET", storage_path=self.str_path_temp_folder
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
                == self.fake_dataframe["data"].astype(int) * 2
        ).all()
        assert (dataset.raw_metadata["data"] == self.fake_dataframe["data"]).all()

    def test_invalid_empty_transformations_provided_for_localdataset(self):
        # Arrange
        dataset = LocalDataset(
            name="TEST-DATASET", storage_path=self.str_path_temp_folder
        )

        # Act & Assert
        with pytest.raises(ValueError):
            dataset.transform_metadata([])

    def test_valid_subset_making_in_a_localdataset(self):
        # Arrange
        dataset = LocalDataset(
            name="TEST-DATASET", storage_path=self.str_path_temp_folder
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
        assert (
                fold.loc[fold["subset"] == "test"].shape[0]
                == dataset.post_processed_metadata.shape[0] * 0.2
        )

    @pytest.mark.parametrize(
        "k_fold", [2, 4, 6]
    )  # Folds must be multiples of 2 when the dataset has a pair #s of rows
    def test_valid_k_fold_subset_making_in_a_localdataset(self, k_fold: int):
        # Arrange
        dataset = LocalDataset(
            name="TEST-DATASET", storage_path=self.str_path_temp_folder
        )
        dataset.load_metadata_from_csv(self.str_path_temp_file)

        # Act
        dataset_ready = dataset._make_k_fold_subsets(
            target_class_for_fold="id", k_fold=k_fold, seed=42
        )

        # Assert
        assert isinstance(dataset_ready, dict)
        assert len(dataset_ready) == k_fold
        assert all(isinstance(fold, pd.DataFrame) for fold in dataset_ready.values())

        for fold in dataset_ready.values():
            assert fold.loc[fold["subset"] == "train"].shape[
                       0
                   ] == dataset.post_processed_metadata.shape[0] * (1 - 1 / k_fold)
            assert (
                    fold.loc[fold["subset"] == "test"].shape[0]
                    == dataset.post_processed_metadata.shape[0] * 1 / k_fold
            )


class TestAudioDatasetShould:
    @classmethod
    def setup_class(cls):
        # Create a temporary resources for testing
        cls.str_path_temp_folder = os.path.join(ROOT_PATH, "test", "temp_folder")
        os.makedirs(cls.str_path_temp_folder, exist_ok=True)

        cls.num_rows = 300
        cls.sample_rate = 16000
        cls.fake_dataframe = mock_a_dataframe_with_metadata_and_audio(
            cls.str_path_temp_folder, cls.sample_rate, cls.num_rows
        )

        cls.temp_file = Path(os.path.join(cls.str_path_temp_folder, "temp_file.csv"))
        cls.fake_dataframe.to_csv(cls.temp_file, index=False, sep=",")

    @classmethod
    def teardown_class(cls):
        # Remove the temporary CSV file after testing
        cls.temp_file.unlink()


if __name__ == "__main__":
    # Run all tests in the module
    pytest.main()
