from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from faker import Faker
from pydantic import ValidationError

from src.dataset.basic_dataset import LocalDataset


def generate_fake_dataframe_with_n_rows(n: int):
    fake = Faker()
    ids = [fake.uuid4() for _ in range(n)]
    labels = [fake.boolean() for _ in range(n)]
    data = np.random.randint(0, 100, size=n)
    genders = [fake.random_element(["Male", "Female"]) for _ in range(n)]
    return pd.DataFrame({"id": ids, "label": labels, "data": data, "gender": genders})


class TestLocalDatasetShould:

    @classmethod
    def setup_class(cls):
        # Create a temporary CSV file for testing
        cls.str_path_temp_file = "temp.csv"
        cls.temp_file = Path(cls.str_path_temp_file)
        cls.fake_dataframe = generate_fake_dataframe_with_n_rows(300)
        cls.fake_dataframe.to_csv(cls.temp_file, index=False, sep=",")

    @classmethod
    def teardown_class(cls):
        # Remove the temporary CSV file after testing
        cls.temp_file.unlink()

    def test_load_metadata_of_dataset_from_a_csv(self):
        # Arrange
        dataset = LocalDataset(name="TEST-DATASET")

        # Act
        dataset.load_metadata_from_csv(self.str_path_temp_file, decimal=",")

        # Assert
        assert isinstance(dataset.raw_metadata, pd.DataFrame)

    def test_valid_initialization_of_localdataset(self):
        # Arrange
        name = "TEST-DATASET"

        # Act
        dataset = LocalDataset(name=name)

        # Assert
        assert dataset.name == name
        assert dataset.raw_metadata.empty
        assert dataset.post_processed_metadata.empty

    def test_invalid_initialization_of_localdataset(self):
        # Arrange
        name = 123

        # Act & Assert
        with pytest.raises(ValidationError):
            dataset = LocalDataset(name=name)
            print(dataset.name)

    def test_valid_transformations_over_metadata_in_a_localdataset(self):
        # Arrange
        dataset = LocalDataset(name="TEST-DATASET")
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
        dataset = LocalDataset(name="TEST-DATASET")

        # Act & Assert
        with pytest.raises(ValueError):
            dataset.transform_metadata([])

    def test_valid_subset_making_in_a_localdataset(self):
        # Arrange
        dataset = LocalDataset(name="TEST-DATASET")
        dataset.load_metadata_from_csv(self.str_path_temp_file)

        # Act
        dataset_ready = dataset.make_1_fold_subsets(
            target_class_for_fold="id",
            target_label_for_fold="label",
            test_size=0.2,
            seed=42,
        )

        fold: pd.DataFrame = dataset_ready.get(0)

        # Assert
        assert isinstance(dataset_ready, dict)
        assert dataset_ready == dataset.folds_data

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
        dataset = LocalDataset(name="TEST-DATASET")
        dataset.load_metadata_from_csv(self.str_path_temp_file)

        # Act
        dataset_ready = dataset.make_k_fold_subsets(
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


# TODO: Add test for AudioDataset

if __name__ == "__main__":
    # Run all tests in the module
    pytest.main()
