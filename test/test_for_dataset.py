import pytest
import pandas as pd
from pathlib import Path
from src.dataset.basic_dataset import LocalDataset


class TestLocalDataset:
    @classmethod
    def setup_class(cls):
        # Create a temporary CSV file for testing
        cls.str_path_temp_file = "temp.csv"
        cls.temp_file = Path(cls.str_path_temp_file)
        cls.temp_file.write_text('key,value\n1,"2,0"\n2,4\n3,8', encoding="utf-8")

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
        dataset = LocalDataset(name)

        # Assert
        assert dataset.name == name
        assert dataset.raw_metadata.empty
        assert dataset.post_processed_metadata.empty
        assert dataset.subsets == []

    def test_invalid_initialization_of_localdataset(self):
        # Arrange
        name = 123

        # Act & Assert
        with pytest.raises(TypeError):
            dataset = LocalDataset(name)

    def test_valid_transformations_over_metadata_in_a_localdataset(self):
        # Arrange
        dataset = LocalDataset(name="TEST-DATASET")
        dataset.load_metadata_from_csv(self.str_path_temp_file, decimal=",")

        def transformation(df: pd.DataFrame) -> pd.DataFrame:
            # Multiply all values by 2
            df["value"] = df["value"].astype(int) * 2
            return df

        # Act
        dataset.transform_metadata([transformation])

        # Assert
        assert not dataset.post_processed_metadata.empty
        assert dataset.post_processed_metadata["value"].tolist() == [4, 8, 16]
        assert dataset.raw_metadata["value"].tolist() == [2.0, 4, 8]

    def test_invalid_empty_transformations_provided_for_localdataset(self):
        # Arrange
        dataset = LocalDataset(name="TEST-DATASET")

        # Act & Assert
        with pytest.raises(ValueError):
            dataset.transform_metadata([])

# TODO: The class LocalDataset isn´t fully tested

if __name__ == "__main__":
    # Run all tests in the module
    pytest.main()
