import pytest
import pandas as pd

from pathlib import Path

from src.dataset.basic_dataset import LocalDataset


def test_load_metadata_of_dataset_from_a_csv():
    # Arrange
    temp_file = Path("temp.csv")
    temp_file.write_text('key,value\n1,"2,017"', encoding="utf-8")

    # Act
    dataset = LocalDataset(name="TEST-DATASET")
    dataset.load_metadata_from_csv("temp.csv", decimal=",")

    # Assert
    assert isinstance(dataset.raw_metadata, pd.DataFrame)

    # Cleanup
    temp_file.unlink()


def test_valid_initialization_of_localdataset():
    # Arrange
    name = "TEST-DATASET"

    # Act
    dataset = LocalDataset(name)

    # Assert
    assert dataset.name == name
    assert dataset.raw_metadata.empty
    assert dataset.post_processed_metadata.empty
    assert dataset.subsets == []


def test_invalid_initialization_of_localdataset():
    # Arrange
    name = 123

    # Act & Assert
    with pytest.raises(TypeError):
        dataset = LocalDataset(name)


def test_valid_transformations_over_metadata_in_a_localdataset():
    # Arrange
    temp_file = Path("temp.csv")
    temp_file.write_text('key,value\n1,"2,0"\n2,4\n3,8', encoding="utf-8")

    dataset = LocalDataset(name="TEST-DATASET")
    dataset.load_metadata_from_csv("temp.csv", decimal=",")

    def transformation(df: pd.DataFrame) -> pd.DataFrame:
        # Multiply the all values by 2
        df["value"] = df["value"].astype(int) * 2
        return df

    # Act
    dataset.transform_metadata([transformation])

    # Assert
    assert not dataset.post_processed_metadata.empty
    # Check the transformation. The value of each position should be multiplied by 2
    assert dataset.post_processed_metadata["value"].tolist() == [4, 8, 16]
    assert dataset.raw_metadata["value"].tolist() == [2.0, 4, 8]

    # Cleanup
    temp_file.unlink()


# TODO: The class LocalDataset isn´t fully tested

if __name__ == "__main__":
    # Run all tests in the module
    pytest.main()
