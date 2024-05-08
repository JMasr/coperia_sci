import pandas as pd
import pytest
from pathlib import Path

from src.files import json_file_to_dict, csv_file_to_dataframe


def test_loads_valid_json_file():
    # Create a temporary JSON file
    temp_file = Path("temp.json")
    temp_file.write_text('{"key": "value"}')

    # Call the function under test
    result = json_file_to_dict("temp.json")

    # Assert that the result is a dictionary
    assert isinstance(result, dict)

    # Assert that the result contains the expected key-value pair
    assert result == {"key": "value"}

    # Clean up the temporary file
    temp_file.unlink()


def test_load_a_non_existent_json():
    # Call the function under test with a non-existent file
    with pytest.raises(FileNotFoundError):
        json_file_to_dict("nonexistent.json")


def test_load_an_empty_json_file():
    # Create a temporary JSON file
    temp_file = Path("temp.json")
    temp_file.write_text('')

    # Call the function under test
    with pytest.raises(ValueError):
        json_file_to_dict("temp.json")

    # Clean up the temporary file
    temp_file.unlink()


def test_load_an_json_file_with_an_empty_dict():
    # Create a temporary JSON file
    temp_file = Path("temp.json")
    temp_file.write_text('{}')

    # Call the function under test
    with pytest.raises(ValueError):
        json_file_to_dict("temp.json")

    # Clean up the temporary file
    temp_file.unlink()


def test_valid_path_to_csv():
    # Arrange
    temp_file = Path("temp.csv")
    temp_file.write_text('key,value\n1,"2,017"')

    # Act
    result = csv_file_to_dataframe("temp.csv", decimal=",")

    # Assert
    assert isinstance(result, pd.DataFrame)

    # Clean up
    temp_file.unlink()


def test_non_existent_csv():
    # Arrange
    str_path_to_csv = "path/to/non_existent/file.csv"

    # Act and Assert
    with pytest.raises(FileNotFoundError):
        csv_file_to_dataframe(str_path_to_csv)


if __name__ == "__main__":
    # Run all tests in the module
    pytest.main()
