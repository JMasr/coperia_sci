import json
from pathlib import Path

import pandas as pd


def is_str_path_an_existent_file(str_path: str) -> bool:
    """
    Check if a path exists and is a file.
    :param str_path: Path to check.
    :return: True if the path exists and is a file, False otherwise.
    """
    path = Path(str_path)
    return path.exists() and path.is_file()


def is_str_path_a_file_with_extension(str_path: str, extension: str) -> bool:
    """
    Check if a path is a file with the specified extension.
    :param str_path: Path to check.
    :param extension: Extension to check.
    :return: True if the path is a file with the specified extension, False otherwise.
    """
    path = Path(str_path)
    return path.suffix == extension


def json_file_to_dict(str_with_path: str) -> dict:
    """
    Load a json file as a dictionary. Useful to load the configuration of the experiments
    :param str_with_path: path to the json file
    :return: dictionary with the configuration
    """
    try:
        path = Path(str_with_path)
        with path.open("r", encoding="utf-8") as file:
            json_as_dict = json.loads(file.read())

        if json_as_dict == {}:
            raise ValueError("The specified file is empty or has an empty dictionary.")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File '{str_with_path}' does not exist.") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"An error occurred while decoding the JSON file: {e}") from e
    except Exception as e:
        raise ValueError(f"An error occurred while reading the JSON file: {e}") from e

    return json_as_dict

def csv_file_to_dataframe(str_path_to_csv: str, **kwargs):
    """
    Read a CSV file into a pandas DataFrame.
    :param str_path_to_csv: Path to the CSV file.
    :return: DataFrame containing the data from the CSV file.
    """
    if (is_str_path_an_existent_file(str_path_to_csv) and
            is_str_path_a_file_with_extension(str_path_to_csv, ".csv")):
        try:
            dataframe = pd.read_csv(str_path_to_csv, **kwargs)
            return dataframe
        except Exception as e:
            raise ValueError(f"An error occurred while reading the CSV file: {e}")
    else:
        raise FileNotFoundError(f"File '{str_path_to_csv}' does not exist or is not a csv file.")
