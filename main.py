import os.path
from pathlib import Path

import pandas as pd

from src.files import json_file_to_dict
from src.logger import app_logger
from src.dataset.basic_dataset import LocalDataset


def make_dicoperia_metadata(metadata: pd.DataFrame,
                            filters_: dict = None,
                            remove_samples: dict = None):
    """
    Make a metadata file for the COPERIA dataset filtering some columns
    :param metadata: a list with all the audio samples in COPERIA as an Audio class
    :param filters_: a dictionary with the columns and values to keep
    :param remove_samples: a dictionary with the columns and values to remove
    :return: a pandas dataframe with the metadata of the DICOPERIA dataset
    """
    df = metadata.copy()

    if filters_ is None:
        filters_ = {'patient_type': ['covid-control', 'covid-persistente']}

    if remove_samples is None:
        remove_samples = {'audio_id': ['c15e54fc-5290-4652-a3f7-ff3b779bd980', '244b61cc-4fd7-4073-b0d8-7bacd42f6202'],
                          'patient_id': ['coperia-rehab']}

    for key, values in remove_samples.items():
        df = df[~df[key].isin(values)]

    for key, values in filters_.items():
        if 'ALL' in values:
            values = list(df[key].unique())

        df = df[df[key].isin(values)]

    df['patient_type'] = df['patient_type'].map({'covid-control': 0, 'covid-persistente': 1}).astype(int)
    return df


app_logger.info("Initialization of a COPERIA's experiment...")
ROOT_PATH = Path(__file__).parent

app_logger.info("Loading the configurations...")

config_file = os.path.join(ROOT_PATH, "config", "exp_config.json")
config = json_file_to_dict(config_file)

config_run_experiment = config.get("run")
target_class = config.get("dataset").get("target_class")
target_data = config.get("dataset").get("target_data")
target_label = config.get("dataset").get("target_label")

test_size = config.get("run").get("test_size")
k_fold = config.get("run").get("k_folds")
seed = config.get("run").get("seed")

dataset = LocalDataset(name="COPERIA-DATASET")

metadata_path = os.path.join(ROOT_PATH, "data", "coperia_metadata.csv")
dataset.load_metadata_from_csv(metadata_path, decimal=",")
dataset.transform_metadata([make_dicoperia_metadata])

app_logger.info("Making the subsets...")

dataset_ready_with_1_fold = dataset.make_1_fold_subsets(target_class_for_fold=target_class,
                                                        target_data_for_fold=target_data,
                                                        target_label_for_fold=target_label,
                                                        test_size=test_size,
                                                        seed=seed)

dataset_ready_with_3_fold = dataset.make_k_fold_subsets(target_class_for_fold=target_class,
                                                        target_data_for_fold=target_data,
                                                        target_label_for_fold=target_label,
                                                        k_fold=k_fold,
                                                        seed=seed)
