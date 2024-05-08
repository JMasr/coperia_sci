import pandas as pd

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
    print('=== Filtering the metadata... ===')
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


app_logger.info("Loading the COPERIA dataset...")
METADATA_PATH = r"C:\Users\jmram\OneDrive\Documents\GitHub\exp_coperia\scientificProject\data\coperia_metadata.csv"

dataset = LocalDataset(name="COPERIA-DATASET")
dataset.load_metadata_from_csv(METADATA_PATH, decimal=",")
dataset.transform_metadata([make_dicoperia_metadata])

app_logger.info("Making the subsets...")
target_class = 'patient_id'
target_data = 'audio_id'
target_label = 'patient_type'
test_size = 0.2
k_fold = 3
seed = 42
