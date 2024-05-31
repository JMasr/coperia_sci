import os.path
from pathlib import Path

import pandas as pd

from src.dataset.basic_dataset import AudioDataset
from src.files import json_file_to_dict
from src.logger import app_logger


def make_dicoperia_metadata(
        metadata: pd.DataFrame, filters_: dict = None, remove_samples: dict = None
):
    """
    Make a metadata file for the COPERIA dataset filtering some columns
    :param metadata: a list with all the audio samples in COPERIA as an Audio class
    :param filters_: a dictionary with the columns and values to keep
    :param remove_samples: a dictionary with the columns and values to remove
    :return: a pandas dataframe with the metadata of the DICOPERIA dataset
    """
    df = metadata.copy()

    if filters_ is None:
        filters_ = {"patient_type": ["covid-control", "covid-persistente"]}

    if remove_samples is None:
        remove_samples = {
            "audio_id": [
                "c15e54fc-5290-4652-a3f7-ff3b779bd980",
                "244b61cc-4fd7-4073-b0d8-7bacd42f6202",
            ],
            "patient_id": ["coperia-rehab"],
        }

    for key, values in remove_samples.items():
        df = df[~df[key].isin(values)]

    for key, values in filters_.items():
        if "ALL" in values:
            values = list(df[key].unique())

        df = df[df[key].isin(values)]

    df["patient_type"] = (
        df["patient_type"].map({"covid-control": 0, "covid-persistente": 1}).astype(int)
    )
    return df


if __name__ == "__main__":
    app_logger.info("Pipeline - Initialization of a COPERIA's experiment")
    ROOT_PATH = Path(__file__).parent

    app_logger.info("Pipeline - Loading the configurations")

    config_file = os.path.join(ROOT_PATH, "config", "exp_config.json")
    config = json_file_to_dict(config_file)
    config_audio = config.get("audio")
    config_run_experiment = config.get("run")
    test_size = config.get("run").get("test_size")
    k_fold = config.get("run").get("k_folds")
    seed = config.get("run").get("seed")

    config_dataset_experiment = config.get("dataset")
    dataset_raw_data_path = config_dataset_experiment.get("raw_data_path")
    target_class = config_dataset_experiment.get("target_class")
    target_data = config_dataset_experiment.get("target_data")
    target_label = config_dataset_experiment.get("target_label")
    metadata_path = config_dataset_experiment.get("path_to_csv")
    object_path = config_dataset_experiment.get("path_to_object", False)

    if object_path:
        dataset = AudioDataset(
            name="COPERIA_DATASET",
            storage_path=os.path.join(ROOT_PATH, "data"),
            config_audio=config_audio,
        ).load_dataset_from_a_serialized_object(object_path)
    else:
        dataset = AudioDataset(
            name="COPERIA_DATASET",
            storage_path=os.path.join(ROOT_PATH, "data"),
            config_audio=config_audio,
        )

        dataset.load_metadata_from_csv(metadata_path, decimal=",")
        dataset.transform_metadata([make_dicoperia_metadata])
        dataset.transform_column_id_2_data_path(
            column_name="audio_id", path=dataset_raw_data_path, extension=".wav"
        )

        app_logger.info("Pipeline - Making the subsets")
        dataset.make_k_fold_subsets(
            target_class_for_fold=target_class, k_fold=k_fold, seed=seed
        )

        dataset.load_raw_data()
        dataset.extract_all_acoustic_features_supported()
        dataset.save_dataset_as_a_serialized_object()
        app_logger.info("Pipeline - Dataset saved with all the feats")

    # models_names = SUPPORTED_MODELS.keys()
    # for model_name in models_names:
    #     model_builder = ModelBuilder(
    #         name=model_name,
    #         path_to_model=os.path.join(ROOT_PATH, "models"),
    #     )
    #     model = model_builder.build_model()
