import os.path
from pathlib import Path

import pandas as pd

from experiments.basic_exp import BasicExperiment
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
        df["patient_type"]
        .map({"covid-control": True, "covid-persistente": False})
        .astype(bool)
    )
    return df


if __name__ == "__main__":
    app_logger.info("Pipeline - Initialization of a COPERIA's experiment")
    ROOT_PATH = Path(__file__).parent

    app_logger.info("Pipeline - Loading the configurations")

    config_file = os.path.join(ROOT_PATH, "config", "exp_config.json")
    config = json_file_to_dict(config_file)

    config_audio = config.get("audio")
    feat_name = config_audio.get("feature_type")


    config_run_experiment = config.get("run")
    seed = config.get("run").get("seed")
    k_fold = config.get("run").get("k_folds")
    test_size = config.get("run").get("test_size")
    run_name = config_run_experiment.get("run_name")
    path_to_save_experiment = config_run_experiment.get("path_to_save_experiment")

    config_dataset_experiment = config.get("dataset")
    dataset_name = config_dataset_experiment.get("name")
    target_class = config_dataset_experiment.get("target_class")
    target_data = config_dataset_experiment.get("target_data")
    target_label = config_dataset_experiment.get("target_label")
    metadata_path = config_dataset_experiment.get("path_to_csv")
    object_path = config_dataset_experiment.get("path_to_object", False)
    dataset_raw_data_path = config_dataset_experiment.get("raw_data_path")

    config_model_experiment = config.get("model")
    model_name = config_model_experiment.get("name")
    model_parameters = config_model_experiment.get("parameters")
    model_parameters["random_state"] = seed

    if os.path.exists(object_path):
        dataset = AudioDataset(
            name=dataset_name,
            storage_path=os.path.dirname(object_path),
            config_audio=config_audio,
        ).load_dataset_from_a_serialized_object(object_path)
        app_logger.info("Pipeline - Dataset loaded.")
    else:
        dataset = AudioDataset(
            name=dataset_name,
            storage_path=os.path.dirname(object_path),
            config_audio=config_audio,
        )

        dataset.load_metadata_from_csv(metadata_path, decimal=",")
        dataset.transform_metadata([make_dicoperia_metadata])
        dataset.transform_column_id_2_data_path(
            column_name="audio_id", path=dataset_raw_data_path, extension=".wav"
        )

        dataset.load_raw_data()
        dataset.extract_acoustic_features(feat_name=feat_name)
        dataset.save_dataset_as_a_serialized_object()
        app_logger.info(f"Pipeline - Dataset saved with {feat_name} features.")

        train_folds, test_folds = dataset.get_k_audio_subsets(
            target_class_for_fold=target_class,
            target_label_for_fold=target_label,
            acoustics_feat_name=feat_name,
            seed=seed,
            k_folds=k_fold,
            test_size=test_size,
        )
        app_logger.info(f"Pipeline - {k_fold} Folds created")

        experiment = BasicExperiment(
            name=run_name,
            seed=seed,
            description=f"{dataset_name} experiment with {k_fold} folds",
            path_to_save_experiment=path_to_save_experiment,
            folds_train=train_folds,
            folds_test=test_folds,
            feature_name=feat_name,
            name_model=model_name,
            parameters_model=model_parameters,
        )
        experiment.save_as_a_serialized_object()
        app_logger.info(f"Pipeline - Experiment saved on {path_to_save_experiment}.")

        model = experiment.training_phase()
