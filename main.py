import os.path
from pathlib import Path

import pandas as pd

from src.experiments.pipeline import Pipeline


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
    ROOT_PATH = Path(__file__).parent
    config_file = Path(os.path.join(ROOT_PATH, "config", "exp_config.json"))

    pipeline_app = Pipeline(
        name="COPERIA-Experiment-Pipeline",
        root_path=ROOT_PATH,
        config_file_path=config_file,
    )

    pipeline_app.run_pipeline(make_dicoperia_metadata)

    # config = json_file_to_dict(config_file)
    #
    # config_audio = config.get("audio")
    # feat_name = config_audio.get("feature_type")
    #
    # config_run_experiment = config.get("run")
    # seed = config.get("run").get("seed")
    # debug = config.get("run").get("debug")
    # k_fold = config.get("run").get("k_folds")
    # test_size = config.get("run").get("test_size")
    # run_name = config_run_experiment.get("run_name")
    # path_to_save_experiment = config_run_experiment.get("path_to_save_experiment")
    #
    # config_dataset_experiment = config.get("dataset")
    # dataset_name = config_dataset_experiment.get("name")
    # target_class = config_dataset_experiment.get("target_class")
    # target_data = config_dataset_experiment.get("target_data")
    # target_label = config_dataset_experiment.get("target_label")
    # metadata_path = config_dataset_experiment.get("path_to_csv")
    # dataset_object_path = config_dataset_experiment.get("path_to_object", False)
    # dataset_raw_data_path = config_dataset_experiment.get("raw_data_path")
    # filters = config_dataset_experiment.get("filters")
    #
    # config_model_experiment = config.get("model")
    # model_name = config_model_experiment.get("name")
    # model_parameters = config_model_experiment.get("parameters")
    # model_parameters["random_state"] = seed
    #
    # if os.path.exists(dataset_object_path):
    #     dataset = AudioDataset(
    #         name=dataset_name,
    #         storage_path=os.path.dirname(dataset_object_path),
    #         filters=filters[0],
    #         config_audio=config_audio,
    #     ).load_dataset_from_a_serialized_object(dataset_object_path)
    #     app_logger.info("Pipeline - Dataset loaded.")
    # else:
    #     dataset = AudioDataset(
    #         name=dataset_name,
    #         filters=filters[0],
    #         storage_path=os.path.dirname(dataset_object_path),
    #         config_audio=config_audio,
    #     )
    #
    #     dataset.load_metadata_from_csv(metadata_path, decimal=",")
    #     if debug:
    #         dataset.sample_metadata(fraction=0.1)
    #
    #     dataset.transform_metadata([make_dicoperia_metadata])
    #     dataset.transform_column_id_2_data_path(
    #         column_name="audio_id", path=dataset_raw_data_path, extension=".wav"
    #     )
    #
    #     dataset.load_raw_data()
    #     dataset.extract_acoustic_features(feat_name)
    #     dataset.save_dataset_as_a_serialized_object()
    #     app_logger.info(f"Pipeline - Dataset saved with {feat_name} features.")
    #
    # for feat_name in SUPPORTED_FEATS:
    #     for model_name in DEFAULT_CONFIG.keys():
    #         model_parameters = DEFAULT_CONFIG[model_name]
    #
    #         experiment = BasicExperiment(
    #             seed=seed,
    #             name=run_name,
    #             dataset=dataset,
    #             k_fold=k_fold,
    #             test_size=test_size,
    #             feature_name=feat_name,
    #             target_class=target_class,
    #             target_label=target_label,
    #             name_model=model_name,
    #             parameters_model=model_parameters,
    #             path_to_save_experiment=path_to_save_experiment,
    #         )
    #
    #         experiment.run_experiment()
    #         experiment.record_experiment()
    #         del experiment
    #         sleep(5)
