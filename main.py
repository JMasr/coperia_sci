import argparse
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

    arguments = argparse.ArgumentParser(description="Run the COPERIA pipeline")
    arguments.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(ROOT_PATH, "config", "exp_config.json"),
        help="Path to the configuration file",
    )
    args = arguments.parse_args()
    config_file = Path(args.config_file)

    pipeline_app = Pipeline(
        name="COPERIA-Experiment-Pipeline",
        root_path=ROOT_PATH,
        config_file_path=config_file,
    )
    pipeline_app.run_pipeline_for_all_model_and_feats(make_dicoperia_metadata)
