import os
import shutil
from pathlib import Path

import pytest

from src.dataset.basic_dataset import AudioDataset
from src.experiments.basic_exp import BasicExperiment
from src.logger import BasicLogger
from test import ROOT_PATH
from test.test_dataset_should import mock_a_dataframe_with_metadata_and_audio


class TestExperimentShould:
    @classmethod
    def setup_class(cls):
        # Create a temporary resources for testing
        cls.str_path_temp_folder = os.path.join(ROOT_PATH, "test", "temp_folder")
        os.makedirs(cls.str_path_temp_folder, exist_ok=True)

        # Create a logger
        cls.app_logger = BasicLogger(
            log_file=os.path.join(cls.str_path_temp_folder, "logs", "experiment.log")
        ).get_logger()

        # Model parameters
        cls.model_name = "LogisticRegression"
        cls.model_parameters = {
            "C": 0.01,
            "max_iter": 40,
            "penalty": "l2",
            "random_state": 42,
            "solver": "sag",
            "multi_class": "auto",
            "class_weight": "balanced",
            "n_jobs": -1,  # Use all processors
            "verbose": True,
        }
        cls.metrics = [
            "acc_score",
            "tn",
            "fp",
            "fn",
            "tp",
            "sensitivity",
            "specificity",
            "optimal_threshold",
            "auc_score",
            "confusion_matrix",
            "f1_scr",
            "true_permutation_score",
            "p_value",
        ]

        # Create a mock dataframe with metadata and audio
        cls.num_rows = 50
        cls.sample_rate = 16000
        cls.mock_dataframe = mock_a_dataframe_with_metadata_and_audio(
            cls.str_path_temp_folder, cls.sample_rate, cls.num_rows
        )

        cls.temp_file = Path(os.path.join(cls.str_path_temp_folder, "temp_file.csv"))
        cls.mock_dataframe.to_csv(cls.temp_file, index=False, sep=",")

        # Create defaults configurations for audio processing
        cls.run_name = "TEST-RUN"
        cls.seed = 42
        cls.k_fold = 2
        cls.test_size = 0.2

        cls.dataset_name = "TEST-DATASET"
        cls.id_column = "id"
        cls.target_class = "data"
        cls.target_label = "label"
        cls.config_audio = {
            "feature_type": "compare_2016_energy",
            "top_db": 30,
            "pre_emphasis_coefficient": 0.97,
            "resampling_rate": 44100,
            "n_mels": 64,
            "n_mfcc": 32,
            "plp_order": 13,
            "conversion_approach": "Wang",
            "f_max": 22050,
            "f_min": 100,
            "window_size": 25.0,
            "hop_length": 10.0,
            "window_type": "hamming",
            "normalize": "mvn",
            "use_energy": True,
            "apply_mean_norm": True,
            "apply_vari_norm": True,
            "compute_deltas_feats": True,
            "compute_deltas_deltas_feats": True,
            "compute_opensmile_extra_features": True,
        }
        cls.feature_name = cls.config_audio["feature_type"]
        cls.filter_testing = {"audio_type": ["ALL"], "audio_moment": ["ALL"]}

        cls.dataset = AudioDataset(
            name=cls.dataset_name,
            storage_path=cls.str_path_temp_folder,
            app_logger=cls.app_logger,
            column_with_ids=cls.id_column,
            column_with_target_class=cls.target_class,
            column_with_label_of_class=cls.target_label,
            filters=cls.filter_testing,
            config_audio=cls.config_audio,
            dataset_raw_data_path=cls.str_path_temp_folder,
        )
        cls.dataset.load_metadata_from_csv(str(cls.temp_file))
        cls.dataset.load_raw_data()
        cls.dataset.extract_acoustic_features(cls.feature_name)

    @classmethod
    def teardown_class(cls):
        # Remove the temporary files after testing
        shutil.rmtree(cls.str_path_temp_folder)

    def test_valid_initialization_should(self):
        # Arrange & Act
        experiment = BasicExperiment(
            seed=self.seed,
            name=self.run_name,
            dataset=self.dataset,
            k_fold=self.k_fold,
            test_size=self.test_size,
            feature_name=self.feature_name,
            target_class=self.target_class,
            target_label=self.target_label,
            name_model=self.model_name,
            parameters_model=self.model_parameters,
            path_to_save_experiment=self.str_path_temp_folder,
            app_logger=self.app_logger,
        )

        # Assert
        assert isinstance(experiment, BasicExperiment)

    def test_valid_run_should(self):
        # Arrange
        experiment = BasicExperiment(
            seed=self.seed,
            name=self.run_name,
            dataset=self.dataset,
            k_fold=self.k_fold,
            test_size=self.test_size,
            feature_name=self.feature_name,
            target_class=self.target_class,
            target_label=self.target_label,
            name_model=self.model_name,
            parameters_model=self.model_parameters,
            path_to_save_experiment=self.str_path_temp_folder,
            app_logger=self.app_logger,
        )

        # Act
        model_performance = experiment.run_experiment()

        # Assert
        assert model_performance is not None
        assert isinstance(model_performance, dict)
        assert len(model_performance) == self.k_fold
        for k in model_performance.keys():
            assert isinstance(model_performance[k], dict)
            assert all([metric in model_performance[k] for metric in self.metrics])

    @pytest.mark.parametrize(
        "feat_name",
        [
            "compare_2016_energy",
            "compare_2016_llds",
            "compare_2016_voicing",
        ],
    )
    def test_valid_run_all_supported_feats_should(self, feat_name: str):
        # Arrange
        experiment = BasicExperiment(
            seed=self.seed,
            name=self.run_name,
            dataset=self.dataset,
            k_fold=self.k_fold,
            test_size=self.test_size,
            feature_name=feat_name,
            target_class=self.target_class,
            target_label=self.target_label,
            name_model=self.model_name,
            parameters_model=self.model_parameters,
            path_to_save_experiment=self.str_path_temp_folder,
            app_logger=self.app_logger,
        )

        # Act
        model_performance = experiment.run_experiment()

        # Assert
        metrics = [
            "acc_score",
            "tn",
            "fp",
            "fn",
            "tp",
            "sensitivity",
            "specificity",
            "optimal_threshold",
            "auc_score",
            "confusion_matrix",
            "f1_scr",
            "true_permutation_score",
            "p_value",
        ]

        assert model_performance is not None
        assert isinstance(model_performance, dict)
        assert len(model_performance) == self.k_fold
        for k in model_performance.keys():
            assert isinstance(model_performance[k], dict)
            assert all([metric in model_performance[k] for metric in metrics])

    @pytest.mark.parametrize(
        "feat_name",
        [
            "compare_2016_energy",
            "compare_2016_llds",
            "compare_2016_voicing",
        ],
    )
    def test_valid_run_all_supported_feats_should(self, feat_name: str):
        # Arrange
        experiment = BasicExperiment(
            seed=self.seed,
            name=self.run_name,
            dataset=self.dataset,
            k_fold=self.k_fold,
            test_size=self.test_size,
            feature_name=feat_name,
            target_class=self.target_class,
            target_label=self.target_label,
            name_model=self.model_name,
            parameters_model=self.model_parameters,
            path_to_save_experiment=self.str_path_temp_folder,
            app_logger=self.app_logger,
        )

        # Act
        model_performance = experiment.run_experiment()

        # Assert
        assert model_performance is not None
        assert isinstance(model_performance, dict)
        assert len(model_performance) == self.k_fold
        for k in model_performance.keys():
            assert isinstance(model_performance[k], dict)
            assert all([metric in model_performance[k] for metric in self.metrics])

    @pytest.mark.parametrize(
        "model_name",
        [
            "LogisticRegression",
            "RandomForest",
            "LinearSVM",
        ],
    )
    def test_valid_run_all_supported_models_should(self, model_name: str):
        # Arrange
        experiment = BasicExperiment(
            seed=self.seed,
            name=self.run_name,
            dataset=self.dataset,
            k_fold=self.k_fold,
            test_size=self.test_size,
            feature_name=self.feature_name,
            target_class=self.target_class,
            target_label=self.target_label,
            name_model=model_name,
            parameters_model=self.model_parameters,
            path_to_save_experiment=self.str_path_temp_folder,
            app_logger=self.app_logger,
        )

        # Act
        model_performance = experiment.run_experiment()

        # Assert
        assert model_performance is not None
        assert isinstance(model_performance, dict)
        assert len(model_performance) == self.k_fold
        for k in model_performance.keys():
            assert isinstance(model_performance[k], dict)
            assert all([metric in model_performance[k] for metric in self.metrics])


if __name__ == "__main__":
    # Run all tests in the module
    pytest.main()
