import multiprocessing
import os
import shutil

import librosa
import pytest
import pandas as pd
from pathlib import Path

import src.features.feature_extractor
import test
from src.features.feature_extractor import extract_wav

import numpy as np
import soundfile as sf


class GeneralFeatureExtraction:
    @classmethod
    def setup_class(cls):
        cls.str_path_temp_folder = os.path.join(test.ROOT_PATH, "test", "temp_folder")
        os.makedirs(cls.str_path_temp_folder, exist_ok=True)

        codec = 'PCM_24'
        sample_rate = 16000

        cls.path_to_dummy_valid_signal = os.path.join(cls.str_path_temp_folder, "valid_dummy_wav.wav")
        valid_dummy_signal = np.random.uniform(-1, 1, size=(sample_rate * 10, 2))
        sf.write(cls.path_to_dummy_valid_signal, valid_dummy_signal, sample_rate, subtype=codec)
        cls.valid_dummy_raw_data = librosa.load(cls.path_to_dummy_valid_signal, sr=None, mono=True)

        cls.path_to_dummy_empty_signal = os.path.join(cls.str_path_temp_folder, "empty_dummy_wav.wav")
        empty_dummy_signal = np.zeros((sample_rate * 10, 2))
        sf.write(cls.path_to_dummy_empty_signal, empty_dummy_signal, sample_rate, subtype=codec)
        cls.empty_dummy_raw_data = librosa.load(cls.path_to_dummy_empty_signal, sr=None, mono=True)

        cls.path_to_dummy_invalid_signal = os.path.join(cls.str_path_temp_folder, "invalid_dummy_wav.wav")
        with open(cls.path_to_dummy_invalid_signal, 'w') as f:
            f.write("This is not a valid wav file")

    @classmethod
    def teardown_class(cls):
        # Remove the temporary files after testing
        shutil.rmtree(cls.str_path_temp_folder)


class TestWavReader(GeneralFeatureExtraction):
    def test_valid_read_of_a_wav_file(self):
        # Arrange
        path_to_dummy_valid_signal = self.path_to_dummy_valid_signal
        valid_dummy_raw_data = self.valid_dummy_raw_data

        # Act
        result_as_dict = extract_wav(path_to_dummy_valid_signal)

        # Assert
        assert result_as_dict[path_to_dummy_valid_signal].shape == valid_dummy_raw_data[0].shape
        assert list(result_as_dict.keys())[0] == path_to_dummy_valid_signal

    def test_empty_read_of_a_wav_file(self):
        # Arrange
        path_to_dummy_empty_signal = self.path_to_dummy_empty_signal

        # Act & Assert
        with pytest.raises(ValueError):
            extract_wav(path_to_dummy_empty_signal)

    def test_invalid_read_of_a_wav_file(self):
        # Arrange
        path_to_dummy_invalid_signal = self.path_to_dummy_invalid_signal

        # Act & Assert
        with pytest.raises(ValueError):
            extract_wav(path_to_dummy_invalid_signal)


class TestMultiProcessor(GeneralFeatureExtraction):
    @pytest.mark.parametrize("num_cores", [1, 2, 4, multiprocessing.cpu_count() - 1])
    def test_valid_initialization_of_multiprocessor(self, num_cores):
        # Act
        multi_processor = src.features.feature_extractor.MultiProcessor(num_cores)

        # Assert
        assert multi_processor.num_cores == num_cores

    @pytest.mark.parametrize("num_cores", [0, -1, -2, multiprocessing.cpu_count() * 2, "string"])
    def test_invalid_initialization_of_multiprocessor_because_number_of_cores(self, num_cores):
        # Act & Assert
        with pytest.raises(ValueError):
            src.features.feature_extractor.MultiProcessor(0)

    def test_invalid_initialization_of_multiprocessor_because_bad_parameters(self):
        # Arrange
        multi_processor = src.features.feature_extractor.MultiProcessor(2)

        # Act & Assert
        with pytest.raises(TypeError):
            multi_processor.process_with_multiprocessing("string",
                                                         extract_wav)

        with pytest.raises(TypeError):
            multi_processor.process_with_multiprocessing([self.path_to_dummy_valid_signal],
                                                         "string")

        with pytest.raises(TypeError):
            multi_processor.process_with_multiprocessing({},
                                                         extract_wav)

        with pytest.raises(ValueError):
            multi_processor.process_with_multiprocessing([],
                                                         extract_wav)

    @pytest.mark.parametrize("num_of_raw_data", [1, 10, 20, 100])
    def test_valid_process_with_progress(self, num_of_raw_data):
        # Arrange
        raw_data_paths = [self.path_to_dummy_valid_signal for _ in range(num_of_raw_data)]
        multi_processor = src.features.feature_extractor.MultiProcessor(num_cores=multiprocessing.cpu_count() - 1)

        # Act
        result = multi_processor.process_with_multiprocessing(raw_data_paths, extract_wav)

        # Assert
        assert len(result) == num_of_raw_data
        assert all(list(r.keys())[0] == self.path_to_dummy_valid_signal for r in result)

    def test_invalid_process_with_progress(self):
        # Arrange
        raw_data_paths_with_empty_file = [self.path_to_dummy_valid_signal,
                                          self.path_to_dummy_valid_signal,
                                          self.path_to_dummy_empty_signal,
                                          self.path_to_dummy_valid_signal, ]

        raw_data_paths_with_invalid_file = [self.path_to_dummy_valid_signal,
                                            self.path_to_dummy_valid_signal,
                                            self.path_to_dummy_invalid_signal,
                                            self.path_to_dummy_valid_signal, ]

        multi_processor = src.features.feature_extractor.MultiProcessor(num_cores=multiprocessing.cpu_count() - 1)

        # Act & Assert
        with pytest.raises(ValueError):
            multi_processor.process_with_multiprocessing(raw_data_paths_with_empty_file, extract_wav)

        with pytest.raises(ValueError):
            multi_processor.process_with_multiprocessing(raw_data_paths_with_invalid_file, extract_wav)


if __name__ == "__main__":
    # Run all tests in the module
    pytest.main()
