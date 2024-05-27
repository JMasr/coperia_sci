import multiprocessing
import os
import shutil

import librosa
import numpy as np
import pytest
import soundfile as sf

import test
from src.features.feature_extractor import MultiProcessor


class GeneralFeatureExtraction:
    @classmethod
    def setup_class(cls):
        cls.str_path_temp_folder = os.path.join(test.ROOT_PATH, "test", "temp_folder")
        os.makedirs(cls.str_path_temp_folder, exist_ok=True)

        cls.codec = "PCM_24"
        cls.sample_rate = 16000

        cls.path_to_dummy_valid_signal = os.path.join(
            cls.str_path_temp_folder, "valid_dummy_wav.wav"
        )
        valid_dummy_signal = np.random.uniform(-1, 1, size=(cls.sample_rate * 10, 2))
        sf.write(
            cls.path_to_dummy_valid_signal,
            valid_dummy_signal,
            cls.sample_rate,
            subtype=cls.codec,
        )
        cls.valid_dummy_raw_data = librosa.load(
            cls.path_to_dummy_valid_signal, sr=None, mono=True
        )

        cls.path_to_dummy_empty_signal = os.path.join(
            cls.str_path_temp_folder, "empty_dummy_wav.wav"
        )
        empty_dummy_signal = np.zeros((cls.sample_rate * 10, 2))
        sf.write(
            cls.path_to_dummy_empty_signal,
            empty_dummy_signal,
            cls.sample_rate,
            subtype=cls.codec,
        )
        cls.empty_dummy_raw_data = librosa.load(
            cls.path_to_dummy_empty_signal, sr=None, mono=True
        )

        cls.path_to_dummy_invalid_signal = os.path.join(
            cls.str_path_temp_folder, "invalid_dummy_wav.wav"
        )
        with open(cls.path_to_dummy_invalid_signal, "w") as f:
            f.write("This is not a valid wav file")

    @classmethod
    def teardown_class(cls):
        # Remove the temporary files after testing
        shutil.rmtree(cls.str_path_temp_folder)


def extract_wav(wav_path: str) -> dict[str, np.ndarray]:
    """
    The code above implements SAD, a pre-emphasis filter with a coefficient of 0.97, and normalization.
    :param wav_path: Path to the audio file
    :return: Audio samples
    """
    if os.path.getsize(wav_path) <= 44:
        raise ValueError(f"File {wav_path} is too small to be a valid wav file.")

    top_db: int = 30
    resample_rate: int = 44000
    pre_emphasis_coefficient: float = 0.97

    try:
        # load the audio file
        s, sr = librosa.load(wav_path, sr=None, mono=True)

        # resample the audio file
        if (sr != resample_rate) and (0 < resample_rate < sr):
            s = librosa.resample(y=s, orig_sr=sr, target_sr=resample_rate)

        # apply speech activity detection
        speech_indices = librosa.effects.split(s, top_db=top_db)
        s = np.concatenate([s[start:end] for start, end in speech_indices])

        # apply a pre-emphasis filter
        s = librosa.effects.preemphasis(s, coef=pre_emphasis_coefficient)

        # normalize
        s /= np.max(np.abs(s))
    except Exception as e:
        raise RuntimeError(f"An error occurred during feature extraction: {str(e)}")

    # Check if the audio file is empty
    if not s.size or not np.any(s) or np.nan_to_num(s).sum() == 0:
        raise ValueError(f"File {wav_path} is empty.")

    return {wav_path: s}


class TestMultiProcessor(GeneralFeatureExtraction):
    @pytest.mark.parametrize("num_cores", [1, 2, 4, multiprocessing.cpu_count() - 1])
    def test_valid_initialization_of_multiprocessor(self, num_cores):
        # Act
        multi_processor = MultiProcessor(num_cores=num_cores)

        # Assert
        assert multi_processor.num_cores == num_cores

    @pytest.mark.parametrize(
        "num_cores", [0, -1, -2, multiprocessing.cpu_count() * 2, "string"]
    )
    def test_invalid_initialization_of_multiprocessor_because_number_of_cores(
            self, num_cores
    ):
        # Act & Assert
        with pytest.raises(ValueError):
            MultiProcessor(num_cores=num_cores)

    def test_invalid_initialization_of_multiprocessor_because_bad_parameters(self):
        # Arrange
        multi_processor = MultiProcessor(num_cores=1)

        # Act & Assert
        with pytest.raises(TypeError):
            multi_processor.process_with_multiprocessing("string", extract_wav)

        with pytest.raises(TypeError):
            multi_processor.process_with_multiprocessing(
                [self.path_to_dummy_valid_signal], "string"
            )

        with pytest.raises(TypeError):
            multi_processor.process_with_multiprocessing({}, extract_wav)

        with pytest.raises(ValueError):
            multi_processor.process_with_multiprocessing([], extract_wav)

    @pytest.mark.parametrize("num_of_raw_data", [1, 10, 20, 100])
    def test_valid_process_with_progress(self, num_of_raw_data):
        # Arrange
        raw_data_paths = []
        for i in range(num_of_raw_data):
            path_to_dummy_valid_signal = os.path.join(
                self.str_path_temp_folder, f"valid_dummy_wav_{i}.wav"
            )
            valid_dummy_signal = np.random.uniform(
                -1, 1, size=(self.sample_rate * 10, 2)
            )
            sf.write(
                path_to_dummy_valid_signal,
                valid_dummy_signal,
                self.sample_rate,
                subtype=self.codec,
            )
            raw_data_paths.append(path_to_dummy_valid_signal)

        multi_processor = MultiProcessor(num_cores=multiprocessing.cpu_count())

        # Act
        result = multi_processor.process_with_multiprocessing(
            raw_data_paths, extract_wav
        )

        # Assert
        assert len(result) == num_of_raw_data
        assert all(isinstance(value, np.ndarray) for value in result.values())
        assert list(result.keys()) == raw_data_paths

    def test_invalid_process_with_progress(self):
        # Arrange
        raw_data_paths_with_empty_file = [
            self.path_to_dummy_valid_signal,
            self.path_to_dummy_valid_signal,
            self.path_to_dummy_empty_signal,
            self.path_to_dummy_valid_signal,
        ]

        raw_data_paths_with_invalid_file = [
            self.path_to_dummy_valid_signal,
            self.path_to_dummy_valid_signal,
            self.path_to_dummy_invalid_signal,
            self.path_to_dummy_valid_signal,
        ]

        multi_processor = MultiProcessor(num_cores=multiprocessing.cpu_count() - 1)

        # Act & Assert
        with pytest.raises(ValueError):
            multi_processor.process_with_multiprocessing(
                raw_data_paths_with_empty_file, extract_wav
            )

        with pytest.raises(ValueError):
            multi_processor.process_with_multiprocessing(
                raw_data_paths_with_invalid_file, extract_wav
            )


if __name__ == "__main__":
    # Run all tests in the module
    pytest.main()
