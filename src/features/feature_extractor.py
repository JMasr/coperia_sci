import multiprocessing
import os.path
from typing import List, Callable

import librosa
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from src.logger import app_logger


# Load audio
def extract_wav(wav_path: str) -> dict[str, ndarray]:
    """
    The code above implements SAD, a pre-emphasis filter with a coefficient of 0.97, and normalization.
    :param wav_path: Path to the audio file
    :return: Audio samples
    """
    if os.path.getsize(wav_path) <= 44:
        app_logger.error(f"File {wav_path} is too small to be a valid wav file.")
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
        app_logger.warning(f"File {wav_path} is empty.")
        raise ValueError(f"File {wav_path} is empty.")

    return {wav_path: s}


class MultiProcessor:
    def __init__(self, num_cores: int = multiprocessing.cpu_count()):
        self.num_cores = num_cores
        self._check_num_cores()

    def _check_num_cores(self) -> None:
        if not isinstance(self.num_cores, int):
            app_logger.error("The `num_cores` argument must be an integer.")
            raise TypeError("The `num_cores` argument must be an integer.")
        if self.num_cores < 1:
            app_logger.error("The `num_cores` argument must be greater than 0.")
            raise ValueError("The `num_cores` argument must be greater than 0.")
        if self.num_cores > multiprocessing.cpu_count():
            app_logger.error("The `num_cores` argument must be less than or equal to the number of available cores.")
            raise ValueError("The `num_cores` argument must be less than or equal to the number of available cores.")
        if self.num_cores == multiprocessing.cpu_count():
            app_logger.warning("The number of cores is equal to the number of available cores.")

    @staticmethod
    def _parameters_validation_for_multiprocessing(raw_data_paths: List[str], process_func: Callable):
        if not isinstance(raw_data_paths, list):
            raise TypeError("The `data_paths` argument must be a list.")
        elif not callable(process_func):
            raise TypeError("The `process_func` argument must be a callable function.")
        elif not raw_data_paths:
            raise ValueError("The `data_paths` argument must not be empty.")
        elif not all(os.path.isfile(path) for path in raw_data_paths):
            raise FileNotFoundError("One or more paths do not exist.")

    def _process_with_progress(self, raw_data_paths: List[str], process_func: Callable) -> List:
        with tqdm(total=len(raw_data_paths), desc="Processing data", unit="item") as progress_bar:
            results = thread_map(process_func, raw_data_paths, max_workers=self.num_cores)
            list_of_features_from_raw_data = list(results)
            progress_bar.update(len(raw_data_paths))

        return list_of_features_from_raw_data

    def process_with_multiprocessing(self, raw_data_paths: List[str], process_func: Callable) -> List:
        try:
            self._parameters_validation_for_multiprocessing(raw_data_paths, process_func)
            list_of_features_from_raw_data = self._process_with_progress(raw_data_paths, process_func)

        except TypeError as e:
            app_logger.error(f"An error related with the arguments occurred during multiprocessing: {str(e)}")
            raise TypeError(f"An error related with the arguments occurred during multiprocessing: {str(e)}")
        except ValueError as e:
            app_logger.error(f"An error related with the raw files occurred during multiprocessing: {str(e)}")
            raise ValueError(f"An error related with the raw files occurred during multiprocessing: {str(e)}")
        except FileNotFoundError as e:
            app_logger.error(f"An error related with the paths occurred during multiprocessing: {str(e)}")
            raise FileNotFoundError(f"An error related with the paths occurred during multiprocessing: {str(e)}")
        except RuntimeError as e:
            app_logger.error(f"An error occurred during multiprocessing: {str(e)}")
            raise RuntimeError(f"An error occurred during multiprocessing: {str(e)}")

        return list_of_features_from_raw_data


class FeatureExtractor:
    def __init__(self, folds_subsets: dict, resampling_rate: int = 0):
        self.folds_subsets = folds_subsets
        self.resampling_rate = resampling_rate
