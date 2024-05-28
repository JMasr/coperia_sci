import concurrent.futures
import multiprocessing
import os.path
from typing import List, Callable, Dict

import librosa
import numpy as np
import opensmile
import pandas as pd
import torch
import torchaudio
from numpy import ndarray
from spafe.features.bfcc import bfcc
from spafe.features.cqcc import cqcc
from spafe.features.gfcc import gfcc
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpc
from spafe.features.lpc import lpcc
from spafe.features.mfcc import mfcc, imfcc
from spafe.features.msrcc import msrcc
from spafe.features.ngcc import ngcc
from spafe.features.pncc import pncc
from spafe.features.psrcc import psrcc
from spafe.features.rplp import plp, rplp
from tqdm import tqdm

from src.logger import app_logger


class MultiProcessor:
    def __init__(self, num_cores: int = multiprocessing.cpu_count()):
        self.num_cores = num_cores

        if not isinstance(self.num_cores, int):
            raise ValueError("The `num_cores` argument must be an integer.")
        elif self.num_cores < 1 or self.num_cores > multiprocessing.cpu_count():
            raise ValueError(
                "The `num_cores` argument must be between 1 and the number of CPUs."
            )

    @staticmethod
    def _parameters_validation_for_multiprocessing(
            raw_data_paths: List[str], process_func: Callable
    ):
        if not isinstance(raw_data_paths, list):
            raise TypeError("The `raw_data_paths` argument must be a list.")
        elif not callable(process_func):
            raise TypeError("The `process_func` argument must be a callable function.")
        elif not raw_data_paths:
            raise ValueError("The `raw_data_paths` argument must not be empty.")
        elif not all(os.path.isfile(path) for path in raw_data_paths):
            raise FileNotFoundError("One or more paths do not exist.")

    def _process_with_progress(
            self, raw_data_paths: List[str], process_func: Callable
    ) -> Dict[str, np.ndarray]:
        results = {}
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_cores
        ) as executor:
            future_to_data = {
                executor.submit(process_func, path): path for path in raw_data_paths
            }
            for future in tqdm(
                    concurrent.futures.as_completed(future_to_data),
                    total=len(future_to_data),
                    desc="Processing data",
                    unit="item",
            ):
                path = future_to_data[future]
                try:
                    result = future.result()
                    results.update(result)
                except Exception as e:
                    app_logger.error(f"An error occurred during processing {path}: {e}")
                    raise RuntimeError(
                        f"An error occurred during processing {path}: {e}"
                    )
        return results

    def process_with_multiprocessing(
            self, raw_data_paths: List[str], process_func: Callable
    ) -> Dict[str, np.ndarray]:
        try:
            self._parameters_validation_for_multiprocessing(
                raw_data_paths, process_func
            )
            dict_with_id_and_features_from_raw_data = self._process_with_progress(
                raw_data_paths, process_func
            )
            return dict_with_id_and_features_from_raw_data

        except (TypeError, ValueError, FileNotFoundError, RuntimeError) as e:
            app_logger.error(f"An error occurred during multiprocessing: {e}")
            raise


class AudioProcessor:
    def __init__(self, arguments: dict):
        self.arguments = arguments

        self.supported_feats: list = [
            "MFCC",
            "MelSpec",
            "logMelSpec",
            "ComParE_2016_energy",
            "ComParE_2016_voicing",
            "ComParE_2016_spectral",
            "ComParE_2016_basic_spectral",
            "ComParE_2016_mfcc",
            "ComParE_2016_rasta",
            "ComParE_2016_llds",
            "Spafe_mfcc",
            "Spafe_imfcc",
            "Spafe_cqcc",
            "Spafe_gfcc",
            "Spafe_lfcc",
            "Spafe_lpc",
            "Spafe_lpcc",
            "Spafe_msrcc",
            "Spafe_ngcc",
            "Spafe_pncc",
            "Spafe_psrcc",
            "Spafe_plp",
            "Spafe_rplp",
        ]

        try:
            self.feature_type = self.arguments["feature_type"]
            self.resampling_rate = int(self.arguments["resampling_rate"])
            self.top_db = float(self.arguments["top_db"])
            self.resampling_rate = int(self.arguments["resampling_rate"])
            self.pre_emphasis_coefficient = float(
                self.arguments["pre_emphasis_coefficient"]
            )

            self.f_min = int(self.arguments["f_min"])
            self.f_max = int(self.arguments["f_max"])
            self.window_size = int(self.arguments["window_size"])
            self.hop_length = int(self.arguments["hop_length"])
            self.nfft = int(float(self.window_size) * 1e-3 * self.resampling_rate)
            self.hop_length = int(float(self.hop_length) * 1e-3 * self.resampling_rate)

            self.n_mels = int(self.arguments["n_mels"])
            self.n_mfcc = int(self.arguments["n_mfcc"])

            self.plp_order = int(self.arguments["plp_order"])
            self.conversion_approach = self.arguments["conversion_approach"]

            self.normalize = self.arguments["normalize"]
            self.use_energy = bool(self.arguments["use_energy"])
            self.apply_mean_norm = bool(self.arguments["apply_mean_norm"])
            self.apply_vari_norm = bool(self.arguments["apply_vari_norm"])

            self.compute_deltas = bool(self.arguments["compute_deltas_feats"])
            self.compute_deltas_deltas = bool(
                self.arguments["compute_deltas_deltas_feats"]
            )
            self.compute_opensmile_extra_features = bool(
                self.arguments["compute_opensmile_extra_features"]
            )

            if self.feature_type not in self.supported_feats:
                raise ValueError(f"Feature type {self.feature_type} not supported yet")
            else:
                self.feature_transform = self._create_feature_transformer()

        except Exception as e:
            app_logger.error(f"An error occurred during feature extraction: {str(e)}")
            raise RuntimeError(f"An error occurred during feature extraction: {str(e)}")

    def _create_feature_transformer(self):
        if self.feature_type == "MFCC":
            feature_transform = torchaudio.transforms.MFCC(
                sample_rate=self.resampling_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    "n_fft": self.nfft,
                    "n_mels": self.n_mels,
                    "f_max": self.f_max,
                    "hop_length": self.hop_length,
                },
            )
        elif self.feature_type in ["MelSpec", "logMelSpec"]:
            feature_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.resampling_rate,
                n_fft=self.nfft,
                n_mels=self.n_mels,
                f_max=self.f_max,
                hop_length=self.hop_length,
            )
        elif "ComParE_2016" in self.feature_type:
            feature_transform = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                sampling_rate=self.resampling_rate,
            )
        elif "Spafe_" in self.feature_type:
            spafe_feature_transformers = {
                "Spafe_mfcc": mfcc,
                "Spafe_imfcc": imfcc,
                "Spafe_bfcc": bfcc,
                "Spafe_cqcc": cqcc,
                "Spafe_gfcc": gfcc,
                "Spafe_lfcc": lfcc,
                "Spafe_lpc": lpc,
                "Spafe_lpcc": lpcc,
                "Spafe_msrcc": msrcc,
                "Spafe_ngcc": ngcc,
                "Spafe_pncc": pncc,
                "Spafe_psrcc": psrcc,
                "Spafe_plp": plp,
                "Spafe_rplp": rplp,
            }
            feature_transform = spafe_feature_transformers[self.feature_type]
        else:
            raise ValueError("Feature type not implemented")

        return feature_transform

    def _do_feature_extraction(self, s: np.ndarray, sr: int):
        """Feature preparation
        Steps:
        1. Apply feature extraction to waveform
        2. Convert amplitude to dB if required
        3. Append delta and delta-delta features
        """
        matrix_with_feats = None

        if self.feature_type == "MelSpec":
            matrix_with_feats = self.feature_transform(s)

        if self.feature_type == "logMelSpec":
            matrix_with_feats = self.feature_transform(s)
            matrix_with_feats = torchaudio.functional.amplitude_to_DB(
                matrix_with_feats, amin=1e-10, multiplier=10, db_multiplier=0
            )

        if self.feature_type == "MFCC":
            matrix_with_feats = self.feature_transform(s)

        if "ComParE_2016" in self.feature_type:
            s = s[None, :]
            matrix_with_feats = self.feature_transform.process_signal(s, sr)

            # feature subsets
            opensmile_extra_feats_set = {}
            if self.feature_type == "ComParE_2016_voicing":
                opensmile_extra_feats_set["subset"] = [
                    "F0final_sma",
                    "voicingFinalUnclipped_sma",
                    "jitterLocal_sma",
                    "jitterDDP_sma",
                    "shimmerLocal_sma",
                    "logHNR_sma",
                ]

            if self.feature_type == "ComParE_2016_energy":
                opensmile_extra_feats_set["subset"] = [
                    "audspec_lengthL1norm_sma",
                    "audspecRasta_lengthL1norm_sma",
                    "pcm_RMSenergy_sma",
                    "pcm_zcr_sma",
                ]

            if self.feature_type == "ComParE_2016_spectral":
                opensmile_extra_feats_set["subset"] = [
                    "audSpec_Rfilt_sma[0]",
                    "audSpec_Rfilt_sma[1]",
                    "audSpec_Rfilt_sma[2]",
                    "audSpec_Rfilt_sma[3]",
                    "audSpec_Rfilt_sma[4]",
                    "audSpec_Rfilt_sma[5]",
                    "audSpec_Rfilt_sma[6]",
                    "audSpec_Rfilt_sma[7]",
                    "audSpec_Rfilt_sma[8]",
                    "audSpec_Rfilt_sma[9]",
                    "audSpec_Rfilt_sma[10]",
                    "audSpec_Rfilt_sma[11]",
                    "audSpec_Rfilt_sma[12]",
                    "audSpec_Rfilt_sma[13]",
                    "audSpec_Rfilt_sma[14]",
                    "audSpec_Rfilt_sma[15]",
                    "audSpec_Rfilt_sma[16]",
                    "audSpec_Rfilt_sma[17]",
                    "audSpec_Rfilt_sma[18]",
                    "audSpec_Rfilt_sma[19]",
                    "audSpec_Rfilt_sma[20]",
                    "audSpec_Rfilt_sma[21]",
                    "audSpec_Rfilt_sma[22]",
                    "audSpec_Rfilt_sma[23]",
                    "audSpec_Rfilt_sma[24]",
                    "audSpec_Rfilt_sma[25]",
                    "pcm_fftMag_fband250-650_sma",
                    "pcm_fftMag_fband1000-4000_sma",
                    "pcm_fftMag_spectralRollOff25.0_sma",
                    "pcm_fftMag_spectralRollOff50.0_sma",
                    "pcm_fftMag_spectralRollOff75.0_sma",
                    "pcm_fftMag_spectralRollOff90.0_sma",
                    "pcm_fftMag_spectralFlux_sma",
                    "pcm_fftMag_spectralCentroid_sma",
                    "pcm_fftMag_spectralEntropy_sma",
                    "pcm_fftMag_spectralVariance_sma",
                    "pcm_fftMag_spectralSkewness_sma",
                    "pcm_fftMag_spectralKurtosis_sma",
                    "pcm_fftMag_spectralSlope_sma",
                    "pcm_fftMag_psySharpness_sma",
                    "pcm_fftMag_spectralHarmonicity_sma",
                    "mfcc_sma[1]",
                    "mfcc_sma[2]",
                    "mfcc_sma[3]",
                    "mfcc_sma[4]",
                    "mfcc_sma[5]",
                    "mfcc_sma[6]",
                    "mfcc_sma[7]",
                    "mfcc_sma[8]",
                    "mfcc_sma[9]",
                    "mfcc_sma[10]",
                    "mfcc_sma[11]",
                    "mfcc_sma[12]",
                    "mfcc_sma[13]",
                    "mfcc_sma[14]",
                ]

            if self.feature_type == "ComParE_2016_mfcc":
                opensmile_extra_feats_set["subset"] = [
                    "mfcc_sma[1]",
                    "mfcc_sma[2]",
                    "mfcc_sma[3]",
                    "mfcc_sma[4]",
                    "mfcc_sma[5]",
                    "mfcc_sma[6]",
                    "mfcc_sma[7]",
                    "mfcc_sma[8]",
                    "mfcc_sma[9]",
                    "mfcc_sma[10]",
                    "mfcc_sma[11]",
                    "mfcc_sma[12]",
                    "mfcc_sma[13]",
                    "mfcc_sma[14]",
                ]

            if self.feature_type == "ComParE_2016_rasta":
                opensmile_extra_feats_set["subset"] = [
                    "audSpec_Rfilt_sma[0]",
                    "audSpec_Rfilt_sma[1]",
                    "audSpec_Rfilt_sma[2]",
                    "audSpec_Rfilt_sma[3]",
                    "audSpec_Rfilt_sma[4]",
                    "audSpec_Rfilt_sma[5]",
                    "audSpec_Rfilt_sma[6]",
                    "audSpec_Rfilt_sma[7]",
                    "audSpec_Rfilt_sma[8]",
                    "audSpec_Rfilt_sma[9]",
                    "audSpec_Rfilt_sma[10]",
                    "audSpec_Rfilt_sma[11]",
                    "audSpec_Rfilt_sma[12]",
                    "audSpec_Rfilt_sma[13]",
                    "audSpec_Rfilt_sma[14]",
                    "audSpec_Rfilt_sma[15]",
                    "audSpec_Rfilt_sma[16]",
                    "audSpec_Rfilt_sma[17]",
                    "audSpec_Rfilt_sma[18]",
                    "audSpec_Rfilt_sma[19]",
                    "audSpec_Rfilt_sma[20]",
                    "audSpec_Rfilt_sma[21]",
                    "audSpec_Rfilt_sma[22]",
                    "audSpec_Rfilt_sma[23]",
                    "audSpec_Rfilt_sma[24]",
                    "audSpec_Rfilt_sma[25]",
                ]

            if self.feature_type == "ComParE_2016_basic_spectral":
                opensmile_extra_feats_set["subset"] = [
                    "pcm_fftMag_fband250-650_sma",
                    "pcm_fftMag_fband1000-4000_sma",
                    "pcm_fftMag_spectralRollOff25.0_sma",
                    "pcm_fftMag_spectralRollOff50.0_sma",
                    "pcm_fftMag_spectralRollOff75.0_sma",
                    "pcm_fftMag_spectralRollOff90.0_sma",
                    "pcm_fftMag_spectralFlux_sma",
                    "pcm_fftMag_spectralCentroid_sma",
                    "pcm_fftMag_spectralEntropy_sma",
                    "pcm_fftMag_spectralVariance_sma",
                    "pcm_fftMag_spectralSkewness_sma",
                    "pcm_fftMag_spectralKurtosis_sma",
                    "pcm_fftMag_spectralSlope_sma",
                    "pcm_fftMag_psySharpness_sma",
                    "pcm_fftMag_spectralHarmonicity_sma",
                ]

            if self.feature_type == "ComParE_2016_llds":
                opensmile_extra_feats_set["subset"] = list(matrix_with_feats.columns)

            matrix_with_feats = matrix_with_feats[
                opensmile_extra_feats_set["subset"]
            ].to_numpy()
            matrix_with_feats = np.nan_to_num(matrix_with_feats)
            matrix_with_feats = torch.from_numpy(matrix_with_feats).T

        if "Spafe_" in self.feature_type:
            if self.feature_type in [
                "Spafe_mfcc",
                "Spafe_imfcc",
                "Spafe_gfcc",
                "Spafe_lfcc",
                "Spafe_msrcc",
                "Spafe_ngcc",
                "Spafe_psrcc",
            ]:
                matrix_with_feats = self.feature_transform(
                    s,
                    sr,
                    num_ceps=self.n_mfcc,
                    low_freq=self.f_min,
                    high_freq=int(sr // 2),
                    nfilts=self.n_mels,
                    nfft=self.nfft,
                    use_energy=self.use_energy,
                )

            elif self.feature_type in ["Spafe_pncc"]:
                matrix_with_feats = self.feature_transform(
                    s,
                    sr,
                    nfft=self.nfft,
                    nfilts=self.n_mels,
                    low_freq=self.f_min,
                    num_ceps=self.n_mfcc,
                    high_freq=int(sr // 2),
                )

            elif self.feature_type in ["Spafe_cqcc"]:
                matrix_with_feats = self.feature_transform(
                    s,
                    sr,
                    num_ceps=self.n_mfcc,
                    low_freq=self.f_min,
                    high_freq=(sr // 2),
                    nfft=self.nfft,
                )

            elif self.feature_type in [
                "Spafe_lpc",
                "Spafe_lpcc",
            ]:
                matrix_with_feats = self.feature_transform(s, sr, order=self.plp_order)

                if isinstance(matrix_with_feats, tuple):
                    matrix_with_feats = matrix_with_feats[0]

            elif self.feature_type in ["Spafe_plp", "Spafe_rplp"]:
                matrix_with_feats = self.feature_transform(
                    s,
                    sr,
                    order=self.plp_order,
                    conversion_approach=self.conversion_approach,
                    low_freq=self.f_min,
                    high_freq=int(sr // 2),
                    normalize=self.normalize,
                    nfilts=self.n_mels,
                    nfft=self.nfft,
                )

            matrix_with_feats = np.nan_to_num(matrix_with_feats)
            matrix_with_feats = torch.from_numpy(matrix_with_feats).T

        if self.compute_deltas:
            matrix_with_feats_deltas = torchaudio.functional.compute_deltas(
                matrix_with_feats
            )
            matrix_with_feats = torch.cat(
                (matrix_with_feats, matrix_with_feats_deltas), dim=0
            )

            if self.compute_deltas_deltas:
                matrix_with_feats_deltas_deltas = torchaudio.functional.compute_deltas(
                    matrix_with_feats_deltas
                )
                matrix_with_feats = torch.cat(
                    (matrix_with_feats, matrix_with_feats_deltas_deltas), dim=0
                )

        if self.apply_mean_norm:
            matrix_with_feats = matrix_with_feats - torch.mean(matrix_with_feats, dim=0)

        if self.apply_vari_norm:
            matrix_with_feats = matrix_with_feats / torch.std(matrix_with_feats, dim=0)

        # own feature selection
        if self.compute_opensmile_extra_features and (
                "ComParE_2016" not in self.feature_type
        ):
            s = s[None, :]

            # Config OpenSMILE
            opensmile_extra_feats_set = {
                "subset": [
                    # Voicing
                    "F0final_sma",
                    "voicingFinalUnclipped_sma",
                    "jitterLocal_sma",
                    "jitterDDP_sma",
                    "shimmerLocal_sma",
                    "logHNR_sma",
                    # Energy
                    "audspec_lengthL1norm_sma",
                    "audspecRasta_lengthL1norm_sma",
                    "pcm_RMSenergy_sma",
                    "pcm_zcr_sma",
                    # Spectral
                    "pcm_fftMag_fband250-650_sma",
                    "pcm_fftMag_fband1000-4000_sma",
                    "pcm_fftMag_spectralRollOff25.0_sma",
                    "pcm_fftMag_spectralRollOff50.0_sma",
                    "pcm_fftMag_spectralRollOff75.0_sma",
                    "pcm_fftMag_spectralRollOff90.0_sma",
                    "pcm_fftMag_spectralFlux_sma",
                    "pcm_fftMag_spectralCentroid_sma",
                    "pcm_fftMag_spectralEntropy_sma",
                    "pcm_fftMag_spectralVariance_sma",
                    "pcm_fftMag_spectralSkewness_sma",
                    "pcm_fftMag_spectralKurtosis_sma",
                    "pcm_fftMag_spectralSlope_sma",
                    "pcm_fftMag_psySharpness_sma",
                    "pcm_fftMag_spectralHarmonicity_sma",
                ]
            }
            extra_transform = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                sampling_rate=self.resampling_rate,
            )

            # Extract features
            matrix_with_extra_feats = extra_transform.process_signal(s, sr)
            matrix_with_extra_feats = matrix_with_extra_feats[
                opensmile_extra_feats_set["subset"]
            ].to_numpy()
            matrix_with_extra_feats = np.nan_to_num(matrix_with_extra_feats)
            matrix_with_extra_feats = torch.from_numpy(matrix_with_extra_feats).T

            # Concatenate the features
            common_shape = min(
                matrix_with_feats.shape[1], matrix_with_extra_feats.shape[1]
            )
            matrix_with_feats = torch.cat(
                (
                    matrix_with_feats[:, :common_shape],
                    matrix_with_extra_feats[:, :common_shape],
                ),
                dim=0,
            )

        # Apply the transpose to have the features in the columns
        matrix_with_feats = matrix_with_feats.T
        return matrix_with_feats

    def _read_a_wav_file(self, wav_path: str) -> tuple[ndarray, int]:
        if os.path.getsize(wav_path) <= 44:
            app_logger.error(f"File {wav_path} is too small to be a valid wav file.")
            raise ValueError(f"File {wav_path} is too small to be a valid wav file.")

        try:
            # load the audio file
            s, sr = librosa.load(wav_path, sr=None, mono=True)

            # resample the audio file
            if (sr != self.resampling_rate) and (0 < self.resampling_rate < sr):
                sr = self.resampling_rate
                s = librosa.resample(y=s, orig_sr=sr, target_sr=self.resampling_rate)

            # apply speech activity detection
            speech_indices = librosa.effects.split(s, top_db=self.top_db)
            s = np.concatenate([s[start:end] for start, end in speech_indices])

            # apply a pre-emphasis filter
            s = librosa.effects.preemphasis(s, coef=self.pre_emphasis_coefficient)

            # normalize
            s /= np.max(np.abs(s))

            return s, sr
        except Exception as e:
            raise RuntimeError(f"An error occurred during feature extraction: {str(e)}")

    def simple_thread_wav_2_dict_with_path_and_data(
            self, wav_path: str
    ) -> dict[str, ndarray]:
        """
        The code above implements SAD, a pre-emphasis filter with a coefficient of 0.97, and normalization.
        :param wav_path: Path to the audio file
        :return: Audio samples
        """
        s, sr = self._read_a_wav_file(wav_path)

        # Check if the audio file is empty
        if not s.size or not np.any(s) or np.nan_to_num(s).sum() == 0:
            app_logger.warning(f"File {wav_path} is empty.")
            # raise ValueError(f"File {wav_path} is empty.")

        return {wav_path: s}

    def simple_thread_extract_features_from_raw_data(
            self, id_data: str, raw_data: ndarray, sampling_rate: int
    ) -> dict[str, ndarray]:
        try:
            features = self._do_feature_extraction(raw_data, sampling_rate)
        except Exception as e:
            raise RuntimeError(f"An error occurred during feature extraction: {str(e)}")

        return {id_data: features}

    def load_all_wav_files_from_dataset(
            self,
            dataset: pd.DataFrame,
            name_column_with_path: str,
            num_cores: int = None,
    ) -> dict:
        app_logger.info("Feature Extractor - Loading all wav files from the dataset")
        if num_cores is None:
            num_cores = multiprocessing.cpu_count()

        try:
            multi_processor = MultiProcessor(num_cores=num_cores)

            raw_data_paths = dataset[name_column_with_path].drop_duplicates().tolist()
            raw_data_matrix = multi_processor.process_with_multiprocessing(
                raw_data_paths, self.simple_thread_wav_2_dict_with_path_and_data
            )

        except Exception as e:
            message = f"An error occurred during feature extraction: {str(e)}"
            app_logger.error(message)
            raise RuntimeError(message)

        if len(raw_data_matrix) != len(raw_data_paths):
            app_logger.warning(
                "Some files were not loaded on parallel processing. Processing them sequentially."
            )

            # Get the missing files form raw_data_paths that aren´t in raw_data_matrix
            missing_files = list(set(raw_data_paths) - set(raw_data_matrix.keys()))

            for missing_file in missing_files:
                missing_file_raw_data = (
                    self.simple_thread_wav_2_dict_with_path_and_data(missing_file)
                )
                raw_data_matrix.update(missing_file_raw_data)

        app_logger.info(
            "Feature Extractor - All wav files from the dataset were loaded"
        )
        return raw_data_matrix

    def extract_features_from_raw_data(
            self,
            raw_data_matrix: dict[str, np.ndarray],
            num_cores: int = None,
    ) -> dict[str, np.ndarray]:
        if num_cores is None:
            num_cores = multiprocessing.cpu_count()

        app_logger.info("Feature Extractor - Loading all wav files from the dataset")

        def worker(id_data, raw_data):
            return self.simple_thread_extract_features_from_raw_data(
                id_data, raw_data, self.resampling_rate
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_cores
            ) as executor:
                future_to_id_data = {
                    executor.submit(worker, id_data, raw_data): id_data
                    for id_data, raw_data in raw_data_matrix.items()
                }

                features = {}
                for future in tqdm(
                        concurrent.futures.as_completed(future_to_id_data),
                        total=len(future_to_id_data),
                ):
                    id_data = future_to_id_data[future]
                    try:
                        result = future.result()
                        features.update(result)
                    except Exception as e:
                        app_logger.error(
                            f"An error occurred during feature extraction for {id_data}: {str(e)}"
                        )
                        raise RuntimeError(
                            f"An error occurred during feature extraction: {str(e)}"
                        )

            app_logger.info(
                "Feature Extractor - All wav files from the dataset were loaded"
            )
            return features

        except Exception as e:
            app_logger.error(f"An error occurred during feature extraction: {str(e)}")
            raise RuntimeError(f"An error occurred during feature extraction: {str(e)}")
