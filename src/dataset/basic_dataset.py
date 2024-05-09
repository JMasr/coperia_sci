import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from src.logger import app_logger
from src.files import csv_file_to_dataframe
from src.exceptions import MetadataError


class LocalDataset:
    def __init__(self, name: str):
        # Validation of parameters
        if not isinstance(name, str):
            app_logger.error(f"LocalDataset - The name of the dataset provided isn´t a string: {name}")
            raise TypeError("The name of the dataset must be a string")

        self.name: str = name
        self.raw_metadata: pd.DataFrame = pd.DataFrame()

        self.post_processed_metadata: pd.DataFrame = pd.DataFrame()
        self.subsets: list = []

        app_logger.info(f"LocalDataset - Successful created with name: {name}")

    def load_metadata_from_csv(self,
                               path_to_metadata: str,
                               **kwargs):
        try:
            self.raw_metadata = csv_file_to_dataframe(path_to_metadata, **kwargs)
            app_logger.info("LocalDataset - The CSV file was successful read")
        except ValueError or FileNotFoundError as e:
            app_logger.error(f"LocalDataset - The CSV file was´t read successful: {e}")
            raise MetadataError(f"{e}")

    def transform_metadata(self,
                           transformations: list):
        if self.raw_metadata.empty:
            message = "LocalDataset - Metadata is empty. Please, load a metadata first."
            app_logger.error(message)
            raise ValueError(message)
        else:
            try:
                app_logger.info(f"LocalDataset - Starting {len(transformations)} transformations over metadata: ")
                df = self.raw_metadata.copy()
                for index, transformation in enumerate(transformations):
                    df = transformation(df)
                    app_logger.info(f"LocalDataset - Transformation #{index +1} finished successfully")

                self.post_processed_metadata = df
            except Exception as e:
                raise MetadataError(f"An error occurred while transforming metadata: {e}")

    def make_1_fold_subsets(self,
                            target_class_for_fold: str,
                            target_data_for_fold: str,
                            target_label_for_fold: str,
                            test_size: float = 0.2,
                            seed: int = 42
                            ):
        """
        Split the data into train and test subsets. The split is done using a target column.
        For example, patients ids.
        :param target_class_for_fold: Column with the target class to split the data. For example, patient_id
        :param target_data_for_fold: Column with the data to split. For example, the ids of the audios
        :param target_label_for_fold: Column with the labels of the data.
        :param test_size: Size of the test set
        :param seed: Random seed for the split
        :return: A list with the train and test subsets
        """
        app_logger.info(f"LocalDataset - Starting the creation of train and test subsets")

        # Prepare the metadata
        try:
            exp_metadata = self.post_processed_metadata.copy()
            target_class_data = exp_metadata[[target_class_for_fold, target_label_for_fold]].drop_duplicates()
            target_ids = target_class_data[target_class_for_fold]
            labels_of_class = target_class_data[target_label_for_fold]

            # Split the data
            pat_train, pat_test, pat_labels_train, pat_labels_test = train_test_split(target_ids, labels_of_class,
                                                                                      test_size=test_size,
                                                                                      stratify=labels_of_class,
                                                                                      random_state=seed,
                                                                                      shuffle=True)

            # Using the target subsets to select the samples
            sample_data_train = exp_metadata[(exp_metadata[target_class_for_fold].isin(pat_train))]
            samples_train = sample_data_train[target_data_for_fold]
            labels_train = sample_data_train[target_label_for_fold]

            sample_data_test = exp_metadata[(exp_metadata[target_class_for_fold].isin(pat_test))]
            samples_test = sample_data_test[target_data_for_fold]
            labels_test = sample_data_test[target_label_for_fold]

            # Log the lengths of the subsets
            app_logger.info("LocalDataset - Subsets creation successful")
            app_logger.info(f"LocalDataset - Test-set: {len(pat_test)} instance & {len(sample_data_test)} samples")
            app_logger.info(f"LocalDataset - Train-set: {len(pat_train):} instance & {len(sample_data_train)} samples")
            return [[samples_train, samples_test, labels_train, labels_test]]
        except Exception as e:
            app_logger.error(f"LocalDataset - The subsets creation fails. Error: {e}")
            raise MetadataError(e)

    def make_k_fold_subsets(self,
                            target_class_for_fold: str,
                            target_data_for_fold: str,
                            target_label_for_fold: str,
                            k_fold: int,
                            seed: int):
        app_logger.info(f"LocalDataset - Starting the creation of {k_fold}-folds for train and test")

        try:
            exp_metadata = self.post_processed_metadata.copy()

            # Create the KFold object
            k_folds = []
            sklearn_k_fold_operator = KFold(n_splits=k_fold, shuffle=True, random_state=seed)

            k_folds_index_generator = sklearn_k_fold_operator.split(exp_metadata[target_class_for_fold])
            for ind, (train_index, test_index) in enumerate(k_folds_index_generator):
                # Get the train and test data
                samples_train = exp_metadata.iloc[train_index][target_data_for_fold]
                samples_test = exp_metadata.iloc[test_index][target_data_for_fold]

                lable_train = exp_metadata.iloc[train_index][target_label_for_fold]
                lable_test = exp_metadata.iloc[test_index][target_label_for_fold]

                k_folds.append([samples_train, samples_test, lable_train, lable_test])

            app_logger.info("LocalDataset - K-folds process successful")
            self.subsets = k_folds
            return k_folds
        except Exception as e:
            app_logger.error(f"LocalDataset - K-folds process fails. Error: {e}")
            raise MetadataError(e)
