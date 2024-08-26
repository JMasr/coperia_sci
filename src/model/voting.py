import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class ExpertVotingSystem(BaseEstimator, ClassifierMixin):
    def __init__(self, voting: str = 'soft'):
        """
        Initialize the ExpertVotingSystem object.

        :param voting: Specifies the type of voting. "Soft" for weighted voting, "hard" for majority voting.
                       Default is "soft".
        """
        self.voting = voting

    def fit(self, x, y=None):
        """
        This is a no-op method. The model does not require training but this method is required by scikit-learn API.
        """
        return self

    def predict(self,
                df: pd.DataFrame,
                id_col: str = 'id',
                class_col: str = 'class',
                prob_col: str = 'probabilidad') -> pd.DataFrame:
        """
        Predict the class of each sample in the input DataFrame.

        :param df: DataFrame with the unique identifier, class and probability of each sample.
        :param id_col: Name of the column with the unique identifier of each sample.
        :param class_col: Name of the column with the class of each sample.
        :param prob_col: Name of the column with the probability of the class of each sample.
        :return: DataFrame with the following columns:
        """
        grouped = df.groupby(id_col)

        final_predictions = []
        for sample_id, group in grouped:
            if self.voting == 'soft':
                # Soft voting: select the class with the highest weighted sum of probabilities
                weighted_sum = group.groupby(class_col)[prob_col].sum()
                final_class = weighted_sum.idxmax()
                score = weighted_sum.max()
            elif self.voting == 'hard':
                # Hard voting: select the class with the highest frequency
                final_class = group[class_col].mode()[0]
                score = group[class_col].value_counts().max()
            else:
                raise ValueError("Invalid voting type. Please use 'soft' or 'hard'.")

            final_predictions.append({id_col: sample_id, class_col: final_class, "voting_score": score})

        df_final = pd.DataFrame(final_predictions)
        # Normalize the voting score between 0 and 1
        df_final["voting_score"] = df_final["voting_score"] / df_final["voting_score"].max()

        return df_final
