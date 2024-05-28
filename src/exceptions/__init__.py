class MetadataError(Exception):
    """
    Exception raised when an error occurs while transforming pd.DataFrames
    """
    pass


class AudioProcessingError(Exception):
    """
    Exception raised when an error occurs while processing audio files
    """
    pass
