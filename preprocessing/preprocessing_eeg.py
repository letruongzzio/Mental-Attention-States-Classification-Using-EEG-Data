import pandas as pd
import numpy as np
from filter import butter_highpass_filter
from constants import FS, LOWCUT


def preprocess_eeg_dataframe(df: pd.DataFrame, channel_cols: list[str]) -> pd.DataFrame:
    """
    Preprocess EEG data from DataFrame.
    Args:
        df (DataFrame): DataFrame containing EEG data.
        channel_cols (list[str]): List of EEG channel column names (e.g., ['ED_AF3', 'ED_F7', ...]).
    Returns:
        df (DataFrame): DataFrame after preprocessing.
    """
    # Extract EEG data
    eeg_data = df[channel_cols].values  # Extract EEG data (numpy array)

    # Apply high-pass filter
    eeg_data = butter_highpass_filter(eeg_data, LOWCUT, FS)

    # Re-reference (Common Average Reference)
    mean_ref = np.mean(eeg_data, axis=1, keepdims=True)
    eeg_data = eeg_data - mean_ref

    # Convert back to DataFrame
    df.loc[:, channel_cols] = eeg_data

    return df
