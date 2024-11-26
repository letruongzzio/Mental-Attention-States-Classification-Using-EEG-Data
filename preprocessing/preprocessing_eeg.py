import pandas as pd
import numpy as np
from filter import butter_bandpass_filter, notch_filter

FS = 128  # Sampling frequency
LOWCUT = 0.5  # Low cut-off frequency for band-pass filter
HIGHCUT = 40  # High cut-off frequency for band-pass filter


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

    # Apply band-pass filter
    eeg_data = butter_bandpass_filter(eeg_data, LOWCUT, HIGHCUT, FS)

    # Apply 50Hz notch filter
    eeg_data = notch_filter(eeg_data, FS, freq=50.0)

    # Re-reference (Common Average Reference)
    mean_ref = np.mean(eeg_data, axis=1, keepdims=True)
    eeg_data = eeg_data - mean_ref

    # Epoching
    epochs = np.arange(len(df)) // FS

    # Convert back to DataFrame
    df.loc[:, channel_cols] = eeg_data
    df.loc[:, "epoch"] = epochs

    return df
