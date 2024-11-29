import os
import warnings
from typing import Literal

import kagglehub
import scipy
import pandas as pd

from constants import COLUMN_NAMES, USEFUL_CHANNELS, FS
from preprocessing_eeg import preprocess_eeg_dataframe

DATASET_URL = "inancigdem/eeg-data-for-mental-attention-state-detection"


def download_data() -> list:
    """
    Return list of downloaded file paths
    """
    try:
        path: str = kagglehub.dataset_download(DATASET_URL)
        print(path)
    except ConnectionError:
        path = "~/.cache/kagglehub/datasets/inancigdem/eeg-data-for-mental-attention-state-detection/versions/1"
        path = os.path.expanduser(path) # for all user, all OS
    data_path = os.path.join(path, "EEG Data")
    data_files: list[str] = [
        os.path.join(data_path, file) for file in os.listdir(data_path) if "record" in file
    ]
    data_files.sort(key=lambda x: int(x.split("eeg_record")[1].split(".mat")[0]))

    return data_files

def prepare_matlab_file(matlab_file_path: str) -> pd.DataFrame:
    """
    Return o[data] of .mat file (pd.DataFrame)
    """
    mat_data = scipy.io.loadmat(matlab_file_path)
    data = mat_data["o"][0][0]["data"]

    df = pd.DataFrame(data, columns=COLUMN_NAMES)
    df = df.reset_index()
    df = df.rename(columns={"index": "t"})

    if df['ED_INTERPOLATED'].sum() > 0:
        warnings.warn("Have interpolated values", UserWarning)
    
    return df

def extract_data(matlab_df: pd.DataFrame, take_useful_channels: bool = False, skip_first_5s: bool = False) -> pd.DataFrame:
    """
    Return filtered_df with columns are: t, channels, state
    """

    channel_columns = ["t"]

    if take_useful_channels:
        channel_columns.extend(USEFUL_CHANNELS)
    else:
        channel_columns.extend(matlab_df.columns[4:18])
    matlab_df = matlab_df[channel_columns]

    def get_state(
        time,
    ) -> Literal["focused", "drowsy", "unfocused"]:
        if time <= 10 * 128 * 60:
            return "focused"
        elif time > 20 * 128 * 60:
            return "drowsy"
        else:
            return "unfocused"

    matlab_df = matlab_df.copy() # To fix warning
    matlab_df["state"] = matlab_df["t"].apply(get_state)

    if skip_first_5s:
        matlab_df = matlab_df.iloc[5*FS:, :]

    # Preprocess the data
    filtered_df = preprocess_eeg_dataframe(matlab_df, channel_columns[1:])

    return filtered_df