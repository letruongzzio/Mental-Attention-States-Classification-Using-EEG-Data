import os
import warnings
from typing import Literal

import kagglehub
import scipy
import pandas as pd

from constants import COLUMN_NAMES
from preprocessing_eeg import preprocess_eeg_dataframe

DATASET_URL = "inancigdem/eeg-data-for-mental-attention-state-detection"


def prepare_matlab_file(matlab_file_index: int) -> str:
    """
    Return the path of the first matlab file of the dataset.
    """
    try:
        path: str = kagglehub.dataset_download(DATASET_URL)
        print(path)
    except ConnectionError:
        path = "~/.cache/kagglehub/datasets/inancigdem/eeg-data-for-mental-attention-state-detection/versions/1"
    data_path = os.path.join(path, "EEG Data")
    data_files: list[str] = [
        os.path.join(data_path, file) for file in os.listdir(data_path) if "record" in file
    ]
    data_files.sort(key=lambda x: int(x.split("eeg_record")[1].split(".mat")[0]))

    matlab_file = data_files[matlab_file_index]
    
    mat_data = scipy.io.loadmat(matlab_file)
    data = mat_data["o"][0][0]["data"]

    df = pd.DataFrame(data, columns=COLUMN_NAMES)
    df = df.reset_index()
    df = df.rename(columns={"index": "t"})

    if df['ED_INTERPOLATED'].sum() > 0:
        warnings.warn("Have interpolated values", UserWarning)
    
    return df


def extract_data(file_index: int = 0) -> pd.DataFrame:
    df = prepare_matlab_file(file_index)

    channel_columns = ["t"]
    channel_columns.extend(df.columns[4:18])
    df_channels = df[channel_columns]

    def get_state(
        time,
    ): #-> Literal["focused"] | Literal["drownsy"] | Literal["unfocused"]:
        if time <= 10 * 128 * 60:
            return "focused"
        elif time > 20 * 128 * 60:
            return "drownsy"
        else:
            return "unfocused"

    df_channels.loc[:, "state"] = df_channels.loc[:, "t"].apply(get_state)

    # Preprocess the data
    filtered_df = preprocess_eeg_dataframe(df_channels, channel_columns[1:])

    return filtered_df
