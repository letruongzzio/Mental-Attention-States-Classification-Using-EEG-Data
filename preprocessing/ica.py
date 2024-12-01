import mne
from mne.io.array.array import RawArray
import pandas as pd
from constants import FS, USEFUL_CHANNELS, ALL_CHANNELS

FILE_PATH = "~/Documents/PRML-MidTerm-Project/data/preprocessed_eeg15.csv"

# Đổi tên các kênh trong dữ liệu raw
columns_mapping = {
    "ED_AF3": "AF3",
    "ED_F7": "F7",
    "ED_F3": "F3",
    "ED_FC5": "FC5",
    "ED_T7": "T7",
    "ED_P7": "P7",
    "ED_O1": "O1",
    "ED_O2": "O2",
    "ED_P8": "P8",
    "ED_T8": "T8",
    "ED_FC6": "FC6",
    "ED_F4": "F4",
    "ED_F8": "F8",
    "ED_AF4": "AF4",
}


def create_raw(file_path=FILE_PATH, dataframe: pd.DataFrame = None) -> RawArray:
    if dataframe is None:
        df = pd.read_csv(file_path)
    else:
        df = dataframe

    df = df.rename(columns=columns_mapping)
    eeg_data = df.iloc[:, 1:-1].values.T

    channel_names = df.iloc[:, 1:-1].columns.tolist()
    channel_types = ["eeg"] * eeg_data.shape[0]

    info = mne.create_info(ch_names=channel_names, ch_types=channel_types, sfreq=FS)
    info.set_montage("standard_1020")

    return mne.io.RawArray(eeg_data, info).apply_function(lambda x: x * 1e-6)


def find_bad_channels(
    ica: mne.preprocessing.ICA,
    raw: RawArray,
    use_lof: bool = False,
):
    """
    Return the indices of bad channels.
    Params:
        ica: The fitted ICA instance
    """
    bad_names_set = set()
    eog_channels = [channel for channel in ["AF3", "AF4"] if channel in ica.ch_names]
    bad_eog_indices, _ = ica.find_bads_eog(
        raw, threshold="auto", ch_name=eog_channels
    )
    bad_names_set.update([int(index) for index in bad_eog_indices])
    if use_lof:
        bad_lof_names = mne.preprocessing.find_bad_channels_lof(raw)
        bad_names_set.update(bad_lof_names)
    return list(bad_names_set)


def filter_noise_with_ica(df: pd.DataFrame) -> pd.DataFrame:
    raw = create_raw(dataframe=df)

    ica = mne.preprocessing.ICA(
        n_components=None,
        method="infomax",
        max_iter="auto",
        random_state=42,
        fit_params=dict(extended=True),
    )
    ica.fit(raw)

    bad_channels_indices = find_bad_channels(ica, raw, use_lof=True)
    ica.apply(raw, exclude=bad_channels_indices)
    df[ALL_CHANNELS] = raw.get_data().T
    return df
