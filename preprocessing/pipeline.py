import os
import pandas as pd
from feature_engineer import feature_extraction
from constants import WINDOW_LENGTH, STEP_RATE, USEFUL_CHANNELS
from prepare_raw_data import download_data, prepare_matlab_file, extract_data
from ica import filter_noise_with_ica

TRAIN_PATH = "./data/df_train.csv"
TEST_PATH = "./data/df_test.csv"
PARENT_DIRNAME = os.path.dirname(os.path.dirname(__file__))


def load_raw_data(OUTPUT_FOLDER=os.path.join(PARENT_DIRNAME, "data")) -> None:
    """
    Create .csv files for all matlab record of each person (contains all 7 record files in a .csv file).

    Returns:
        None
    """
    mat_paths = download_data()
    subject = 1
    for i in range(1, 34, 7):
        print(f"Loading subject {subject} raw data...")
        df_subject: pd.DataFrame = None
        for file_num in range(i, min(i + 7, 35)):
            matlab_df = prepare_matlab_file(mat_paths[file_num - 1])
            df_subject = (
                matlab_df if df_subject is None else pd.concat([df_subject, matlab_df])
            )
        df_subject.reset_index(drop=True, inplace=True)
        output_path = os.path.join(OUTPUT_FOLDER, f"raw_{subject}.csv")
        df_subject.to_csv(output_path, index=False)
        subject += 1


def prepare_train_test_csv_files() -> None:
    """
    Create train and test csv files, print out its shape and path.
    Both files are preprocessed, filtered using ICA, and feature engineer.

    Returns:
        None
    """
    print("Downloading Data...")
    mat_paths = download_data()
    train_df: pd.DataFrame = None
    test_df: pd.DataFrame = None

    print("Processing...")
    for idx, path in enumerate(mat_paths):
        matlab_df = prepare_matlab_file(path)
        preprocessed_df = extract_data(
            matlab_df, skip_first_5s=True
        )
        ica_df = filter_noise_with_ica(preprocessed_df)
        fe_df = feature_extraction(ica_df, WINDOW_LENGTH, STEP_RATE)
        if (
            1 <= (idx + 1) % 7 <= 2
        ):  # if file is recorded in the first and the second day -> test
            test_df = fe_df if test_df is None else pd.concat([test_df, fe_df])
        else:
            train_df = fe_df if train_df is None else pd.concat([train_df, fe_df])

    print("Train DataFrame Shape:", train_df.shape)
    print("Test DataFrame Shape:", test_df.shape)

    train_df.to_csv(TRAIN_PATH, index=False)
    train_df.to_csv(TEST_PATH, index=False)

    print("Train DataFrame is saved at", TRAIN_PATH)
    print("Test DataFrame is saved at", TEST_PATH)


if __name__ == "__main__":
    print(os.path.join(PARENT_DIRNAME, "data"))
    choice = input("Load train test data? [Y]/[N] ")
    if choice.lower() == "y":
        prepare_train_test_csv_files()
    else:
        print("Bye!")