import os
import pandas as pd
from feature_engineer import feature_extraction
from constants import WINDOW_LENGTH, STEP_RATE, USEFUL_CHANNELS
from prepare_raw_data import download_data, prepare_matlab_file, extract_data
from ica import filter_noise_with_ica
from sklearn.model_selection import train_test_split

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
    full_df: pd.DataFrame = None

    print("Processing...")
    for idx, path in enumerate(mat_paths):
        matlab_df = prepare_matlab_file(path)
        preprocessed_df = extract_data(
            matlab_df, skip_first_5s=True
        )
        ica_df = filter_noise_with_ica(preprocessed_df)
        fe_df = feature_extraction(ica_df, WINDOW_LENGTH, STEP_RATE, take_useful_channels=True)

        full_df = fe_df if full_df is None else pd.concat([full_df, fe_df])

    # Split the full_df into train and test sets with similar distribution
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42, shuffle=True, stratify=full_df['state'])

    print("Train DataFrame Shape:", train_df.shape)
    print("Test DataFrame Shape:", test_df.shape)

    print("In Train DataFrame:", train_df['state'].value_counts())
    print("In Test DataFrame:", test_df['state'].value_counts())

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("Train DataFrame is saved at", TRAIN_PATH)
    print("Test DataFrame is saved at", TEST_PATH)


if __name__ == "__main__":
    print(os.path.join(PARENT_DIRNAME, "data"))
    choice = input("Load train test data? [Y]/[N] ")
    if choice.lower() == "y":
        prepare_train_test_csv_files()
    else:
        print("Bye!")