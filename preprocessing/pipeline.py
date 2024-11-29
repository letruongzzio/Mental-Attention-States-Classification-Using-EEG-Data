import pandas as pd
from prepare_raw_data import *
from feature_engineer import *
from constants import WINDOW_LENGTH, STEP_RATE

TRAIN_PATH = "./data/df_train.csv"
TEST_PATH = "./data/df_test.csv"

if __name__ == "__main__":
    print("Downloading Data...")
    mat_paths = download_data()
    train_df = None
    test_df = None

    print("Processing...")
    for idx, path in enumerate(mat_paths):
        matlab_df = prepare_matlab_file(path)
        preprocessed_df = extract_data(matlab_df, take_useful_channels=True, skip_first_5s=True)
        fe_df = feature_extraction(preprocessed_df, WINDOW_LENGTH, STEP_RATE)
        if 1 <= (idx + 1) % 7 <= 2: # if file is recoreded in the first and the second day -> test
            test_df = fe_df if test_df is None else pd.concat([test_df, fe_df])
        else:
            train_df = fe_df if train_df is None else pd.concat([train_df, fe_df])
    
    print("Train DataFrame Shape:", train_df.shape)
    print("Test DataFrame Shape:", test_df.shape)

    train_df.to_csv(TRAIN_PATH, index=False)
    train_df.to_csv(TEST_PATH, index=False)

    print("Train DataFrame is saved at", TRAIN_PATH)
    print("Test DataFrame is saved at", TEST_PATH)
    