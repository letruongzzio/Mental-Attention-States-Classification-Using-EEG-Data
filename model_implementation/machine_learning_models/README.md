# Machine Learning Models

This file is used to train EEG data with Machine Learning models (see `training_models.py` for more details).

## Requirements

Install the libraries with the versions shown in the `requirements.txt` file.

**Note**: Since this project is run on Kaggle and is provided with the CuML library, running locally may be affected as this library cannot be installed (especially on Linux OS). You should consider running this project on Kaggle or Google Colab.

## Installation

1. Clone repository:

    First, you need to clone this repository to your computer. Open CMD (or Terminal) and run the command:

    ```bash
    git clone https://github.com/yusnivtr/PRML-MidTerm-Project/tree/main
    cd ~PRML-MidTerm-Project/model_implementation/machine_learning_models
    ```

2. Install libraries (if needed):

    You need to install necessary libraries using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the code:

    After installing the libraries, you can run the code by using the following command:

    ```bash
    python training_models.py
    ```

4. Check the results:

    After running the code, you can check the results in the `output` folder.

    **Note**: Because the `training_models.py` file do not set the specific output folder for each model, such as `lda_results`, I have to manually create the output folders for each model (our output files are just saved in the `output` folder). You can change the output folder in the code if you want to save the results in different folders.

## Notice

- The `training_models.py` file is used to train the EEG data with different Machine Learning models. You can change the models, hyperparameters, and other settings in this file.
- Time to train the models is quite long, so you should consider running this project on Google Colab or Kaggle (about 4 hours for all models).

