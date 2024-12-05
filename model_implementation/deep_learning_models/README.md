# Deep Learning Models

This repository contains implementations of deep learning models for processing and analyzing various datasets. The structure of the directory is organized into Python scripts and Jupyter notebooks for clarity and ease of experimentation.

## Repository Structure

```
deep_learning_models/
├── EEGNet.py          # Python implementation of the EEGNet model
├── MLP_Model.py       # Python implementation of a Multi-Layer Perceptron (MLP)
├── notebooks/         # Folder containing Jupyter notebook versions of the models
│   ├── EEGNet.ipynb   # Notebook implementation of EEGNet
│   ├── MLP_Model.ipynb # Notebook implementation of the MLP model
```

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yusnivtr/PRML-MidTerm-Project
   cd ~/PRML-MidTerm-Project/model_implementation/deep_learning_models
   ```

2. **Install Dependencies:** Ensure you install the required libraries by running:

    ```bash
    pip install -r requirements.txt
    ```

3. **Running the Scripts:**
    - To use the models directly in Python, run:
        ```bash
        python EEGNet.py
        python MLP_Model.py
        ```
    - To experiment with the models interactively, use the Jupyter notebooks:
        ```bash
        jupyter notebook notebooks/EEGNet.ipynb
        juptyer notebook notebooks/MLP_Model.ipynb
        ```

## Acknowledgements

The implementation of the EEGNet model is based on the original paper by [Lawhern et al. (2018)](https://arxiv.org/abs/1611.08024). The implementation of the MLP model is a simple feedforward neural network for classification tasks.