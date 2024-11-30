# Steps to prepare data for model training:
Change directory into `preprocess/` directory
```bash
cd preprocess
```

Create a new Python script, or Jupyter notebook and run the functions in `pipeline.py`.
```python
from pipeline import prepare_train_test_csv_files, load_raw_data

# If you want to have raw data, use load_raw_data(). More detail in its docstring.
# load_raw_data

# If you want to have train and test data, use prepare_train_test_csv_files(). More detail in its docstring.
# prepare_train_test_csv_files
```

In either case, you just call the required function once. It will automatically download the data for you into the `data/` directory.