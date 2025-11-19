import pandas as pd
from pathlib import Path

def load_datasets(base_dir):
    data_dir = base_dir / "data"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_test
