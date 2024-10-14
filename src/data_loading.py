import pandas as pd

def data_loading():
    # Load the data
    train_values = pd.read_csv('data/01_raw/train_values.csv', index_col="building_id")
    train_label = pd.read_csv('data/01_raw/train_labels.csv', index_col="building_id")
    test_values = pd.read_csv('data/01_raw/test_values.csv', index_col="building_id")
    return train_values, train_label, test_values