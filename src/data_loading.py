import pandas as pd

def data_loading():
    # Load the data
    train_values = pd.read_csv('data/01_raw/train_values.csv')
    train_label = pd.read_csv('data/01_raw/train_labels.csv')
    test_values = pd.read_csv('data/01_raw/test_values.csv')
    return train_values, train_label, test_values