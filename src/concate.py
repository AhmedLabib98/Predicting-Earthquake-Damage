import pandas as pd

def concatenate(train_values, test_values):
    train_values["type"] = "TRAIN"
    test_values["type"] = "TEST"
    return pd.concat([train_values, test_values], axis=0)