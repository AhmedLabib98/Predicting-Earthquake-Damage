import numpy as np
import pandas as pd

def convert_to_pd(my_array):
    """"Converts numpy array to numeric array and then to pandas dataframe"""
    # y_pred_test to int 
    my_array = my_array.astype(int)

    # y_pred_test to pd.DataFrame and flatten
    my_df = pd.DataFrame(my_array).values.ravel()

    return my_df


