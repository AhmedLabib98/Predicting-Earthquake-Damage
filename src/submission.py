import numpy as np
import pandas as pd

def submit(df, file_name):
    """ save pd df as csv file"""
    pd.DataFrame.to_csv(df, file_name, index = False)