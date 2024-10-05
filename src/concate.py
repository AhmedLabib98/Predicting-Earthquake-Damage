import numpy as np
import pandas as pd

def concat_pd(df_id, df_label):
    """ 
    Concatenate the
    df_id (has col building_id), with the
    df_label (has col label) 
    """
    df = pd.DataFrame({'building_id': df_id['building_id'], 'damage_grade': df_label})

    return df
