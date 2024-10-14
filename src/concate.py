import numpy as np
import pandas as pd

def concat_pd(df_id, df_label):
    """ 
    Concatenate the following:
    the id info contained in the rownames of the df_id dataset
    with the
    predicted labels in the flat df_label 
    """
    df = pd.DataFrame({'building_id': df_id.index, 'damage_grade': df_label})

    return df
