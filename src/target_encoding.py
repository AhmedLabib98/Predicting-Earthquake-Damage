from category_encoders import TargetEncoder
import pandas as pd 

def target_encoding_train(df, y):
    proc_df = df.copy()

    # Converting the geo cols to str data type
    proc_df['geo_level_1_id'] = proc_df['geo_level_1_id'].astype(str)
    proc_df['geo_level_2_id'] = proc_df['geo_level_2_id'].astype(str)
    proc_df['geo_level_3_id'] = proc_df['geo_level_3_id'].astype(str)

    # Merge the two DataFrames based on building_id
    proc_df = pd.merge(proc_df, y, on = 'building_id')
    y = proc_df['damage_grade']

    # Dropping the target column from the proc_df DataFrame
    proc_df = proc_df.drop(['damage_grade'], axis = 1)

    # Target Encoding the cols:
    geo_level_cols = proc_df[['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']]

    T_encoder = TargetEncoder()

    # We train the encoder on the training data, and transform the geo_level cols, and we replace them in the proc_df DataFrame
    for col in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']:
        proc_df[col] = T_encoder.fit_transform(proc_df[col], y)

    return proc_df, T_encoder

def target_encoding_test(df, T_encoder):
    proc_df = df.copy()
    # Converting the geo cols to str data type
    proc_df['geo_level_1_id'] = proc_df['geo_level_1_id'].astype(str)
    proc_df['geo_level_2_id'] = proc_df['geo_level_2_id'].astype(str)
    proc_df['geo_level_3_id'] = proc_df['geo_level_3_id'].astype(str)

    # We do not use the fit_transform method here to prevent data leakage
    for col in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']:
        proc_df[col] = T_encoder.transform(proc_df[col])

    return proc_df

