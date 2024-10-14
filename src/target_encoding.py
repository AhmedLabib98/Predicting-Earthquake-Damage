from category_encoders import TargetEncoder

def target_encoding_train(df, y):
    proc_df = df.copy()
    # Converting the geo cols to str data type
    proc_df['geo_level_1_id'] = proc_df['geo_level_1_id'].astype(str)
    proc_df['geo_level_2_id'] = proc_df['geo_level_2_id'].astype(str)
    proc_df['geo_level_3_id'] = proc_df['geo_level_3_id'].astype(str)

    # Target Encoding the cols:
    geo_level_cols = proc_df[['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']]

    T_encoder = TargetEncoder()

    for col in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']:
        proc_df[col] = T_encoder.fit_transform(proc_df[col], y)

    return proc_df

def target_encoding_test(df, y):
    proc_df = df.copy()
    # Converting the geo cols to str data type
    proc_df['geo_level_1_id'] = proc_df['geo_level_1_id'].astype(str)
    proc_df['geo_level_2_id'] = proc_df['geo_level_2_id'].astype(str)
    proc_df['geo_level_3_id'] = proc_df['geo_level_3_id'].astype(str)

    # Target Encoding the cols:
    geo_level_cols = proc_df[['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']]

    T_encoder = TargetEncoder()

    for col in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']:
        proc_df[col] = T_encoder.fit(proc_df[col], y)

    return proc_df

