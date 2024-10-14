from category_encoders import TargetEncoder
import pandas as pd 

def target_encoding_train(df, y):
    proc_df = df.copy()

    # Converting the geo cols to str data type
    for col in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']:
        proc_df[col] = proc_df[col].astype(str)

    # Merge the two DataFrames based on building_id
    proc_df = pd.merge(proc_df, y, on = 'building_id')
    y = proc_df['damage_grade']

    # Dropping the target column from the proc_df DataFrame
    proc_df = proc_df.drop(['damage_grade'], axis = 1)

    t_encoder = TargetEncoder()

    # We train the encoder on the training data, and transform the geo_level cols, and we replace them in the proc_df DataFrame
    for col in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']:
        if col in proc_df.columns:
            proc_df[col] = t_encoder.fit_transform(proc_df[col], y)
        else:
            print(f"Column '{col}' not found in the DataFrame.")
            
    return proc_df, t_encoder

def target_encoding_test(df, t_encoder):
    
    proc_df = df.copy()
    
    # Trouble shooting geo_level_3_id:
    print("Columns in test data:", df.columns)
    # Rename columns if necessary
    column_mapping = {
        'geo level 1 id': 'geo_level_1_id',
        'geo level 2 id': 'geo_level_2_id',
        'geo level 3 id': 'geo_level_3_id'
    }
    proc_df.rename(columns=column_mapping, inplace=True)

 

    # Converting the geo cols to str data type
    for col in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']:
        if col in proc_df.columns:
            proc_df[col] = proc_df[col].astype(str)
            proc_df[col] = t_encoder.transform(proc_df[col]) # We do not use the fit_transform method here to prevent data leakage
        else:
            print(f"Column '{col}' not found in the DataFrame.")

    return proc_df

