from category_encoders import TargetEncoder
import pandas as pd 

def target_encoding(df_train, df_test, y):
    t_encoder = TargetEncoder()

    # Create a copy of the train and test dataframes
    df_train = df_train.copy()
    df_test = df_test.copy()

    # Merge the two DataFrames based on building_id
    df_train = df_train.merge(y, on = 'building_id')  

    # Defining the target variable
    y = y['damage_grade'] 

    #Dropping the target column from the df_train DataFrame
    df_train = df_train.drop(['damage_grade'], axis = 1)

    # Converting geo_levels to str
    for col in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype(str)
            df_test[col] = df_test[col].astype(str)

    # for loop to encode geo levels cols
    for col in ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']:
        if col in df_train.columns and col in df_test.columns:
            df_train[col] = t_encoder.fit_transform(df_train[col], y)
            df_test[col] = t_encoder.transform(df_test[col])

    return df_train, df_test