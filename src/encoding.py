from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
def basic_encoding(df):
    """
    a for loop iterates over all the columns with object
    data type, and then creates a new column into the copied
    data frame, with the suffix "_encoded" appended to the
    original column name.
    We take the 3 geo columns, change their type to str,
    and then target encode them and add them to the DataFrame
    """
    # Selecting all the columns that have the object data type
    object_cols = df.select_dtypes(include = ['object']).columns
    # Copying the data, so the old data does not change
    proc_df = df.copy()
    # Creating a LabelEncoder object
    encoder = LabelEncoder()
    # for loop iterates over cols with object type, replaces all
    # of them with the new encoded cols
    for col in object_cols:
        proc_df[col] = encoder.fit_transform(proc_df[col])
    # # Converting the geo cols to str data type
    # proc_df['geo_level_1_id'] = proc_df['geo_level_1_id'].astype(str)
    # proc_df['geo_level_2_id'] = proc_df['geo_level_2_id'].astype(str)
    # proc_df['geo_level_3_id'] = proc_df['geo_level_3_id'].astype(str)
    # # Target Encoding the cols:
    # geo_level_cols = proc_df['geo_level_1_id',
    #                     'geo_level_2_id',
    #                     'geo_level_3_id'
    #                     ]
    # T_encoder = TargetEncoder()
    # proc_df[['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']] = T_encoder.fit_transform(geo_level_cols)
    
    return proc_df