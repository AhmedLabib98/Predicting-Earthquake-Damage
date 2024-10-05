from sklearn.preprocessing import LabelEncoder

def basic_encoding(df):
    
    """a for loop iterates over all the columns with object
    data type, and then creates a new column into the copied
    data frame, with the suffix "_encoded" appended to the
    original column name"""

    # Selecting all the columns that have the object data type
    object_cols = df.select_dtypes(include = ['object']).columns
    
    # Copying the data, so the old data does not change
    proc_df = df.copy()

    # Creating a LabelEncoder object
    encoder = LabelEncoder() 
    
    # TODO: different encoding for some location cols
    # geo_level_1_id, geo_level_2_id, geo_level_3_id 
    # numbers indicate locations
    # BUT: no pattern for these location cols
    # encode them as strings to express the relationship
    # target encoding maybe?

    # for loop iterates over cols with object type, replaces all
    # of them with the new encoded cols
    for col in object_cols:
        proc_df[col] = encoder.fit_transform(proc_df[col])
    return proc_df