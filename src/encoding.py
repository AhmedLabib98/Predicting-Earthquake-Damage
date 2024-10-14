from sklearn.preprocessing import LabelEncoder

def basic_encoding(df):
    """
    a for loop iterates over all the columns with object
    data type, and then creates a new column into the copied
    data frame, with the suffix "_encoded" appended to the
    original column name.
    """

    # List of columns to exclude from label-encoding
    exclude_cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

    # Selecting all the columns that have the object data type and
    # excluding target encoding columns 
    object_cols = df.select_dtypes(include = ['object']).columns
    object_cols = [col for col in object_cols if col not in exclude_cols]
    
    # Copying the data, so the old data does not change
    proc_df = df.copy()

    # Creating a LabelEncoder object
    encoder = LabelEncoder()

    # for loop iterates over cols with object type, replaces all
    # of them with the new encoded cols
    for col in object_cols:
        proc_df[col] = encoder.fit_transform(proc_df[col])
    return proc_df