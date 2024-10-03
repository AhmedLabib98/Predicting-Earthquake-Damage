from data_loading import data_loading
from sklearn.preprocessing import LabelEncoder


# Load the data
train_values, train_labels, test_values = data_loading()

# Selecting all the columns that have the object data type
object_cols = train_values(include = ['object']).columns


def basic_encoding():
    encoder = LabelEncoder()

    # Copying the data, so the old data does not change
    train_values_copy = train_values.copy()

    # for loop iterates over cols with object type, replaces all
    # of them with the new encoded cols
    for col in object_cols:
        train_values_copy[col] = encoder.fit_transform(train_values_copy[col])
    
    
    
# FOR LOOP EXPLANATION
# the for loop iterates over all the columns with object 
# data type, and then creates a new column into the copied
# data frame, with the suffix "_encoded" appended to the 
# original column name