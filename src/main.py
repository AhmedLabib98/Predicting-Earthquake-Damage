# Import packages
from src.data_loading import data_loading
from src.concate import concatenate
from src.encoding import basic_encoding
from src.train import train_model
from sklearn.ensemble import RandomForestClassifier

# Load the data
train_values, train_labels, test_values = data_loading()

# Concatenate the data
features = concatenate(train_values, test_values)

# Encoding
encod_features = basic_encoding(features)

# Train a model
my_model = train_model(encod_features.loc[encod_features.type=='TRAIN', :],
                        train_labels, 
                        model=RandomForestClassifier(), 
                        cols=encod_features.drop(['type','building_id'].columns)

# Make predictions
# TODO: Make predictions

# Create submission
# TODO: Create submission
