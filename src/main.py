# Import packages
from data_loading import data_loading
from concate import concatenate

# Load the data
train_values, train_labels, test_values = data_loading()

# Concatenate the data
features = concatenate(train_values, test_values)

# Encoding
# TODO: Encoding function

# Train a model
# TODO: Train a model

# Make predictions
# TODO: Make predictions

# Create submission
# TODO: Create submission
