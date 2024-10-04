# Import packages
from data_loading import data_loading
from concate import concatenate
from encoding import basic_encoding
from train import train_model
from selection import select_features
from predict import predict
from f1_score import f1
from sklearn.ensemble import RandomForestClassifier

# Load the data
train_values, train_labels, test_values = data_loading()

# Concatenate the data
features = concatenate(train_values, test_values)

# Encoding
encod_features = basic_encoding(features)

# Select features
selected_features_df = select_features(
    encod_features,
    columns_to_drop=['building_id']
)
selected_labels = select_features(
    train_labels,
    columns_to_keep=['damage_grade']
)

# Train a model
my_model = train_model(
    values=selected_features_df,
    label=selected_labels, 
    model=RandomForestClassifier(), 
)

# Make predictions
y_pred_train, y_pred_test = predict(my_model, selected_features_df)

# Evaluate the model
f1(train_labels=train_labels, predictions=y_pred_train)

# Create submission
# TODO: Create submission
