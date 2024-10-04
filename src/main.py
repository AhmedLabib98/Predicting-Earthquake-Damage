# Import packages
from src.data_loading import data_loading
from src.concate import concatenate
from src.encoding import basic_encoding
from src.train import train_model
from src.selection import select_features
from src.predict import predict
from src.f1_score import f1
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

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

# Converting y_pred_test to int 
y_pred_test = y_pred_test.astype(int)

# Converting the predictions to a pandas DataFrame
y_pred_test_df = pd.DataFrame(y_pred_test)
"""
#y_pred_test_with_id = pd.concat([selected_features_df['building_id'].reset_index(drop = True)])
Building_id = encod_features.loc[encod_features.type==0, "building_id"]

y_pred_test.concat(Building_id)

#y_pred_test.concat('building_id')

# Create submission
frames = [Building_id, y_pred_test_df]
result = pd.concat(frames)
print(result)
"""

y_pred_test_df_flat = y_pred_test_df.values.ravel()

submission = pd.DataFrame({'building_id': test_values['building_id'], 'damage_grade': y_pred_test_df_flat})

pd.DataFrame.to_csv(submission, '1stSub.csv', index = False)