# Import packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from splitting import make_split 
from data_loading import data_loading
from encoding import basic_encoding
from target_encoding import target_encoding_train, target_encoding_test
from selection import select_features
from train import train_model
from predict import predict
from f1_score import f1
from convert_to_pd import convert_to_pd
from concate import concat_pd
from submission import submit


# Load the data
train_values, train_label, test_values = data_loading()

# Label encoding
encod_train_values = basic_encoding(train_values)
encod_test_values = basic_encoding(test_values)

# Target encoding
t_encoded_train_values, t_encoder  = target_encoding_train( df = train_values, y = train_label)
t_encoded_test_values = target_encoding_test( df = test_values, T_encoder = t_encoder)

# Select features
selected_train_values = select_features(
    df=encod_train_values,
    columns_to_drop=['building_id'] 
    )

selected_train_label = select_features(
    df=train_label,
    columns_to_drop=['building_id']
)

selected_test_values = select_features(
    df=encod_test_values,
    columns_to_drop=['building_id']
)

# Train a model in full dataset
full_model = train_model(
    train_data=selected_train_values,
    label_data=selected_train_label, 
    model=RandomForestClassifier(), 
)

# Make predictions
y_pred_train = predict(model = full_model, X_values = selected_train_values)
y_pred_test = predict(model = full_model, X_values = selected_test_values)

# Evaluate the model on the training
f1(df=train_label, target_col="damage_grade", predictions=y_pred_train)

# Convert to pd
y_pred_test_pd = convert_to_pd(y_pred_test)

# Concat building_id and label cols into one df
final_pd = concat_pd(df_id=test_values, df_label=y_pred_test_pd)

# Submit
submit(df=final_pd, file_name='data/04_submissions/second_sub.csv')
