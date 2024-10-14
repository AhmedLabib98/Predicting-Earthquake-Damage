# Import packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from splitting import make_split 
from data_loading import data_loading
from label_encoding import label_encoding
from selection import select_features
from train import train_model
from predict import predict
from f1_score import f1
from convert_to_pd import convert_to_pd
from concate import concat_pd
from submission import submit

# Load the data
train_X, train_y, test_X = data_loading()

# Label encoding for all string columns in train_X and test_X
en_train_X = label_encoding(train_X)
en_test_X = label_encoding(test_X)

# Split train into cv_train and cv_validation
cv_train_X, cv_valid_X, cv_train_y, cv_valid_y = make_split(train_data=en_train_X,
                                                            label_data=train_y)

# Train a model in split dataset
split_model = train_model(
    train_data=cv_train_X,
    label_data=cv_train_y, 
    model=RandomForestClassifier(), 
)

# Get predicted label column
y_pred_cv_train = predict(model = split_model, X_values = cv_train_X)
y_pred_cv_valid = predict(model = split_model, X_values = cv_valid_X)

# Get f1 score comparing observed y label and predicted label for split dataset
f1_cv_train = f1(df=cv_train_y, target_col="damage_grade", predictions=y_pred_cv_train)
f1_cv_valid = f1(df=cv_valid_y, target_col="damage_grade", predictions=y_pred_cv_valid)

print(f1_cv_train)
print(f1_cv_valid)


# Now train a model in full dataset
full_model = train_model(
    train_data=en_train_X,
    label_data=train_y, 
    model=RandomForestClassifier(), 
)

# Get predicted label column for full dataset
y_pred_train = predict(model = full_model, X_values = en_train_X)
y_pred_test = predict(model = full_model, X_values = en_test_X)

# Get f1 score comparing observed y label and predicted label for full dataset
f1_train = f1(df=train_y, target_col="damage_grade", predictions=y_pred_train)

print(f1_train)


# Convert to pd
y_pred_test_pd = convert_to_pd(y_pred_test)

# Concat rownames of df_id with the pd dataset y_pred_test_pd
final_pd = concat_pd(df_id=en_test_X, df_label=y_pred_test_pd)

# Submit
submit(df=final_pd, file_name='data/04_submissions/second_sub.csv')
