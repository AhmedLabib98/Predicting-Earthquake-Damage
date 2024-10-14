# Import packages
import pandas as pd
from splitting import make_split 
from data_loading import data_loading
from label_encoding import label_encoding
# from selection import select_features
from train import train_model
from pipeline import pipe
from predict import predict
from f1_score import f1
from convert_to_pd import convert_to_pd
from concate import concat_pd
from submission import submit
from potential_t_enc import target_encoding

# Load the data
train_X, train_y, test_X = data_loading()

# Target encoding 
target_enc_train_X, target_enc_test_X = target_encoding(train_X, test_X, train_y)

# Label encoding for all string columns in train_X and test_X
# Excluding geo_level_1_id, geo_level_2_id, geo_level_3_id
en_train_X = label_encoding(train_X)
en_test_X = label_encoding(test_X)


# Split train into cv_train and cv_validation
# we split the training set into a 80% (cv_train_X) and 20% (cv_valid_X)
cv_train_X, cv_valid_X, cv_train_y, cv_valid_y = make_split(train_data=en_train_X,
                                                            label_data=train_y)

# Train a model using the cv_train_X (which is 80% of the original training set: train_X)
cv_pipe = train_pipe()
split_model = cv_pipe.fit(cv_train_X,cv_train_y.values.ravel())

# Get predictions for target columns for each of the two splits
# to compare them with the observed label columns and generate the f1-score below
y_pred_cv_train = predict(model = split_model, X_values = cv_train_X)
y_pred_cv_valid = predict(model = split_model, X_values = cv_valid_X) 

# Get f1 score comparing observed vs predicted target column for each split
# if the f1-scores are close to each other, then no overfitting
f1_cv_train = f1(df=cv_train_y, target_col="damage_grade", predictions=y_pred_cv_train)
f1_cv_valid = f1(df=cv_valid_y, target_col="damage_grade", predictions=y_pred_cv_valid)

print(f1_cv_train)
print(f1_cv_valid)


# Now get the previous model (it was the best one during cv)
# and fit it using the full 100% training set
full_model = cv_pipe.best_estimator_.fit(X=en_train_X, y=train_y)

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
