from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from sklearn.model_selection import GridSearchCV
from cv_params import cv_params
from cv_metric import cv_metric

def train_pipe():
    """
    Trains a Random Forest using GridSearchCV
    """
    # add steps to a pipeline e.g., a model
    pipe = Pipeline([('model', RandomForestClassifier())])

    # grid search cross-validation 5-folds
    # using f1 score for hyperparameter tuning
    search = GridSearchCV(estimator=pipe, 
                        param_grid=cv_params, # hyperparameter list
                        scoring=cv_metric,
                        cv=5,
                        refit=True, # after the best model is selected, fit it using all folds
                        verbose=3, # level of information on progress
                        n_jobs=-1) # use all cores for computation  
    return search