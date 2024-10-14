from sklearn.model_selection import GridSearchCV
from pipeline import pipe
from cv_params import cv_params
from cv_metric import cv_metric

def train_model(train_data, label_data): 

    """
    Trains a model 
    using train_data as features 
    and label_data as a target, 
    within a 5-fold cross validation
    using f1 score and gridsearch for hyperparameter tuning
    (remember to use the parameters of selected model)

    train_data: dataframe of training data
    label_data: dataframe of training data
    """
    
    # grid search cross-validation 5-folds
    search = GridSearchCV(estimator=pipe, 
                        param_grid=cv_params, # hyperparameter list
                        scoring=cv_metric,
                        cv=5,
                        refit=True,
                        verbose=3, # level of information on progress
                        n_jobs=-1) # use all cores for computation

    # fit the model
    search.fit(train_data, label_data.values.ravel())

    return search