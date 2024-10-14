from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from cv_metric import cv_metric
from cv_params import cv_params

def train_model(train_data, label_data, model): 

    """
    Trains a model 
    using train_data as features 
    and label_data as a target, 
    within a 5-fold cross validation
    using f1 score and gridsearch for hyperparameter tuning
    (remember to use the parameters of selected model)

    train_data: dataframe of training data
    label_data: dataframe of training data
    model: a model to be trained
    """

    # pipeline - now basic, may add more steps here
    pipe = Pipeline([('model', model)])

    # define scoring function
    scoring = cv_metric
    
    # grid search cross-validation 5-folds
    search = GridSearchCV(pipe, 
                        param_grid = cv_params,
                        scoring = scoring,
                        cv = 5,
                        refit = True,
                        verbose = 3,
                        n_jobs=-1) 

    # fit the model
    search.fit(train_data, label_data.values.ravel())

    return search