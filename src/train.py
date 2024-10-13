from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

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

    # define potential parameter values - uncomment several for best results
    param_grid = { 
        # 'model__n_estimators': [25, 50, 100, 150], 
        # 'model__max_features': ['sqrt', 'log2', None], 
        'model__max_depth': [3, 6, 9], 
        # 'model__max_leaf_nodes': [3, 6, 9], 
    } 

    # define scoring function
    scoring = make_scorer(f1_score, 
                        average='micro')
    
    # grid search cross-validation 5-folds
    search = GridSearchCV(pipe, 
                        param_grid,
                        scoring = scoring,
                        cv = 5,
                        refit = True,
                        verbose = 3,
                        n_jobs=-1) 

    # fit the model
    search.fit(train_data, label_data)

    return search