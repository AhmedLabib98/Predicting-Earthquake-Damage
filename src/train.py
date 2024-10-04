from sklearn.pipeline import Pipeline

def train_model(train_X, train_y, model, cols=None): 

    """
    Trains a model using training data with the option 
    to consider selected columns.

    train_data: dataframe of training data
    cols: optional list of desired columns to be considered
    model: a model to be trained
    """

    # optional: select columns before training
    if cols is not None:
        train_X = train_X.loc[:, cols]

    # pipeline - now basic, may add more steps here
    pipe = Pipeline([('model', model)])

    # fit pipeline
    pipe.fit(train_X,train_y)
