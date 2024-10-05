from sklearn.pipeline import Pipeline

def train_model(train_X, train_y, model): 

    """
    Trains a model using training data with the option 
    to consider selected columns.

    train_data: dataframe of training data
    cols: optional list of desired columns to be considered
    model: a model to be trained
    """

    # pipeline - now basic, may add more steps here
    pipe = Pipeline([('model', model)])

    # fit pipeline
    pipe.fit(train_X, train_y)

    return pipe