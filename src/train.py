from sklearn.pipeline import Pipeline

def train_model(train_data, label_data, model): 

    """
    Trains a model using training data with the option 
    to consider selected columns.

    train_data: dataframe of training data
    label_data: dataframe of training data
    model: a model to be trained
    """

    # pipeline - now basic, may add more steps here
    pipe = Pipeline([('model', model)])

    # fit pipeline
    pipe.fit(train_data, label_data)

    return pipe