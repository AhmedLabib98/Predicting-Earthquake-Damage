from sklearn.pipeline import Pipeline

def train_model(values, label, model): 

    """
    Trains a model using training data with the option 
    to consider selected columns.

    train_data: dataframe of training data
    cols: optional list of desired columns to be considered
    model: a model to be trained
    """

    # Select the rows that have the `type` equal to 1
    train_X = values[values["type"] == 1].drop(columns=["type"])

    # pipeline - now basic, may add more steps here
    pipe = Pipeline([('model', model)])

    # fit pipeline
    pipe.fit(train_X, label)

    return pipe