from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

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

    # test-train split 
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.2, random_state = 1)
    
    # fit pipeline
    pipe.fit(train_X, train_y)

    return pipe