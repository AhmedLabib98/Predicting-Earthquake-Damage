def predict(model, X_data):
    """
    This function takes the input as the model, and the X data.
    It prints out the predicted values for the target variable.
    In this case, the target variable is 
    """
    test_values = X_data[X_data["type"] == 0].drop(columns=["type"])
    train_values = X_data[X_data["type"] == 1].drop(columns=["type"])

    y_pred_train = model.predict(train_values)
    y_pred_test = model.predict(test_values)

    return y_pred_train, y_pred_test
 