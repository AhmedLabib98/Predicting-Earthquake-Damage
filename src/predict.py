def predict(model, X_values):
    """
    Applies model to X_values to get predictions
    """

    y_pred = model.predict(X_values)

    return y_pred
