from sklearn.model_selection import train_test_split

def make_split(train_data, label_data):
    """
    Split values and label dataframes
    """
    X_train, X_valid, y_train, y_valid  = train_test_split(train_data, label_data, stratify=label_data, test_size=0.2, random_state=42)
    return X_train, X_valid, y_train, y_valid 