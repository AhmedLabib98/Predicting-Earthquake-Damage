from sklearn.metrics import f1_score

def f1(train_labels, predictions):

    number = f1_score(
        train_labels["damage_grade"],
        predictions,
        average = 'micro'
    )
    return number

