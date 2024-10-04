from sklearn.metrics import f1_score
from src.predict import predict 

def f1(train_labels, predictions):
    f1_score(train_labels, predictions, average = 'micro')

    return f1_score

