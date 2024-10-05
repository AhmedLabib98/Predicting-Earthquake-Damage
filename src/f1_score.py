from sklearn.metrics import f1_score

def f1(df, target_col, predictions):
    """ Returns the micro-f1 score"""

    score = f1_score(
        df[target_col],
        predictions,
        average = 'micro'
    )
    
    return score

