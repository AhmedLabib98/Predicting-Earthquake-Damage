from sklearn.metrics import f1_score, make_scorer

cv_metric = make_scorer(f1_score, 
                        average='micro')