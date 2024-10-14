from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 

pipe = Pipeline([('model', RandomForestClassifier())])