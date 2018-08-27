# learnerFunctions.py
# Helper functions for sklearn.
# Last updated: 2018-08-20

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ModelTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X)).values

