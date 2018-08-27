# ensembleLearner.py
# Benchmark model
# Last updated 2018-08-25

import sys
import os
import time
import pickle
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

class ModelTransformer(BaseEstimator, TransformerMixin):
    '''
    Turns estimators into transformers so that predicted values returned.
    These predictions can be ensembled via stacking.
    '''
    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X)).values

class ensembleLearner:
    '''
    A baseline model to benchmark deep learning architectures against.

    This class is only tested against Pandas dataframes.

    '''

    def __init__(self):

        self.randSeed = 10
        self.fitRunTime_ = 0.0
        self.r2Fit_ = 0.0
        self.maeFit_ = 0.0
        self.modelObject = None
        self.idVar_ = None


    def buildModel(self, X_train, y_train, param_grid, idVar):
        '''
        Fits a model to the data using 5 fold cross validation and grid search.

        The following model attributes are stored:
            - self.X_train_cols_  : List object containing feature names.
            - self.param_grid     : Dict object containing search grid.
            - self.modelObject    : Fitted model pipeline object.
            - self.fitRunTime_    : Float object containing total run time in seconds.
            - self.maeFit_        : Best mean absolute error from cross-validation.
            - self.r2Fit_         : Best r2_score from cross-validation.

        Arguments
        ----------------
        @ X_train: Pandas data frame for feature space.

        @ y_train: Pandas series containing labels.

        @ param_grid: Dictionary containing parameters for the ensemble.
            Example:
            param_grid = {'dr__varThresh__threshold': [0.01, 0.001],
                          'dr__varImp__threshold': ['0.25*mean', '0.75*mean', '1.25*mean'],
                          'submodels__dtr__model__max_depth': [5, 10, 15, 20, 25],
                          }
              

        Returns
        ----------------
        Pandas dataframe.

        '''
        self.X_train_cols_ = list(X_train.columns)
        self.param_grid = param_grid
        self.idVar_ = idVar

        startTime = time.time()
        
        # Feature selection pipeline
        drPipeline = Pipeline([
             ('varThresh', VarianceThreshold()),
             ('varImp', SelectFromModel(estimator=DecisionTreeRegressor())),
        ])

        # Ensembler
        stacker = LinearRegression()

        # Ensemble pipeline
        ensemble = Pipeline([
            ('dr', drPipeline),
            ('submodels', FeatureUnion([ 
                ('dtr', ModelTransformer(DecisionTreeRegressor(random_state=self.randSeed))),
                ('gbr', ModelTransformer(GradientBoostingRegressor(random_state=self.randSeed))),
                ('lr', ModelTransformer(Lasso(random_state=self.randSeed))),
            ])),
            ('ensemble', stacker)
            ])

        # Fit grid search using params and ensemble pipeline
        ensembleGrid = GridSearchCV(ensemble, param_grid=param_grid, cv=5, scoring='r2', refit=True, n_jobs=3, verbose=3)
        self.modelObject = ensembleGrid.fit(X_train, y_train)

        # Training accuracies
        inPreds = self.modelObject.predict(X_train)
        self.r2Fit_ = ensembleGrid.best_score_
        self.maeFit_ = mean_absolute_error(np.exp(y_train), np.exp(inPreds))  

        # End run and save run time in seconds
        self.fitRunTime_ = time.time() - startTime

        print('Training run is complete with', round(self.fitRunTime_, 6), 'seconds elapsed.')
        p = pickle.dumps(ensembleGrid)
        print('Model object is', round(sys.getsizeof(p)/1000000, 4), 'Mb in size.')
        joblib.dump(self.modelObject, 'models/ensemble/ensemble_model_'+dt.datetime.now().strftime('%Y_%m_%d')+'.pkl') 

    def makePrediction(self, X_test, model):
        '''
        Returns predictions using the model and writes output to file as a csv.

        Run dataModelScoringPrepare() to reconcile datasets.

        Arguments
        ----------------
        @ X_test: Pandas data frame for feature space of the test data.

        @ model: Fitted pipeline model object.

        Returns
        ----------------
        Pandas dataframe.

        '''

        # Store row index and predictions to a data frame
        pred_df = pd.DataFrame(columns=['id','preds'])
        pred_df['id'] = X_test.index
        pred_df['preds'] = np.exp(model.predict(X_test))

        # Write prediction data frame to disk
        pred_df.to_csv('data/predictions/pred_ensemble'+dt.datetime.now().strftime('%Y_%m_%d')+'.csv', index=False)

        # Return prediction data frame for analysis
        return pred_df

    
    def getDrivers(self, X_train):
        '''
        Returns a Pandas dataframe containing the features and feature importance
        scores. It is also sorted in descending order of the feature importances.

        Arguments
        ----------------
        @ X_train: Pandas data frame for feature space.

        Returns
        ----------------
        Pandas dataframe.

        '''
        
        # Ensure model has been built first
        try:
            # grab feature importance scores
            varThreshIndex = self.modelObject.best_estimator_.named_steps['dr'].named_steps['varThresh'].get_support(indices=True)

            X_train_cols_ = X_train.iloc[:, list(varThreshIndex)].columns
            
            fi = self.modelObject.best_estimator_.named_steps['dr'].named_steps['varImp'].estimator_.feature_importances_
            
            # save feature importances to a data frame with variable names
            fi_df = pd.DataFrame(
                {'var': list(X_train_cols_),
                'feature_importance': list(fi)
                })

            # sort the data frame desc order of feature importance
            fi_df = fi_df.sort_values(by=['feature_importance'],ascending=False).reset_index(drop=True)
            
            # create feature rank column
            rankNum = list(range(len(fi_df)))
            rankNum = [x + 1 for x in rankNum]
            fi_df['rank'] = rankNum

            # write results to file and return fi_df
            fi_df.to_csv('models/ensemble/feature_importances_'+dt.datetime.now().strftime('%Y_%m_%d')+'.csv', index=False)
            
            return fi_df
        except:
            print('Model must be fit first. Run method buildModel().')

