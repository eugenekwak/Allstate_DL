# ensembleLearner.py
# Benchmark model
# Last updated 2018-08-29

# need to fix parallel processing for grid search cv

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
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split
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
        self.param_grid = None


    def buildModel(self, X_train, y_train, param_grid, val_frac=0.15, dr_threshold='0.75*mean'):
        '''
        Fits a model to the data using 5 fold cross validation and grid search.
        The following model attributes are stored:
            - self.param_grid     : Dict object containing search grid.
            - self.modelObject    : Fitted model pipeline object.
            - self.fitRunTime_    : Float object containing total run time in seconds.
            - self.maeFit_        : Best mean absolute error from cross-validation.
            - self.r2Fit_         : Best r2_score from cross-validation.

        Arguments
        ----------------
        @ X_train: Pandas data frame for feature space in training data.

        @ y_train: Pandas series containing labels.

        @ param_grid: Dictionary containing parameters for the ensemble.
            ex)
            param_grid = {'submodels__dtr__model__max_depth': [5, 10, 15, 20, 25],
                          ...
                          }
        
        @ dr_threshold: Threshold for determining features to include in the model.
                        See scikit-learn documentation for SelectFromModel().
              
        Returns
        ----------------
        Pandas dataframe.
        '''

        np.random.seed(self.randSeed)

        self.param_grid = param_grid

        startTime = time.time()
        
        # Feature selection pipeline
        self.drPipeline = Pipeline([
             ('varImp', SelectFromModel(estimator=DecisionTreeRegressor(), threshold=dr_threshold)),
        ])

        X_train_dr = self.drPipeline.fit_transform(X_train, y_train)

        # Split training and validation data
        X_train, X_val, y_train, y_val = train_test_split(X_train_dr, y_train, test_size=val_frac, random_sate=self.randSeed)

        # Ensembler
        stacker = LinearRegression()

        # Ensemble pipeline
        ensemble = Pipeline([
            ('submodels', FeatureUnion([ 
                ('dtr', ModelTransformer(DecisionTreeRegressor(random_state=self.randSeed))),
                ('sgd', ModelTransformer(SGDRegressor(random_state=self.randSeed))),
                ('lr', ModelTransformer(LinearRegression())),
            ])),
            ('ensemble', stacker)
            ])

        # Fit grid search using params and ensemble pipeline
        ensembleGrid = GridSearchCV(ensemble, param_grid=param_grid, cv=10, scoring='r2', refit=True, verbose=3, n_jobs=2)
        self.modelObject = ensembleGrid.fit(X_train, y_train)

        # Training and validation accuracies
        inPreds = self.modelObject.predict(X_train)
        valPreds = self.modelObject.predict(X_val)
        self.r2Fit_ = r2_score(y_train, inPreds)
        self.maeFit_ = mean_absolute_error(np.exp(y_train), np.exp(inPreds))  
        self.r2Val_ = r2_score(y_val, valPreds)
        self.maeVal_ = mean_absolute_error(np.exp(y_val), np.exp(valPreds))  

        # End run and save run time in seconds
        self.fitRunTime_ = time.time() - startTime

        print('Training run is complete with', round(self.fitRunTime_, 6), 'seconds elapsed.')
        p = pickle.dumps(ensembleGrid)
        print('Model object is', round(sys.getsizeof(p)/1000000, 4), 'Mb in size.')
        joblib.dump(self.modelObject, 'models/ensemble/ensemble_model_'+dt.datetime.now().strftime('%Y_%m_%d')+'.pkl') 

        # Save down DR pipeline for scoring.
        joblib.dump(self.drPipeline, 'models/ensemble/drPipeline_'+dt.datetime.now().strftime('%Y_%m_%d')+'.pkl') 

    def makePrediction(self, X_test, model_object_file, dr_pipeline_file):
        '''
        Returns predictions using the model and writes output to file as a csv.
        Run dataModelScoringPrepare() to reconcile datasets.

        Arguments
        ----------------
        @ X_test: Pandas data frame for feature space of the test data.

        @ model: Fitted pipeline model object.

        @ dr_pipeline_file: Path and file name for the fitted dimension reduction pipeline.
                            ex) 'models/deep_learning/cpu_drPipeline_YYYY_MM_DD.pkl'

        Returns
        ----------------
        Pandas dataframe.
        '''

        # Get fitted pipeline model

        model = joblib.load(model_object_file)

        # Get fitted DR pipeline
        drPipe = joblib.load(dr_pipeline_file)

        # Store row index and predictions to a data frame
        pred_df = pd.DataFrame(columns=['id','loss'])
        pred_df['id'] = X_test.index
        pred_df['loss'] = np.exp(model.predict(drPipe.transform(X_test)))

        # Write prediction data frame to disk
        pred_df.to_csv('data/predictions/pred_ensemble_'+dt.datetime.now().strftime('%Y_%m_%d')+'.csv', index=False)

        # Return prediction data frame for analysis
        return pred_df

    def getDrivers(self, X_train, dr_pipeline_file):
        '''
        Returns a Pandas dataframe containing the features and feature importance
        scores. It is also sorted in ascending order of the feature importances.

        Arguments
        ----------------
        @ X_train: Pandas data frame for feature space in training data.

        @ dr_pipeline_file: Path and file name for the fitted dimension reduction pipeline.
                            ex) 'models/deep_learning/cpu_drPipeline_YYYY_MM_DD.pkl'

        Returns
        ----------------
        Pandas dataframe.
        '''
        
        try:
            # Get DR pipeline object
            drPipe = joblib.load(dr_pipeline_file)

            # grab feature importance scores
            drFeatures = drPipe.named_steps['varImp'].get_support()
            X_train_cols_ = X_train.iloc[:, list(drFeatures)].columns
            fi = drPipe.named_steps['varImp'].estimator_.feature_importances_
            fi = [fi[x] for x in range(len(fi)) if drFeatures[x]]
            
            # save feature importances to a data frame with variable names
            fi_df = pd.DataFrame()
            fi_df['var'] = list(X_train_cols_)
            fi_df['feature_importance'] = list(fi)

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