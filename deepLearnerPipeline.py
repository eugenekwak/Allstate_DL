# deepLearner.py
# Deep learning model
# Last updated 2018-08-27

import sys
import os
import time
import pickle
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import adam, SGD
from sklearn.model_selection import cross_val_score, KFold

class deepLearner:
    '''
    A wrapper to train a network with Keras.

    '''
    def __init__(self):

        self.randSeed = 10
        self.fitRunTime_ = 0.0
        self.r2Fit_ = 0.0
        self.maeFit_ = 0.0
        self.modelObject = None
        self.idVar_ = None


    def buildModel(self, X_train, y_train, network_function, epochs, batch_size):
        '''
        Fits a model to the data using 5 fold cross validation.

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

        self.modelObject = network_function()

        # write model summary to file
        with open('models/deep_learning/model_summary.txt', 'w') as text_file:
            text_file.write(self.modelObject.summary())
        
        # Feature selection pipeline
        drPipeline = Pipeline([
             ('varThresh', VarianceThreshold(.0001)),
             ('varImp', SelectFromModel(estimator=DecisionTreeRegressor(), '0.5*mean')),
        ])

        # Features that made it through dimension reduction
        X_train_reduced = drPipeline.fit_transform(X_train, y_train)
        vt_cols = [X_train.columns[x] for x in range(len(X_train.columns)) if drPipeline.named_steps['varThresh'].get_support()[x]]
        final_cols = [vt_cols[x] for x in range(len(vt_cols)) if drPipeline.named_steps['varImp'].get_support()[x]]
        final_cols_df = pd.DataFrame(final_cols, columns=['var'])

        self.modelObject.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # Training accuracies
        inPreds = self.modelObject.predict(X_train)
        self.r2Fit_ = r2_score(y_train, inPreds)
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
