# deepLearner.py
# Deep learning model
# Last updated 2018-08-27

import sys
import os
import time
import math
import json
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib as plt
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

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

    def buildModel(self, X_train, y_train, network_function, decay_function, val_frac=0.15, epochs=50, batch_size=10):
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
        '''
        # set random seed for reproducibility
        np.random.seed(self.randSeed)
        
        startTime = time.time()

        print('Reducing dimensions...')
        
        # Feature selection pipeline
        self.drPipeline = Pipeline([
            ('varImp', SelectFromModel(estimator=DecisionTreeRegressor(), threshold='0.75*mean')),
            ])

        # Features that made it through dimension reduction
        X_train_reduced = self.drPipeline.fit_transform(X_train, y_train)
        y_train = np.array(y_train)

        # get network
        self.networkFunc = network_function(X_train_reduced.shape[1])

        # write model summary to file
        with open('models/deep_learning/model_summary_'+dt.datetime.now().strftime('%Y_%m_%d')+'.txt','w') as fh:
            self.networkFunc.summary(print_fn=lambda x: fh.write(x + '\n'))

        # learning rate scheduler for drop decay
        learningRate = LearningRateScheduler(decay_function)

        # model checkpointer for storing best weights
        checkpointer = ModelCheckpoint(filepath='models/deep_learning/weights.best.from_cpu.hdf5', verbose=1, save_best_only=True)

        # train the model
        print('Training model...')
        self.modelObject = self.networkFunc.fit(X_train_reduced, y_train, validation_split=val_frac, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, learningRate], verbose=1)

        # Training accuracies
        inPreds = self.modelObject.model.predict(X_train_reduced)
        self.r2Fit_ = r2_score(y_train, inPreds)
        self.maeFit_ = mean_absolute_error(np.exp(y_train), np.exp(inPreds))  

        # Save down DR pipeline for scoring.
        joblib.dump(self.drPipeline, 'models/deep_learning/cpu_drPipeline_'+dt.datetime.now().strftime('%Y_%m_%d')+'.pkl') 

        # End run and save run time in seconds
        self.fitRunTime_ = time.time() - startTime

        print('Training run is complete with', round(self.fitRunTime_, 6), 'seconds elapsed.')

    def makePrediction(self, X_test, model, dr_pipeline_file):
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

        drPipe = joblib.load(dr_pipeline_file)

        # Store row index and predictions to a data frame
        pred_df = pd.DataFrame(columns=['id','preds'])
        pred_df['id'] = X_test.index
        pred_df['preds'] = np.exp(model.predict(dr_pipe.transform(X_test)))

        # Write prediction data frame to disk
        pred_df.to_csv('data/predictions/pred_ensemble'+dt.datetime.now().strftime('%Y_%m_%d')+'.csv', index=False)

        # Return prediction data frame for analysis
        return pred_df

    def getDrivers(self, X_train, dr_pipeline_file):
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
        
        try:
            # Load dr pipeline object
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
            fi_df.to_csv('models/deep_learning/cpu_feature_importances_'+dt.datetime.now().strftime('%Y_%m_%d')+'.csv', index=False)
            
            return fi_df
        except:
            print('Model must be fit first. Run method buildModel().')