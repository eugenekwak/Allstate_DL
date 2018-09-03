# deepLearner.py
# Deep learning model
# Last updated 2018-08-27

import os
# Keras with tensorflow backend automatically uses all available GPUs...
# https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will?answertab=oldest#tab-top
# uncomment below to use CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import time
import math
import json
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

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
        self.drPipeline = None

    def buildModel(self, X_train, y_train, network_function, decay_function, val_frac=0.15, epochs=50, batch_size=10, dr_threshold='0.75*mean'):
        '''
        Trains a network defined by a function.

        The following model attributes are stored:
            - self.X_train_cols_  : List object containing feature names.
            - self.modelObject    : Network train history.
            - self.drPipeline     : Fitted dimension reduction pipeline
            - self.fitRunTime_    : Float object containing total run time in seconds.
            - self.maeFit_        : Best mean absolute error from validation.
            - self.r2Fit_         : Best r2_score from validation.

        Arguments
        ----------------
        @ X_train: @ X_train: Pandas data frame for feature space in training data.

        @ y_train: Pandas series containing labels.
        @ network_function: Function defining Keras neural network.

        @ decay_function: Function defining the learning rate decay.

        @ val_frac: Fraction of training data to partition for validation.

        @ epochs: Number of epochs.

        @ batch_size: Batch size per epoch.

        @ dr_threshold: Threshold for determining features to include in the model.
                        See scikit-learn documentation for SelectFromModel().

        '''

        # Set random seed for reproducibility
        np.random.seed(self.randSeed)
        
        startTime = time.time()

        print('Reducing dimensions...')
        
        # Feature selection pipeline
        self.drPipeline = Pipeline([
            ('varImp', SelectFromModel(estimator=DecisionTreeRegressor(), threshold=dr_threshold)),
            ])

        # Fit DR pipeline and convert label to numpy array
        X_train_reduced = self.drPipeline.fit_transform(X_train, y_train)
        y_train = np.array(y_train)
        
        # Partition validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train_reduced, y_train, test_size=val_frac)

        # Get network architecture
        self.networkFunc = network_function(X_train.shape[1])

        # Write model summary to file
        with open('models/deep_learning/gpu_model_summary_'+dt.datetime.now().strftime('%Y_%m_%d')+'.txt','w') as fh:
            self.networkFunc.summary(print_fn=lambda x: fh.write(x + '\n'))

        # Learning rate scheduler for decay
        learningRate = LearningRateScheduler(decay_function)

        # Model checkpointer for storing best weights
        checkpointer = ModelCheckpoint(filepath='models/deep_learning/weights.best.from_gpu_'+dt.datetime.now().strftime('%Y_%m_%d')+'.hdf5', verbose=1, save_best_only=True)

        # Early stopping
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=1, mode='auto')

        # Train the model
        print('Training the network...')
        self.modelObject = self.networkFunc.fit(X_train, y_train, validation_split=val_frac, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, learningRate, earlystopper], verbose=1)

        # Get training accuracies
        inPreds = self.modelObject.model.predict(X_train)
        valPreds = self.modelObject.model.predict(X_val)
        self.r2Fit_ = r2_score(y_train, inPreds)
        self.maeFit_ = mean_absolute_error(np.exp(y_train), np.exp(inPreds))
        self.r2Val_ = r2_score(y_val, valPreds)
        self.maeVal_ = mean_absolute_error(np.exp(y_val), np.exp(valPreds))  

        # Save down DR pipeline for production scoring
        joblib.dump(self.drPipeline, 'models/deep_learning/gpu_drPipeline_'+dt.datetime.now().strftime('%Y_%m_%d')+'.pkl') 

        # End run and save run time in seconds
        self.fitRunTime_ = time.time() - startTime
        print('Training run is complete with', round(self.fitRunTime_, 6), 'seconds elapsed.')

    def makePrediction(self, X_test, model_file, dr_pipeline_file):
        '''
        Returns predictions using the model and writes output to file as a csv.
        Run dataModelScoringPrepare() to reconcile datasets.

        Arguments
        ----------------
        @ X_test: Pandas data frame for feature space of the test data.

        @ model_file: Fitted model from *.hdf5 file.
                      ex) 'models/deep_learning/weights.best.from_gpu_YYYY_MM_DD.hdf5'

        @ dr_pipeline_file: Path and file name for the fitted dimension reduction pipeline.
                            ex) 'models/deep_learning/gpu_drPipeline_YYYY_MM_DD.pkl'

        Returns
        ----------------
        Pandas dataframe of predictions.

        '''

        # Get fitted DR pipeline and transform new data.
        drPipe = joblib.load(dr_pipeline_file)
        X_test_transformed = drPipe.transform(X_test)

        # Load model with trained weights.
        model = load_model(model_file)

        # Store row index and predictions to a data frame
        pred_df = pd.DataFrame()
        pred_df['id'] = X_test.index
        pred_df['loss'] = np.exp(model.predict(X_test_transformed))

        # Write prediction data frame to disk
        pred_df.to_csv('data/predictions/pred_gpu_dl_'+dt.datetime.now().strftime('%Y_%m_%d')+'.csv', index=False)

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
                            ex) 'models/deep_learning/gpu_drPipeline_YYYY_MM_DD.pkl'

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
            fi_df.to_csv('models/deep_learning/gpu_feature_importances_'+dt.datetime.now().strftime('%Y_%m_%d')+'.csv', index=False)
            
            return fi_df
        except:
            print('Model must be fit first. Run method buildModel().')