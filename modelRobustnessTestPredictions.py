# modelRobustnessTestPredictions.py

# Test model robustness by bootstrapping predictions of the final model.

import os
import sys
import time
import math
import json
import pandas as pd
import numpy as np
import datetime as dt
from helperFunctions.dataCleanLoad import *
from keras.models import Sequential, load_model
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# set paths
model_file = 'models/deep_learning/weights.best.from_gpu_2018_09_03.hdf5'
dr_pipeline_file = 'models/deep_learning/gpu_drPipeline_2018_09_03.pkl'
train_col_file = 'models/deep_learning/train_cols.txt'
dataPath = 'data/input/'
trainFile = 'train.csv'
testFile = 'test.csv'

# read in data and prepare for model
train_df, test_df = dataLoad(dataPath, trainFile, testFile)

# load dimension reduction pipeline
drPipe = joblib.load(dr_pipeline_file)

# load in the model
model = load_model(model_file)

# prep data
X_train, y_train = dataModelTrainingPrepare(train_df, train_col_file)

# recombine features and labels
X_train['log_loss'] = y_train

# create lists to store bootstrap sample results
iter_list = []
r2_list = []
mae_list = []

# create empty dataframe for final results
bootstrap_df = pd.DataFrame()

for n in range(1, 1001):

    # print iteration
    print(n)

    # sample the data with replacement
    X_train_boot = X_train.sample(frac=0.25, replace=True, random_state=n).copy()

    # separate features from label
    y_train_sample = X_train_boot['log_loss']
    X_train_sample = X_train_boot.drop('log_loss', axis=1)
    
    # apply feature selection
    X_transformed = drPipe.transform(X_train_sample) 

    # make predictions on the sample
    preds = model.predict(X_transformed)

    # score the sample predictions
    r2 = r2_score(y_train_sample, preds)
    mae = mean_absolute_error(np.exp(y_train_sample), np.exp(preds))

    # store results to lists
    iter_list.append(n)
    r2_list.append(r2)
    mae_list.append(mae)

# add results to df
bootstrap_df['iteration'] = iter_list
bootstrap_df['r2'] = r2_list 
bootstrap_df['mae'] = mae_list 

# write df
bootstrap_df.to_csv('modelRobustnessTestPredictions.csv', index=False)
