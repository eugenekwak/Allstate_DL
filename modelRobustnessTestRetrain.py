# modelRobustnessTestTraining.py

# Test model robustness by bootstrapping the training on many samples of the train data.

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
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

# set paths
model_file = 'models/deep_learning/weights.best.from_gpu_2018_09_03.hdf5'
dr_pipeline_file = 'models/deep_learning/gpu_drPipeline_2018_09_03.pkl'
train_col_file = 'models/deep_learning/train_cols.txt'
dataPath = 'data/input/'
trainFile = 'train.csv'
testFile = 'test.csv'

# network training configurations
epochs=500
batch_size=32
val_frac=0.20

# Define the network architecture
def createModel(input_size):
    model = Sequential()
    model.add(Dense(1024, input_dim=input_size, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25, seed=10))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25, seed=10))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.15, seed=10))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae', 'mse'])
    return model
    
# Learning rate decay function
def stepDecay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

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
r2_train_list = []
r2_val_list = []
mae_train_list = []
mae_val_list = []

# create empty dataframe for final results
bootstrap_df = pd.DataFrame()

# retrain the model 50 times with different samples 
for n in range(1, 51):

    # print iteration
    print('Begin iteration: ' + str(n) + '...')

    # sample the data with replacement
    X_train_boot = X_train.sample(frac=0.5, replace=True, random_state=n).copy()

    # separate features from label
    y_train_sample = X_train_boot['log_loss']
    X_train_sample = X_train_boot.drop('log_loss', axis=1)
    
    # apply feature selection
    X_transformed = drPipe.transform(X_train_sample) 

    # Partition validation set
    X_train_, X_val_, y_train_, y_val_ = train_test_split(X_transformed, y_train_sample, test_size=val_frac)

    # create the model
    model_network = createModel(X_train_.shape[1])

    # Model checkpoints
    checkpointer = ModelCheckpoint(filepath='models/deep_learning/robustness_test.hdf5', verbose=1, save_best_only=True)
    
    # Learning rate scheduler
    learningRate = LearningRateScheduler(stepDecay)

    # early stopping criteria
    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    # fit the model
    model_network.fit(X_train_, y_train_, validation_split=val_frac, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, learningRate, earlystopper], verbose=1)

    # make predictions on the sample using best model
    loaded_model = load_model('models/deep_learning/robustness_test.hdf5')
    preds_train = loaded_model.predict(X_train_)
    preds_val = loaded_model.predict(X_val_)

    # score the sample predictions
    r2_train = r2_score(y_train_, preds_train)
    r2_val = r2_score(y_val_, preds_val)
    mae_train = mean_absolute_error(np.exp(y_train_), np.exp(preds_train))
    mae_val = mean_absolute_error(np.exp(y_val_), np.exp(preds_val))

    # store results to lists
    iter_list.append(n)
    r2_train_list.append(r2_train)
    r2_val_list.append(r2_val)
    mae_train_list.append(mae_train)
    mae_val_list.append(mae_val)

# add results to df
bootstrap_df['iteration'] = iter_list
bootstrap_df['r2_train'] = r2_train_list 
bootstrap_df['r2_val'] = r2_val_list 
bootstrap_df['mae_train'] = mae_train_list 
bootstrap_df['mae_val'] = mae_val_list 

# write df
bootstrap_df.to_csv('modelRobustnessTestRetrain.csv', index=False)
