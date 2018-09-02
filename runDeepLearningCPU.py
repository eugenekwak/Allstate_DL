# deepLearning.py
# For interacting with and running the deep learner.

import os
import sys
import datetime as dt 
import time
import pandas as pd 
import math
import seaborn as sns
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import adam, SGD
from deepLearnerPipelineCPU import deepLearner
from helperFunctions.dataCleanLoad import *

def main():
    
    # Configure run parameters here:

    # Data file path and file names
    dataPath = 'data/input/'
    trainFile = 'train.csv'
    testFile = 'test.csv'
    train_col_file = 'models/deep_learning/train_cols.txt'
    model_file = 'models/deep_learning/weights.best.from_cpu_'+dt.datetime.now().strftime('%Y_%m_%d')+'.hdf5'
    dr_pipeline_file = 'models/deep_learning/cpu_drPipeline_'+dt.datetime.now().strftime('%Y_%m_%d')+'.pkl'

    # Network training configuration
    dr_threshold = '0.75*mean'
    epochs=10
    batch_size=32
    val_frac=0.15

    # Define the network architecture
    def createModel(input_size):
        model = Sequential()
        model.add(Dense(128, input_dim=input_size, kernel_initializer='normal', activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae', 'mse'])
        return model
    
    # Learning rate decay function
    def stepDecay(epoch):
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 5.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    ######################################################

    print(dt.datetime.now())

    # Load datasets into memory
    print('Loading datasets...')
    train_df, test_df = dataLoad(dataPath, trainFile, testFile)

    # Prep data for the model and scoring
    print('Prepping train and test data for models...')
    X_train, y_train = dataModelTrainingPrepare(train_df, train_col_file)
    X_test = dataModelScoringPrepare(test_df, train_col_file)

    # Instantiate learner class
    print('Training a model...')
    dl = deepLearner()

    # Train the model
    dl.buildModel(X_train, y_train, network_function=createModel, decay_function=stepDecay, val_frac=val_frac, epochs=epochs, batch_size=batch_size, dr_threshold=dr_threshold)

    # Make predictions on the test data
    # Also writes predictions to file
    print('Making predictions on test data...')
    preds_startTime = time.time()
    dl.makePrediction(X_test, model_file=model_file, dr_pipeline_file=dr_pipeline_file)
    preds_endTime = time.time() - preds_startTime

    # Get drivers
    print('Getting model drivers...')
    dl.getDrivers(X_train, dr_pipeline_file=dr_pipeline_file)

    # Visualize training performance across epochs.
    sns.set_style('whitegrid')
    plt.plot(dl.modelObject.history['mean_absolute_error'], label='train')
    plt.plot(dl.modelObject.history['val_mean_absolute_error'], label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE per Epoch')
    plt.legend()
    plt.savefig('models/deep_learning/train_performance.png')

    # Gather performance metrics
    print('Gathering performance metrics...')
    with open('models/deep_learning/cpu_deep_learning_report_'+dt.datetime.now().strftime('%Y_%m_%d')+'.txt', 'w') as text_file:
        text_file.write('Training R2 score: ' + str(dl.r2Fit_) + '\n')
        text_file.write('Training MAE score: ' + str(dl.maeFit_) + '\n') 
        text_file.write('Training run time: ' + str(dl.fitRunTime_) + ' seconds' + '\n')
        text_file.write('Prediction run time: ' + str(round(preds_endTime,6)) + ' seconds' + '\n')

    print('Job complete!')

if __name__ == '__main__':
    main()