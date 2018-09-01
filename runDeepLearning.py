# deepLearning.py
# For interacting with and running the deep learner.

import os
import sys
import datetime # for tracking run times
import pandas as pd 
import math
import seaborn as sns
from matplotlib import pyplot as plt
from deepLearnerPipeline import deepLearner
from helperFunctions.dataCleanLoad import *

def main():
    
    # Configure run parameters here:

    # Data file path and file names
    dataPath = 'data/input/'
    trainFile = 'train.csv'
    testFile = 'test.csv'
    train_cols_file = 'train_cols.txt'
    epochs=50
    batch_size=10

    # Define the network architecture
    def createModel(input_size):
        model = Sequential()
        model.add(Dense(128, input_dim=input_size, kernel_initializer='normal', activation='relu'))
        model.add(Dense(64,kernel_initializer='normal', activation='relu'))
        model.add(Dense(32,kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        return model
    
    # Learning rate decay function
    def stepDecay(epoch):
	    initial_lrate = 0.1
	    drop = 0.5
	    epochs_drop = 10.0
	    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	    return lrate

    ######################################################

    print(datetime.datetime.now())

    # Load datasets into memory
    print('Loading datasets...')
    train_df, test_df = dataLoad(dataPath, trainFile, testFile)

    # Prep data for the model and scoring
    print('Prepping train and test data for models...')
    X_train, y_train = dataModelTrainingPrepare(train_df, train_cols_file)
    X_test = dataModelScoringPrepare(test_df, train_cols_file)

    # Instantiate learner class
    print('Training a model...')
    deepLearner = deepLearner()

    # Train the model
    deepLearner.buildModel(X_train, y_train, createModel, epochs=epochs, batch_size=batch_size, lr_init=lr_init, lr_drop=lr_drop, lr_epochs_drop=lr_epochs_drop)

    # Make predictions on the test data
    # Also writes predictions to file
    print('Making predictions on test data...')
    testPreds = deepLearner.makePrediction(X_test, deepLearner.modelObject)

    # Write out training chart
    sns.set_style("whitegrid")
    plt.plot(dlearner.modelObject.history['mean_absolute_error'], label='train')
    plt.plot(dlearner.modelObject.history['val_mean_absolute_error'], label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title('MAE per Epoch')
    plt.legend()
    plt.savefig('foo.png')

    # Gather performance metrics
    print('Gathering performance metrics...')
    with open('ensemble_report.txt', 'w') as text_file:
        text_file.write('Training R2 score: ' + str(deepLearner.r2Fit_) + '\n')
        text_file.write('Training MAE score: ' + str(deepLearner.maeFit_) + '\n') 
        text_file.write('Training run time: ' + str(deepLearner.fitRunTime_) + ' seconds' + '\n') 
        text_file.write('Network architecture: ' + str(deepLearner.modelObject.summary()) + '\n') 

    print('Job complete!')

if __name__ == '__main__':
    main()
