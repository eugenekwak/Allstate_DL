# deepLearning.py
# For interacting with and running the deep learner.

import os
import sys
import datetime # for tracking run times
import pandas as pd 
from deepLearnerPipeline import deepLearner
from helperFunctions.dataCleanLoad import *

def main():
    
    # Configure run parameters here:

    # Data file path and file names
    dataPath = 'data/input/'
    trainFile = 'train.csv'
    testFile = 'test.csv'
    epochs = 500
    batch_size = 100

    # Define the network architecture
    def createModel():
        model = Sequential()
        model.add(Dense(128, input_dim=10, kernel_initializer='normal', activation='relu'))
        model.add(Dense(64,kernel_initializer='normal', activation='relu'))
        model.add(Dense(32,kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model

    ######################################################

    print(datetime.datetime.now())

    # Load datasets into memory
    print('Loading datasets...')
    train_df, test_df = dataLoad(dataPath, trainFile, testFile)

    # Prep data for the model and scoring
    print('Prepping train and test data for models...')
    X_train, y_train = dataModelTrainingPrepare(train_df)
    X_test = dataModelScoringPrepare(test_df)

    # Instantiate learner class
    print('Training a model...')
    deepLearner = deepLearner()

    # Train the model
    deepLearner.buildModel(X_train, y_train, createModel,  batch_size=batchs_size, epochs=epochs)

    # Make predictions on the test data
    # Also writes predictions to file
    print('Making predictions on test data...')
    testPreds = deepLearner.makePrediction(X_test, deepLearner.modelObject)

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
