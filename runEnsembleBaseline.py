# runEnsembleLearner.py

# For interacting with and running the ensemble learner.

import os
import sys
import datetime # for tracking run times
import pandas as pd 
from ensembleLearnerPipeline import ensembleLearner
from helperFunctions.dataCleanLoad import *

def main():
    
    ###########################################################
    # Configure run parameters here:

    # Data file path and file names
    dataPath = 'data/input/'
    trainFile = 'train.csv'
    testFile = 'test.csv'
    train_col_file = 'data/ensemble/train_cols.txt'

    # Define the parameter grid
    param_grid = {'submodels__dtr__model__min_samples_split': [2, 10, 20],
                  'submodels__dtr__model__max_depth': [5, 10, 20],
                  'submodels__sgd__model__learning_rate': ['optimal'],
                  'submodels__sgd__model__max_iter': [1000, 10000],
                  'submodels__sgd__model__penalty': ['l1'],
                  'submodels__sgd__model__tol': [0.001, 0.0001],
                 }

    # Pipeline architecture for reference
    #
    # drPipeline = Pipeline([
    #        ('varThresh', VarianceThreshold()),
    #        ('varImp', SelectFromModel(estimator=DecisionTreeRegressor())),
    #    ])
    # stacker = LinearRegression()
    #
    # ensemble = Pipeline([
    #           ('submodels', FeatureUnion([ 
    #               ('dtr', ModelTransformer(DecisionTreeRegressor(random_state=self.randSeed))),
    #               ('sgd', ModelTransformer(SGDRegressor(random_state=self.randSeed))),
    #               ('lr', ModelTransformer(LinearRegression())),
    #           ])),
    #           ('ensemble', stacker)
    #           ])
    #
    # 
    ###########################################################

    print(datetime.datetime.now())

    # Load datasets into memory
    print('Loading datasets...')
    train_df, test_df = dataLoad(dataPath, trainFile, testFile)

    # Prep data for the model and scoring
    print('Prepping train and test data for models...')
    X_train, y_train = dataModelTrainingPrepare(train_df, train_col_file)
    X_test = dataModelScoringPrepare(test_df, train_col_file)

    # Instantiate learner class
    print('Training a model...')
    ensPipelineLearner = ensembleLearner()

    # Train the model
    ensPipelineLearner.buildModel(X_train, y_train, param_grid, 'id')

    # Make predictions on the test data
    # Also writes predictions to file
    print('Making predictions on test data...')
    testPreds = ensPipelineLearner.makePrediction(X_test, ensPipelineLearner.modelObject)

    # Compute drivers and write driver output to file
    print('Getting drivers...')
    drivers = ensPipelineLearner.getDrivers(X_train)

    # Gather performance metrics
    print('Gathering performance metrics...')
    with open('ensemble_report.txt', 'w') as text_file:
        text_file.write('Training R2 score: ' + str(ensPipelineLearner.r2Fit_) + '\n')
        text_file.write('Training MAE score: ' + str(ensPipelineLearner.maeFit_) + '\n') 
        text_file.write('Training run time: ' + str(ensPipelineLearner.fitRunTime_) + ' seconds' + '\n') 
        text_file.write('Best params: ' + str(ensPipelineLearner.modelObject.best_params_) + '\n') 
        text_file.write('Best estimator: ' + str(ensPipelineLearner.modelObject.best_estimator_) + '\n') 

    print('Job complete!')

if __name__ == '__main__':
    main()
