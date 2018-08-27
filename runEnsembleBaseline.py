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

    # Define the parameter grid
    param_grid = {'dr__varThresh__threshold': [0.001, 0.0001],
                  'dr__varImp__threshold': ['0.5*mean' ,'1.5*mean'],
                  'submodels__dtr__model__max_depth': [5, 10, 25],
                  'submodels__dtr__model__min_samples_split': [2, 20],
                  'submodels__gbr__model__learning_rate': [0.1, 0.01, 0.001],
                  'submodels__gbr__model__max_depth': [5, 10, 25],
                  'submodels__gbr__model__n_estimators': [100, 250, 500],
                  'submodels__lr__model__alpha':[0.001, 0.0001],
                  }

    # Pipeline architecture for reference
    #
    # drPipeline = Pipeline([
    #         ('varThresh', VarianceThreshold()),
    #         ('varImp', SelectFromModel(estimator=ExtraTreesRegressor())),
    #    ])
    # stacker = LinearRegression()
    # ensemble = Pipeline([
    #        ('dr', drPipeline),
    #        ('submodels', FeatureUnion([ 
    #            ('dtr', ModelTransformer(DecisionTreeRegressor(random_state=10))),
    #            ('gbr', ModelTransformer(GradientBoostingRegressor(random_state=10))),
    #            ('lr', ModelTransformer(Lasso())),
    #        ])),
    #        ('ensemble', stacker)
    #        ])
    #
    # 
    ###########################################################

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
