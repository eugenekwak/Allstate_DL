# runEnsembleLearner.py

# For interacting with and running the ensemble learner.

import os
import sys
import time
import datetime as dt 
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
    train_col_file = 'models/ensemble/train_cols.txt'
    model_object_file = 'models/ensemble/ensemble_model_'+dt.datetime.now().strftime('%Y_%m_%d')+'.pkl'
    dr_pipeline_file = 'models/ensemble/drPipeline_'+dt.datetime.now().strftime('%Y_%m_%d')+'.pkl'
    dr_threshold = '1.05*mean'
    val_frac = 0.15

    # Define the parameter grid
    param_grid = {'submodels__dtr__model__min_samples_split': [0.15],
                  'submodels__dtr__model__max_depth': [10],
                  'submodels__dtr__model__min_samples_leaf': [0.05],
                  'submodels__dtr__model__max_features': [10],
                  'submodels__sgd__model__learning_rate': ['optimal'],
                  'submodels__sgd__model__max_iter': [7000],
                  'submodels__sgd__model__penalty': ['l1', 'l2'],
                  'submodels__sgd__model__tol': [0.001],
                 }

    # Pipeline architecture for reference
    #
    # drPipeline = Pipeline([
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
    ensPipelineLearner = ensembleLearner()

    # Train the model
    ensPipelineLearner.buildModel(X_train, y_train, param_grid, val_frac, dr_threshold)

    # Make predictions on the test data
    # Also writes predictions to file
    print('Making predictions on test data...')
    preds_startTime = time.time()
    ensPipelineLearner.makePrediction(X_test, model_object_file, dr_pipeline_file)
    preds_endTime = time.time() - preds_startTime

    # Compute drivers and write driver output to file
    print('Getting drivers...')
    ensPipelineLearner.getDrivers(X_train, dr_pipeline_file)

    # Gather performance metrics
    print('Gathering performance metrics...')
    with open('models/ensemble/ensemble_report_'+dt.datetime.now().strftime('%Y_%m_%d')+'.txt', 'w') as text_file:
        text_file.write('Training R2 score: ' + str(ensPipelineLearner.r2Fit_) + '\n')
        text_file.write('Validation R2 score: ' + str(ensPipelineLearner.r2Val_) + '\n')
        text_file.write('Training MAE score: ' + str(ensPipelineLearner.maeFit_) + '\n') 
        text_file.write('Validation MAE score: ' + str(ensPipelineLearner.maeVal_) + '\n') 
        text_file.write('Training run time: ' + str(ensPipelineLearner.fitRunTime_) + ' seconds' + '\n')
        text_file.write('Prediction run time: ' + str(round(preds_endTime,6)) + ' seconds' + '\n')
        text_file.write('Best params: ' + str(ensPipelineLearner.modelObject.best_params_) + '\n') 
        text_file.write('Best estimator: ' + str(ensPipelineLearner.modelObject.best_estimator_) + '\n') 

    print('Job complete!')

if __name__ == '__main__':
    main()
