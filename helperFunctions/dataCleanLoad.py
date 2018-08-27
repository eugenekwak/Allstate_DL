# dataCleanLoad.py
# Functions for loading and prepping Allstate Kaggle data for modeling.
# Last updated: 2018-08-20

import pandas as pd
import os
import numpy as np
from sklearn.utils import shuffle

def dataLoad(dataDirectory, trainFileName, testFileName):
    '''
    Cleans and returns train and test data from the Allstate Claim Severity 
    Kaggle challenge.

    Arguments
    ----------------
    @ dataDirectory: Directory where data files are.
                     Expected format: '/home/user/.../'

    @ trainFileName: Name of the train data file. Only accepts CSV. 
                     Expected format: 'train.csv' 

    @ testFileName: Name of the test data file. Only accepts CSV. 
                    Expected format: 'train.csv' 

    Returns
    ----------------
    Train and test Pandas dataframes.

    '''

    # Set file names.
    trainFile = os.path.join(os.path.dirname('__file__'), dataDirectory, trainFileName)
    testFile = os.path.join(os.path.dirname('__file__'), dataDirectory, testFileName)

    # Read csv files.
    train_df = pd.read_csv(trainFile, sep=',')
    test_df = pd.read_csv(testFile, ',')

    # Log transform the target variable.
    train_df['log_loss'] = np.log(train_df['loss'])
    train_df.drop(columns=['loss'], inplace=True)

    # Set row index.
    train_df.set_index('id', inplace=True)
    test_df.set_index('id', inplace=True)

    return train_df, test_df

def dataModelTrainingPrepare(df):
    '''
    Converts Pandas dataframe into a format that can be used for training models.

    - Shuffles the data.
    - One-hot encode categorical variables.
    - Writes list of column names following one-hot encoding.

    Note - No further work needed for this data as it is pre cleaned.

    Arguments
    ----------------
    @ df: Name of the Pandas dataframe.

    Returns
    ----------------
    Two data frames for the training data. One for predictors and one for target.

    '''

    # One hot encode categorical variables.
    cat_features = list(df.select_dtypes(exclude=['number','bool_','int']).columns)
    df = pd.get_dummies(df, columns=cat_features)

    # Shuffle data frame.
    df = shuffle(df, random_state=10)

    # Get list of columns and write to file for use later.
    train_cols = list(df.columns)

    with open('train_cols.txt', 'w') as f:
        for s in train_cols:
            f.write(str(s) +'\n')
    
    # separate features from target
    X = df.drop('log_loss', axis=1)
    y = df['log_loss']

    return X.copy(), y.copy()

def dataModelScoringPrepare(df):
    '''
    Converts Pandas dataframe intest_df_cleanat that can be used for scoring new 
    data.

    - One-hot encode categoricaltest_df_cleans.
    - Reconcile dataset to the training data.

    Arguments
    ----------------
    @ df: Name of the Pandas dataframe.

    Returns
    ----------------
    Pandas dataframe.

    '''

    # Get list of columns in the training data to reconcile differences in new
    # data.
    train_cols = []
    with open('train_cols.txt', 'r') as f:
        for line in f:
            train_cols.append(line.strip())

    # One hot encode categorical variables.
    cat_features = list(df.select_dtypes(exclude=['number','bool_','int']).columns)
    df = pd.get_dummies(df, columns=cat_features)

    # Cycle through list of columns in test data frame and default missing to 0.
    score_cols = list(df.columns)
    missing_cols = list(set(train_cols) - set(score_cols))

    for m in missing_cols:
        df[m] = 0
    
    # Re-order columns to match training column order.
    df = df[train_cols]
    df.drop('log_loss', axis=1, inplace=True)

    return df.copy()

