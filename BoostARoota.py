#BoostARoota_Better.py
## newest
#
#Basic algorithm runs and works just fine (BoostARoota.py)
#
#This script adds in a few features to the function.  The plan:
    #widening feature set, create functions for:
        #PCA - Done
        #Ordinal and OHE for categorical - OHE Done
    #Specifying the eval metric
        #Need to allow for custom eval_metric to be passed in - xgboost allows it
#Next steps are:
    #Check performance of the variable subsets and keep removing until metric drops
    #Also, check individual performance across folds

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
import operator
from sklearn.decomposition import PCA



########################################################################################
#
# Main functions
#
########################################################################################

#Take all X variables, creating copies and randomly shuffling them
def CreateShadow(X_train):
    X_shadow = X_train.copy()
    for c in X_shadow.columns:
        np.random.shuffle(X_shadow[c].values)
    # rename the shadow
    shadow_names = ["V" + str(i + 1) for i in range(X_train.shape[1])]
    X_shadow.columns = shadow_names
    # Combine to make one new dataframe
    new_X = pd.concat([X_train, X_shadow], axis=1)
    return new_X, shadow_names

#Main function exposed to run the algorithm
def BoostARoota(X, Y, metric):
    n_iterations = 10
    param = {'eval_metric': metric}
    cutoff = 4
    for i in range(1, n_iterations + 1):
        # Create the shadow variables and run the model to obtain importances
        new_X, shadow_names = CreateShadow(X)
        dtrain = xgb.DMatrix(new_X, label=Y)
        bst = xgb.train(param, dtrain)
        if i == 1:
            df = pd.DataFrame({'feature': new_X.columns})
            pass

        importance = bst.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df2 = pd.DataFrame(importance, columns=['feature', 'fscore'+str(i)])
        df2['fscore'+str(i)] = df2['fscore'+str(i)] / df2['fscore'+str(i)].sum()
        df = pd.merge(df, df2, on='feature', how='outer')

    df['Mean'] = df.mean(axis=1)
    #Split them back out
    real_vars = df[~df['feature'].isin(shadow_names)]
    shadow_vars = df[df['feature'].isin(shadow_names)]

    # Get mean value from the shadows
    mean_shadow = shadow_vars['Mean'].mean() / cutoff  #TODO: At what level of conservativeness do I cut this off?
    real_vars = real_vars[(real_vars.Mean > mean_shadow)]
    return real_vars['feature']


#Define helper function to train a model: returns the predictions
def TrainGetPreds(X_train, Y_train, X_test, metric):
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test)
    param = {'eval_metric': metric}
    bst = xgb.train(param, dtrain)
    # evaluate predictions
    y_pred = bst.predict(dtest)
    return y_pred


########################################################################################
#
# Preprocessing Functions
#
########################################################################################
def GetCatDummies(X, ReplaceNA=True):
    """Takes in a pd.DataFrame, checks for the categorical variables and returns a OHE DataFrame
    This needs to be done after any ordinal transformations because the original variable is replaced"""
    categoricals = []
    for col, col_type in X.dtypes.iteritems():
         if col_type == 'O':
              categoricals.append(col)
         else:
              X[col].fillna(0, inplace=True)
    return pd.get_dummies(X, columns=categoricals, dummy_na=ReplaceNA) #Default set to True

def AddPCA(X):
    "Creates the square root of N columns PCA variables"
    n_pca = round(X.shape[1]**0.5)
    pca = PCA(n_components=n_pca)
    pca.fit(X)
    pca_X = pd.DataFrame(pca.transform(X))
    pca_X.columns = ["pca" + str(x) for x in range(1,n_pca+1)]
    X = pd.concat([X, pca_X], axis=1)
    return X
#TODO: need to generate pca.transform(test_X) to allow for appropriate test predictions
    #Because where it is at right now, it doesn't allow for

def AddOrdinals(X):
    #TODO: this one is challenging because it needs to know the ordering of the categoricals.
        #Maybe I just run some naive approach on the data
    return X

def ClassifyMissing(X):
    #Creates dummy variable for if that variable is missing, missing = 1, otherwise = 0
        #TODO: also bring in the
    return X

