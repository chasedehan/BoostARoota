#testBAR.py

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error, mean_absolute_error
#then import BoostARoota to get the proper functions for evaluation

########################################################################################
#
# Functions for testing output
#
########################################################################################

def PrepLL(y, y_pred):
# takes y, y_hat and returns the log loss
    preds = []
    for i in y_pred:
        preds.append([1 - i, i])
    logloss = log_loss(y, preds)
    return logloss

def rmse(y, y_pred):
# returns rmse for the
    return mean_squared_error(y, y_pred) ** 0.5


def evalReg(df):
#input df, output regression evaluation - rmse, mae
    results = [rmse(df.y_actual, df.y_hat_BR),
               rmse(df.y_actual, df.y_hat_boruta),
               rmse(df.y_actual, df.y_hat),
               mean_absolute_error(df.y_actual, df.y_hat_BR2),
               mean_absolute_error(df.y_actual, df.y_hat_boruta2),
               mean_absolute_error(df.y_actual, df.y_hat2) ]
    return results

def evalClass(df):
#input df, output classification evaluation - logloss, auc
    results = [PrepLL(df.y_actual, df.y_hat_BR),
               PrepLL(df.y_actual, df.y_hat_boruta),
               PrepLL(df.y_actual, df.y_hat),
               roc_auc_score(df.y_actual, df.y_hat_BR2),
               roc_auc_score(df.y_actual, df.y_hat_boruta2),
               roc_auc_score(df.y_actual, df.y_hat2) ]
    return results

def evalResults(df, eval):
    if eval == 'reg':
        results = evalReg(df)
    else:
        results = evalClass(df)
    return results

def evalBARBAR(df, eval):
    if eval == 'reg':
        results = [rmse(df.y_actual, df.y_hat_BR),
                   rmse(df.y_actual, df.y_hat2_BR),
                   rmse(df.y_actual, df.y_hat),
                   mean_absolute_error(df.y_actual, df.y_hat_BR),
                   mean_absolute_error(df.y_actual, df.y_hat2_BR),
                   mean_absolute_error(df.y_actual, df.y_hat)]
    else:
        results = [PrepLL(df.y_actual, df.y_hat_BR),
                   PrepLL(df.y_actual, df.y_hat2_BR),
                   PrepLL(df.y_actual, df.y_hat),
                   roc_auc_score(df.y_actual, df.y_hat_BR),
                   roc_auc_score(df.y_actual, df.y_hat2_BR),
                   roc_auc_score(df.y_actual, df.y_hat)]
    return results



########################################################################################
#
# Testing BAR against itself
#
########################################################################################

#just run through getting the predictions for each of the folds all features
    #Only test on a single metric
    #Compare the results from each iteration
def testBARvSelf(X, Y, eval, folds=5):
    if eval == "reg":
        eval_metric = "rmse"
    else:
        eval_metric = "logloss"

    np.random.seed(None) #removing any seed to ensure that the folds are created differently

    #initialize empty lists - not the most efficient, but it works
    bar_times = []
    bar2_times = []
    y_hat = []
    y_hat_BR = []
    y_hat2_BR = []
    y_actual = []
    fold = []

    #Start the cross validation
    kf = KFold(n_splits=folds)
    i = 1
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], Y[train], Y[test]

        #Get predictions on all features
        y_pred = TrainGetPreds(X_train, y_train, X_test, metric=eval_metric)

        #BAR1
        tmp = time.time()
        BR_vars = BoostARoota(X_train, y_train, metric=eval_metric)
        bar_times.append(time.time() - tmp)
        BR_X = X_train[BR_vars]
        BR_test = X_test[BR_vars]
        BR_preds = TrainGetPreds(BR_X, y_train, BR_test, metric=eval_metric)

        #BAR2
        tmp = time.time()
        BR_vars = BoostARoota2(X_train, y_train, metric=eval_metric)
        bar2_times.append(time.time() - tmp)
        BR_X = X_train[BR_vars]
        BR_test = X_test[BR_vars]
        BR2_preds = TrainGetPreds(BR_X, y_train, BR_test, metric=eval_metric)

        # evaluate predictions and append to lists
        y_hat.extend(y_pred)
        y_hat_BR.extend(BR_preds)
        y_hat2_BR.extend(BR2_preds)
        y_actual.extend(y_test)
        #Set the fold it is trained on
        fold.extend([i] * len(y_pred))
        i+=1

    values = [np.mean(bar_times), np.mean(bar2_times)]
    #Start building the array to be passed out; first is the timings, then the eval results
    #Build the dataframe to pass into the evaluation functions
    results = pd.DataFrame({"y_hat": y_hat,
                            "Fold": fold,
                            "y_hat_BR": y_hat_BR,
                            "y_hat2_BR": y_hat2_BR,
                            "y_actual": y_actual})
    values.extend(evalBARBAR(results, eval=eval))

    return pd.DataFrame(values, ["BarTime1", "BarTime2",
                                 'BAR1_Metric1', 'BAR2_Metric1', 'AllMetric1',
                                 'BAR1_Metric2', 'BAR2_Metric2', 'AllMetric2'])




########################################################################################
#
# Functions for rigorous testing of the approaches
#
########################################################################################
#just run through getting the predictions for each of the folds all features
def trainKFolds(X, Y, eval, folds=5):
    if eval == "reg":
        eval_metric = "rmse"
        eval_metric2 = "mae"
    else:
        eval_metric = "logloss"
        eval_metric2 = "auc"
    np.random.seed(None) #removing any seed to ensure that the folds are created differently

    #initialize empty lists - not the most efficient, but it works
    bar_times = []
    boruta_times = []
    y_hat = []
    y_hat2 = []
    y_hat_BR = []
    y_hat_BR2 = []
    y_hat_boruta = []
    y_hat_boruta2 = []
    y_actual = []
    fold = []

    #Start the cross validation
    kf = KFold(n_splits=folds)
    i = 1
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], Y[train], Y[test]

        #Get predictions on all features
        y_pred = TrainGetPreds(X_train, y_train, X_test, metric=eval_metric)
        y_pred2 = TrainGetPreds(X_train, y_train, X_test, metric=eval_metric2)

        #BoostARoota - tune to metric 1
        tmp = time.time()
        BR_vars = BoostARoota2(X_train, y_train, metric=eval_metric)
        bar_times.append(time.time() - tmp)
        BR_X = X_train[BR_vars]
        BR_test = X_test[BR_vars]
        BR_preds = TrainGetPreds(BR_X, y_train, BR_test, metric=eval_metric)

        #BoostARoota - tune to metric 2
        tmp = time.time()
        BR_vars = BoostARoota2(X_train, y_train, metric=eval_metric2)
        bar_times.append(time.time() - tmp)
        BR_X = X_train[BR_vars]
        BR_test = X_test[BR_vars]
        BR_preds2 = TrainGetPreds(BR_X, y_train, BR_test, metric=eval_metric2)

        # #Boruta - get predictions
        tmp = time.time()
        rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
        feat_selector.fit(X_train.values, y_train.values)
        boruta_times.append(time.time() - tmp)
        X_train_filter = feat_selector.transform(X_train.values)
        X_test_filter = feat_selector.transform(X_test.values)
        Boruta_preds = TrainGetPreds(X_train_filter, y_train, X_test_filter, metric=eval_metric)
        Boruta_preds2 = TrainGetPreds(X_train_filter, y_train, X_test_filter, metric=eval_metric2)

        # evaluate predictions and append to lists
        y_hat.extend(y_pred)
        y_hat2.extend(y_pred2)
        y_hat_BR.extend(BR_preds)
        y_hat_BR2.extend(BR_preds2)
        y_hat_boruta.extend(Boruta_preds)
        y_hat_boruta2.extend(Boruta_preds2)
        y_actual.extend(y_test)
        #Set the fold it is trained on
        fold.extend([i] * len(y_pred))
        i+=1

    #Start building the array to be passed out; first is the timings, then the eval results
    values = [np.mean(boruta_times), np.mean(bar_times)]
    #Build the dataframe to pass into the evaluation functions
    results = pd.DataFrame({"y_hat": y_hat,
                            "y_hat2": y_hat2,
                            "Fold": fold,
                            "y_hat_BR": y_hat_BR,
                            "y_hat_BR2": y_hat_BR2,
                            "y_hat_boruta": y_hat_boruta,
                            "y_hat_boruta2": y_hat_boruta2,
                            "y_actual": y_actual})

    #then append the evaluation results to values
    values.extend(evalResults(results, eval=eval))

    return values


def repCV(X, Y, eval, repeats=3):
    #runs trainKFolds() for however many repeats specified here
    #Returns the results
    names = ['BorutaTime', 'BarTime',
             'BarMetric1', 'BorutaMetric1', 'AllMetric1',
             'BarMetric2', 'BorutaMetric2', 'AllMetric2']
    for i in range(repeats):
        if i == 0:
            df = pd.DataFrame(trainKFolds(X, Y, eval)).T
        df = df.append(pd.DataFrame(trainKFolds(X, Y, eval)).T, ignore_index=True)
    df.columns = names
    return df








