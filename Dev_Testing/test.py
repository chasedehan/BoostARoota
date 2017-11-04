import pandas as pd
import numpy as np
from boostaroota import BoostARoota
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import time

#Define helper function to train a model: returns the predictions
def TrainGetPreds(X_train, Y_train, X_test, metric):
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test)
    param = {'eval_metric': metric}
    bst = xgb.train(param, dtrain)
    # evaluate predictions
    y_pred = bst.predict(dtest)
    return y_pred

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
               rmse(df.y_actual, df.y_hat)]
    return results

def evalClass(df):
#input df, output classification evaluation - logloss, auc
    results = [PrepLL(df.y_actual, df.y_hat_BR),
               PrepLL(df.y_actual, df.y_hat) ]
    return results

def evalResults(df, eval):
    if eval == 'reg':
        results = evalReg(df)
    else:
        results = evalClass(df)
    return results

def trainKFolds(X, Y, eval, folds=5):
    if eval == "reg":
        eval_metric = "rmse"
    else:
        eval_metric = "logloss"
    np.random.seed(None) #removing any seed to ensure that the folds are created differently

    #initialize empty lists - not the most efficient, but it works
    bar_times = []
    y_hat = []
    y_hat_BR = []
    y_actual = []
    fold = []

    #Start the cross validation
    kf = KFold(n_splits=folds)
    i = 1
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], Y[train], Y[test]

        #Get predictions on all features
        y_pred = TrainGetPreds(X_train, y_train, X_test, metric=eval_metric)

        #BoostARoota - tune to metric 1
        tmp = time.time()
        br = BoostARoota(eval_metric)
        br.fit(X_train, y_train)
        bar_times.append(time.time() - tmp)
        BR_X = br.transform(X_train)
        BR_test = br.transform(X_test)
        BR_preds = TrainGetPreds(BR_X, y_train, BR_test, metric=eval_metric)

        # evaluate predictions and append to lists
        y_hat.extend(y_pred)
        y_hat_BR.extend(BR_preds)
        y_actual.extend(y_test)
        #Set the fold it is trained on
        fold.extend([i] * len(y_pred))
        i+=1

    #Start building the array to be passed out; first is the timings, then the eval results
    values = [np.mean(bar_times)]
    #Build the dataframe to pass into the evaluation functions
    results = pd.DataFrame({"y_hat": y_hat,
                            "Fold": fold,
                            "y_hat_BR": y_hat_BR,
                            "y_actual": y_actual})

    #then append the evaluation results to values
    values.extend(evalResults(results, eval=eval))

    return values


def repCV(X, Y, eval, repeats=3):
    #runs trainKFolds() for however many repeats specified here
    #Returns the results
    names = ['BarTime', 'BarMetric1','AllMetric1']
    for i in range(repeats):
        if i == 0:
            df = pd.DataFrame(trainKFolds(X, Y, eval)).T
        df = df.append(pd.DataFrame(trainKFolds(X, Y, eval)).T, ignore_index=True)
    df.columns = names
    return df


#################
#
#LSVT Voice Rehab data
    # Data came from here: https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation
lsvt = pd.read_csv(data_path + "LSVT_VR.csv")

#Split and make appropriate transformations
lsvt_X = lsvt[lsvt.columns[1:lsvt.shape[1]]].copy()
lsvt_Y = lsvt[lsvt.columns[0]].copy() - 1
lsvt_X = pd.get_dummies(lsvt_X)
del lsvt

lsvt_results = repCV(lsvt_X, lsvt_Y, eval='class', repeats=1)