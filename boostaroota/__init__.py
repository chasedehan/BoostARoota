import numpy as np
import pandas as pd
import xgboost as xgb
import operator

########################################################################################
#
# Main functions
#
########################################################################################


def CreateShadow(X_train):
    """
    Take all X variables, creating copies and randomly shuffling them
    :param X_train: the dataframe to create shadow features on
    :return: dataframe 2x width and the names of the shadows for removing later
    """
    X_shadow = X_train.copy()
    for c in X_shadow.columns:
        np.random.shuffle(X_shadow[c].values)
    # rename the shadow
    shadow_names = ["V" + str(i + 1) for i in range(X_train.shape[1])]
    X_shadow.columns = shadow_names
    # Combine to make one new dataframe
    new_X = pd.concat([X_train, X_shadow], axis=1)
    return new_X, shadow_names

########################################################################################
#
# BoostARoota
#
########################################################################################
def reduceVars(X, Y, metric, round):
    """
    Function to run through each
    :param X: Input dataframe - X
    :param Y: Target variable
    :param metric: Metric to optimize in XGBoost
    :param round: Round so it can be printed to screen
    :return: tuple - stopping criteria and the variables to keep
    """
    cutoff = 4
    n_iterations = 10

    #Split out the parameters if it is a multi class problem
    if metric == 'mlogloss':
        param = {'objective': 'multi:softmax',
                 'eval_metric': 'mlogloss',
                 'num_class': len(np.unique(Y)),
                 'silent': 1}
    else:
        param = {'eval_metric': metric,
                 'silent': 1}

    for i in range(1, n_iterations+1):
        # Create the shadow variables and run the model to obtain importances
        new_X, shadow_names = CreateShadow(X)
        dtrain = xgb.DMatrix(new_X, label=Y)
        bst = xgb.train(param, dtrain, verbose_eval=False)
        if i == 1:
            df = pd.DataFrame({'feature': new_X.columns})
            pass

        importance = bst.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df2 = pd.DataFrame(importance, columns=['feature', 'fscore'+str(i)])
        df2['fscore'+str(i)] = df2['fscore'+str(i)] / df2['fscore'+str(i)].sum()
        df = pd.merge(df, df2, on='feature', how='outer')
        print("Round: ", round, " iteration: ", i)

    df['Mean'] = df.mean(axis=1)
    #Split them back out
    real_vars = df[~df['feature'].isin(shadow_names)]
    shadow_vars = df[df['feature'].isin(shadow_names)]

    # Get mean value from the shadows
    mean_shadow = shadow_vars['Mean'].mean() / cutoff
    real_vars = real_vars[(real_vars.Mean > mean_shadow)]

    #Check for the stopping criteria
        #Basically looking to make sure we are removing at least 10% of the variables, or we should stop
    if (len(real_vars['feature']) / len(X.columns)) > 0.90:
        criteria = 1
    else:
        criteria = 0

    return criteria, real_vars['feature']

#Main function exposed to run the algorithm
def BoostARoota(X, Y, metric):
    """
    Function loops through, waiting for the stopping criteria to change
    :param X: X dataframe One Hot Encoded
    :param Y: Labels for the target variable
    :param metric: The metric to optimize in XGBoost
    :return: names of the variables to keep
    """

    new_X = X.copy()
    #Run through loop until "crit" changes
    i = 0
    while True:
        #Inside this loop we reduce the dataset on each iteration exiting with keep_vars
        i += 1
        crit, keep_vars = reduceVars(new_X, Y, metric=metric, round=i)

        if crit == 1:
            break #exit and use keep_vars as final variables
        else:
            new_X = new_X[keep_vars].copy()
    print("BoostARoota ran successfully! Algorithm went through ", i, " rounds.")
    return keep_vars