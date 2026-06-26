import numpy as np
import pandas as pd
import xgboost as xgb
import operator
import warnings


########################################################################################
#
# Main Class and Methods
#
########################################################################################


class BoostARoota(object):

    def __init__(self, metric=None, clf=None, cutoff=4, iters=10, max_rounds=100, delta=0.1, silent=False, task='auto'):
        self.metric = metric
        self.clf = clf
        self.cutoff = cutoff
        self.iters = iters
        self.max_rounds = max_rounds
        self.delta = delta
        self.silent = silent
        self.task = task
        self.keep_vars_ = None

        #Throw errors if the inputted parameters don't meet the necessary criteria
        if (metric is None) and (clf is None):
            raise ValueError('you must enter one of metric or clf as arguments')
        if cutoff <= 0:
            raise ValueError('cutoff should be greater than 0. You entered' + str(cutoff))
        if iters <= 0:
            raise ValueError('iters should be greater than 0. You entered' + str(iters))
        if (delta <= 0) | (delta > 1):
            raise ValueError('delta should be between 0 and 1, was ' + str(delta))

        #Issue warnings for parameters to still let it run
        if (metric is not None) and (clf is not None):
            warnings.warn('You entered values for metric and clf, defaulting to clf and ignoring metric')
        if delta < 0.02:
            warnings.warn("WARNING: Setting a delta below 0.02 may not converge on a solution.")
        if max_rounds < 1:
            warnings.warn("WARNING: Setting max_rounds below 1 will automatically be set to 1.")

    def fit(self, x, y):
        self.keep_vars_ = _BoostARoota(x, y,
                                       metric=self.metric,
                                       clf = self.clf,
                                       cutoff=self.cutoff,
                                       iters=self.iters,
                                       max_rounds=self.max_rounds,
                                       delta=self.delta,
                                       silent=self.silent,
                                       task=self.task)
        return self

    def transform(self, x):
        if self.keep_vars_ is None:
            raise ValueError("You need to fit the model first")
        return x[self.keep_vars_]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

########################################################################################
#
# Helper Functions to do the Heavy Lifting
#
########################################################################################


def _create_shadow(x_train):
    """
    Take all X variables, creating copies and randomly shuffling them
    :param x_train: the dataframe to create shadow features on
    :return: dataframe 2x width and the names of the shadows for removing later
    """
    x_shadow = x_train.copy()
    for c in x_shadow.columns:
        # np.random.shuffle on DataFrame column values fails with numpy 2.x (read-only)
        # Use permutation which returns a new shuffled array, compatible with numpy 1.x and 2.x
        x_shadow[c] = np.random.permutation(x_shadow[c].values)
    # rename the shadow
    shadow_names = ["ShadowVar" + str(i + 1) for i in range(x_train.shape[1])]
    x_shadow.columns = shadow_names
    # Combine to make one new dataframe
    new_x = pd.concat([x_train, x_shadow], axis=1)
    return new_x, shadow_names

########################################################################################
#
# BoostARoota
#
########################################################################################


def _reduce_vars_xgb(x, y, metric, this_round, cutoff, n_iterations, delta, silent, task='auto'):
    """
    Function to run through each
    :param x: Input dataframe - X
    :param y: Target variable
    :param metric: Metric to optimize in XGBoost
    :param this_round: Round so it can be printed to screen
    :return: tuple - stopping criteria and the variables to keep
    """
    # Determine task type if auto
    if task == 'auto':
        # Heuristic: if y is float or has many unique values, treat as regression
        y_unique = np.unique(y)
        if np.issubdtype(np.array(y).dtype, np.floating) or len(y_unique) > 20:
            task = 'regression'
        else:
            task = 'classification'

    # Set up the parameters for running the model in XGBoost
    # Regression metrics
    regression_metrics = {'rmse', 'mae', 'mape', 'rmsle', 'mphe'}
    classification_metrics = {'logloss', 'error', 'auc', 'aucpr', 'error_rate'}

    if metric == 'mlogloss':
        param = {'objective': 'multi:softmax',
                 'eval_metric': 'mlogloss',
                 'num_class': len(np.unique(y)),
                 'verbosity': 0}
    elif metric in regression_metrics or task == 'regression':
        param = {'objective': 'reg:squarederror',
                 'eval_metric': metric if metric in regression_metrics else 'rmse',
                 'verbosity': 0}
    else:
        # classification (binary)
        param = {'objective': 'binary:logistic',
                 'eval_metric': metric if metric in classification_metrics else 'logloss',
                 'verbosity': 0}
    for i in range(1, n_iterations+1):
        # Create the shadow variables and run the model to obtain importances
        new_x, shadow_names = _create_shadow(x)
        dtrain = xgb.DMatrix(new_x, label=y)
        # xgboost 1.x+ uses verbosity in params, older versions use verbose_eval
        try:
            bst = xgb.train(param, dtrain, num_boost_round=10)
        except TypeError:
            bst = xgb.train(param, dtrain, verbose_eval=False)
        if i == 1:
            df = pd.DataFrame({'feature': new_x.columns})
            pass

        # Get feature importance - try new API first, fallback to old
        try:
            importance = bst.get_score(importance_type='weight')
        except AttributeError:
            importance = bst.get_fscore()
        if not importance:
            # No splits were made, assign 0 to all features
            importance = {f: 0 for f in new_x.columns}
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df2 = pd.DataFrame(importance, columns=['feature', 'fscore'+str(i)])
        # Normalize, avoid division by zero
        fscore_sum = df2['fscore'+str(i)].sum()
        if fscore_sum > 0:
            df2['fscore'+str(i)] = df2['fscore'+str(i)] / fscore_sum
        df = pd.merge(df, df2, on='feature', how='outer')
        if not silent:
            print("Round: ", this_round, " iteration: ", i)

    df = df.fillna(0)
    # pandas 2.x requires numeric_only=True to exclude 'feature' string column
    try:
        df['Mean'] = df.mean(axis=1, numeric_only=True)
    except TypeError:
        # pandas <1.5 fallback
        df['Mean'] = df.mean(axis=1)
    #Split them back out
    real_vars = df[~df['feature'].isin(shadow_names)]
    shadow_vars = df[df['feature'].isin(shadow_names)]

    # Get mean value from the shadows
    mean_shadow = shadow_vars['Mean'].mean() / cutoff
    real_vars = real_vars[(real_vars.Mean > mean_shadow)]

    #Check for the stopping criteria
    #Basically looking to make sure we are removing at least 10% of the variables, or we should stop
    if (len(real_vars['feature']) / len(x.columns)) > (1-delta):
        criteria = True
    else:
        criteria = False

    return criteria, real_vars['feature']


def _reduce_vars_sklearn(x, y, clf, this_round, cutoff, n_iterations, delta, silent):
    """
    Function to run through each
    :param x: Input dataframe - X
    :param y: Target variable
    :param clf: the fully specified classifier passed in by user
    :param this_round: Round so it can be printed to screen
    :return: tuple - stopping criteria and the variables to keep
    """
    #Set up the parameters for running the model in XGBoost - split is on multi log loss

    for i in range(1, n_iterations+1):
        # Create the shadow variables and run the model to obtain importances
        new_x, shadow_names = _create_shadow(x)
        clf = clf.fit(new_x, np.ravel(y))

        if i == 1:
            df = pd.DataFrame({'feature': new_x.columns})
            df2 = df.copy()
            pass

        try:
            importance = clf.feature_importances_
            df2['fscore' + str(i)] = importance
        except ValueError:
            print("this clf doesn't have the feature_importances_ method.  Only Sklearn tree based methods allowed")

        # importance = sorted(importance.items(), key=operator.itemgetter(1))

        # df2 = pd.DataFrame(importance, columns=['feature', 'fscore'+str(i)])
        df2['fscore'+str(i)] = df2['fscore'+str(i)] / df2['fscore'+str(i)].sum()
        df = pd.merge(df, df2, on='feature', how='outer')
        if not silent:
            print("Round: ", this_round, " iteration: ", i)

    df = df.fillna(0)
    # pandas 2.x requires numeric_only=True to exclude 'feature' string column
    try:
        df['Mean'] = df.mean(axis=1, numeric_only=True)
    except TypeError:
        # pandas <1.5 fallback
        df['Mean'] = df.mean(axis=1)
    #Split them back out
    real_vars = df[~df['feature'].isin(shadow_names)]
    shadow_vars = df[df['feature'].isin(shadow_names)]

    # Get mean value from the shadows
    mean_shadow = shadow_vars['Mean'].mean() / cutoff
    real_vars = real_vars[(real_vars.Mean > mean_shadow)]

    #Check for the stopping criteria
    #Basically looking to make sure we are removing at least 10% of the variables, or we should stop
    if (len(real_vars['feature']) / len(x.columns)) > (1-delta):
        criteria = True
    else:
        criteria = False

    return criteria, real_vars['feature']

#Main function exposed to run the algorithm
def _BoostARoota(x, y, metric, clf, cutoff, iters, max_rounds, delta, silent, task='auto'):
    """
    Function loops through, waiting for the stopping criteria to change
    :param x: X dataframe One Hot Encoded
    :param y: Labels for the target variable
    :param metric: The metric to optimize in XGBoost
    :return: names of the variables to keep
    """

    new_x = x.copy()
    #Run through loop until "crit" changes
    i = 0
    while True:
        #Inside this loop we reduce the dataset on each iteration exiting with keep_vars
        i += 1
        if clf is None:
            crit, keep_vars = _reduce_vars_xgb(new_x,
                                               y,
                                               metric=metric,
                                               this_round=i,
                                               cutoff=cutoff,
                                               n_iterations=iters,
                                               delta=delta,
                                               silent=silent,
                                               task=task)
        else:
            crit, keep_vars = _reduce_vars_sklearn(new_x,
                                                   y,
                                                   clf=clf,
                                                   this_round=i,
                                                   cutoff=cutoff,
                                                   n_iterations=iters,
                                                   delta=delta,
                                                   silent=silent)

        if crit | (i >= max_rounds):
            break  # exit and use keep_vars as final variables
        else:
            new_x = new_x[keep_vars].copy()
    if not silent:
        print("BoostARoota ran successfully! Algorithm went through ", i, " rounds.")
    return keep_vars
