import pandas as pd
from boostaroota import BoostARoota
import urllib

#################
#Madelon Dataset
train_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
# download the file
raw_data = urllib.request.urlopen(train_url)
train = pd.read_csv(raw_data, delim_whitespace=True, header=None)
train.columns = ["Var"+str(x) for x in range(len(train.columns))]
labels_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
raw_data = urllib.request.urlopen(labels_url)
labels = pd.read_csv(raw_data, delimiter=",", header=None)
labels.columns = ["Y"]


########################################################################################################################
#
#  Test that BoostARoota is working
#
########################################################################################################################
br = BoostARoota(metric='logloss')

br.fit(train,labels)
len(train.columns)
len(br.keep_vars_)
new_train = br.transform(train)
new_train2 = br.fit_transform(train,labels)


#Dimension Reduction
print("Original training set has " + str(train.shape) + " dimensions.")
print("BoostARoota with .fit() and .transform() reduces to " + str(new_train.shape) + " dimensions.")
print("BoostARoota with .fit_transform() reduces to " + str(new_train2.shape) + " dimensions.")
print("The two methods may give a slightly different dimensions because of random variation as it is being refit")




########################
#Check to make sure that the algorithm is running correctly

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

data_path = "//pf.stormwind.local/DDE/Chase/Data/"
#################
#LSVT Voice Rehab data
    # Data came from here: https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation
lsvt = pd.read_csv(data_path + "LSVT_VR.csv")

#Split and make appropriate transformations
lsvt_X = lsvt[lsvt.columns[1:lsvt.shape[1]]].copy()
lsvt_Y = lsvt[lsvt.columns[0]].copy() - 1
lsvt_X = pd.get_dummies(lsvt_X)
del lsvt

TrainGetPreds(lsvt_X, lsvt_Y, lsvt)
