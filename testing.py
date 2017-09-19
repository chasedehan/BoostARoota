from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#################
#LSVT Voice Rehab data
    # Data came from here: https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation
lsvt = pd.read_csv(data_path + "LSVT_VR.csv")

#Split and make appropriate transformations
lsvt_X = lsvt[lsvt.columns[1:lsvt.shape[1]]].copy()
lsvt_Y = lsvt[lsvt.columns[0]].copy() - 1
lsvt_X = GetCatDummies(lsvt_X)
del lsvt

names = BoostARoota2(lsvt_X, lsvt_Y, metric='logloss')

lsvt_results = testBARvSelf(lsvt_X, lsvt_Y, eval='class')
print(lsvt_results)


##############################################################################################
#
#
# Now trying to work it up with multi-class target
#
#
##############################################################################################
#It works as expected, generating at least equivalent logloss

iris = pd.read_csv(data_path + "iris.csv")
del iris['Id']
iris_Y = iris.Species
iris_X = iris.copy()
del iris_X['Species']

iris_X = GetCatDummies(iris_X)

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(iris_Y)
label_encoded_y = label_encoder.transform(iris_Y)
#Split and train model
X_train, X_test, y_train, y_test = train_test_split(iris_X, label_encoded_y, test_size=0.2)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, y_test)
# fit model no training data
param = {'objective': 'multi:softprob',
         'eval_metric': 'mlogloss',
         'num_class':3}
bst = xgb.train(param, dtrain)
y_pred = bst.predict(dtest)

log_loss(y_test, y_pred)
#Just testing to make sure it works
vars = BoostARoota2(X_train, y_train, metric="mlogloss")



##############################################################################################
#
#
# Original Tests
#
#
##############################################################################################
data_path = "~/DDE/Chase/Data/"
#################
#LSVT Voice Rehab data
    # Data came from here: https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation
lsvt = pd.read_csv(data_path + "LSVT_VR.csv")

#Split and make appropriate transformations
lsvt_X = lsvt[lsvt.columns[1:lsvt.shape[1]]].copy()
lsvt_Y = lsvt[lsvt.columns[0]].copy() - 1
lsvt_X = GetCatDummies(lsvt_X)
del lsvt
BoostARoota2(lsvt_X, lsvt_Y, metric="logloss")
lsvt_results = testBARvSelf(lsvt_X, lsvt_Y, eval='class')