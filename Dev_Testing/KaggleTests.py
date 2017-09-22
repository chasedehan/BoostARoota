#KaggleTests.py

# This file builds applies to the predictions of house prices on Kaggle
# There is nothing clever or advanced about what is going on:
    #Purpose is to simply compare how it does on the public leaderboard
    #Does pretty well:
        #BoostARoota > Boruta > All Features


########################################################################################
#
# Predict on Kaggle for House Prices
#
########################################################################################
#BoostARoota outperforms all features and Boruta (although Boruta looks too bad)

house = pd.read_csv("~/DDE/Chase/Data/house_prices.csv")

# Data came from here: https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation
X = house[house.columns[1:(house.shape[1]-1)]].copy()
Y = house.SalePrice

#Determine which variables are categorical and OHE
categoricals = []
for col, col_type in X.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          X[col].fillna(0, inplace=True)

X = pd.get_dummies(X, columns=categoricals, dummy_na=True)


eval_metric = 'rmse'

X_test = pd.read_csv("~/DDE/Chase/Data/house_prices_test.csv")
ID = X_test.Id
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

#Trying the boostaroota
BR_vars = BoostARoota(X, Y, metric=eval_metric)
BR_X = X[BR_vars]
BR_test = X_test[BR_vars]
preds = TrainGetPreds(BR_X, Y, BR_test, metric=eval_metric)


#Trying all features
preds = TrainGetPreds(X, Y, X_test, metric=eval_metric)

#Trying Boruta
rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
feat_selector.fit(X.values, Y.values)
X_train_filter = feat_selector.transform(X.values)
X_test_filter = feat_selector.transform(X_test.values)
preds = TrainGetPreds(X_train_filter, Y, X_test_filter)


submit = pd.DataFrame({'ID':ID, 'SalePrice':preds})
submit.to_csv('~/Projects/house_submission.csv', index=False)

##############
### Testing the diagnostics for each variable selected
y_log = [log10(x) for x in y_actual]
mean_squared_error(y_log, [log10(x) for x in y_hat]) ** 0.5
mean_squared_error(y_log, [log10(x) for x in y_hat_BR]) ** 0.5
