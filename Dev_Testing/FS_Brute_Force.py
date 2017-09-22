#Starting with a 0/1 classification task


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from boruta import BorutaPy
from random import randint

import os
os.chdir('/home/chase.dehan/source/DataScience/Tools/Academic/Variable Selection')



########################################################################################
#
# Preprocessing
#
########################################################################################


#Bring in data and split into X and Y
lsvt = pd.read_csv("LSVT_VR.csv")
# Data came from here: https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation
X = lsvt[lsvt.columns[1:lsvt.shape[1]]]
Y = lsvt[lsvt.columns[0]] - 1

#The next step would be doing an initial ranking and split out of the variables,
    #BUT, we will skip that for now

#Split into a test/train    
seed = 1228
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    

########################################################################################
#
# Running the model - ALL DATA
#
########################################################################################

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


########################################################################################
#
# Run with Boruta
#
########################################################################################

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)
# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
# find all relevant features
feat_selector.fit(X_train.values, y_train.values)

# call transform() on X to filter it down to selected features
X_train_filter = feat_selector.transform(X_train.values)
X_test_filter = feat_selector.transform(X_test.values)

# fit model to training data
model = XGBClassifier()
model.fit(X_train_filter, y_train)

# make predictions for test data
y_pred = model.predict(X_test_filter)
boruta_pred = [round(value) for value in y_pred]

# evaluate predictions
boruta_acc = accuracy_score(y_test, boruta_pred)


#Compare Boruta to all_features
print("Boruta Accuracy: %.2f%%" % (boruta_acc * 100.0))
print("All Feature Accuracy: %.2f%%" % (accuracy * 100.0))
    #Boruta outperforms using 26 variables versus 310 in all features
    #Performs pretty well


########################################################################################
#
# First phase - Create random subsets of variables and run through a model
#
########################################################################################

ncol = X_train.shape[1]

#Generate random samples of variables
var_list = []
num_iterations = 100
for i in np.arange(num_iterations):
    num_vars = randint(10,50)  #Should hardcode the max size to scale with the dataset
    var_list.append( np.random.choice(ncol, num_vars, replace=False) )

#Then evaluate the performance of each model:
model = XGBClassifier()
#Split out the train/test from the above, because I created a validation
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=test_size, random_state=seed)
#Declare empty dataframe to be filled
eval_results = pd.DataFrame(index=np.arange(num_iterations), columns=np.insert(X_train2.columns.values, 0, ["Iteration", "Accuracy"]) ) #declare an empty list to place output of each into
for i in np.arange(num_iterations):
    these_vars = list(X_train2.columns[var_list[i]])
    X_train_filter = X_train2[these_vars]
    X_test_filter = X_test2[these_vars]
    #Fit and make predictions off the data
    model.fit(X_train_filter, y_train2)
    predictions = [round(value) for value in model.predict(X_test_filter)]
    acc_score = accuracy_score(y_test2, predictions)

    #Set the values of the scores and place into a larger
    var_scores = []
    for j in np.arange(ncol):
        if X_train2.columns[j] not in these_vars:
            value = -1
        else:
            value_index = these_vars.index(X_train2.columns[j])
            value = model.feature_importances_[value_index]
        var_scores.append(value)

    eval_results.loc[i] = np.insert(var_scores, 0, [i, acc_score])

eval_results = eval_results.sort("Accuracy")
#Ok, I now have results for each of the models
#The timing of running through these models is pretty fast, much faster than 100 iterations on Boruta


#Just going to run the model on the top model from eval_results
a = list(eval_results.tail(11).ix[:, "Iteration"] )
top_values = [int(x) for x in a]
#top_values = [120, 373, 225, 807, 121]
preds = pd.DataFrame()
for i in top_values:
    these_vars = list(X_train.columns[var_list[i]])
    X_train_filter = X_train[these_vars]
    X_test_filter = X_test[these_vars]
    model.fit(X_train_filter, y_train)
    predictions = [round(value) for value in model.predict(X_test_filter)]
    preds.insert(0, i, predictions)

preds.insert(0, "combined", np.where(preds.sum(1) >= 6, 1, 0))
ensemble_brute_acc = accuracy_score(y_test, preds["combined"])

#Then, just taking the top model
these_vars = list(X_train.columns[var_list[top_values[4]]])
X_train_filter = X_train[these_vars]
X_test_filter = X_test[these_vars]
model.fit(X_train_filter, y_train)
predictions = [round(value) for value in model.predict(X_test_filter)]
brute_force_acc = accuracy_score(y_test, predictions)
print("Brute Force Accuracy: %.2f%%" % (brute_force_acc * 100.0))
print("Boruta Accuracy: %.2f%%" % (boruta_acc * 100.0))
print("All Feature Accuracy: %.2f%%" % (accuracy * 100.0))
print("Ensemble Brute Accuracy: %.2f%%" % (ensemble_brute_acc * 100.0))

#I now know that the brute force approach can outperform Boruta, but it is an expensive process
#And could take an immense amount of time to force through it, especially on large datasets

########################################################################################
#
# Second Phase - Determine which variables are important
#
########################################################################################

#This should follow some sort of optimization process, where the algo automatically selects variables

#Want to rank the models for each iteration
#And rank the variables according to relative scalse

#Variable rankings from highest value to lowest
    # High ranking in high scoring models
    # High ranking with low scores
    # Low on most, but high in at least 1, or in top model(s)
        #Looking for non linear interactions
    # Low ranking in high scoring models
    # Low or no ranking in only poor models

#So, how do we move towards the objective?
    #Look at the relative scores - how many 0 values are there?
        #Importances will add up to 1
        #Should I normalize for num variables?
    #Take average of the scores
    #Find places where two variables together are above their averages
        #These variables should probably be tested together

    #Make some new combinations:
        # Make it probabilistic
            # What combinations are most likely to yield higher performance?
            # Add these points and recalculate the probabilities
            # Repeat
















