# BoostARoota  
A Fast XGBoost Feature Selection Algorithm  

## Why Create Another Algorithm?  
Automated processes like Boruta showed early promise as they were able to provide superior performance with Random Forests, but has some deficiencies including slow computation time: especially with high dimensional data. Regardless of the run time, Boruta does perform well on Random Forests, but performs poorly on other algorithms such as boosting or neural networks. Similar deficiencies occur with regularization on LASSO, elastic net, or ridge regressions in that they perform well on linear regressions, but poorly on other modern algorithms.

I am proposing and demonstrating a feature selection algorithm (called BoostARoota) in a similar spirit to Boruta utilizing XGBoost as the base model rather than a Random Forest. The algorithm runs in a fraction of the time it takes Boruta and has superior performance on a variety of datasets.  While the spirit is similar to Boruta, BoostARoota takes a slightly different approach for the removal of attributes that executes much faster.

## How to use the code  
Easiest way is to use `pip`:
```
$ pip install boostaroota
```

The only function you _have_ to have is `br.BoostARoota(X, Y, metric)`.  In order to use the function, it does require X to be one-hot-encoded(OHE), so using the pandas function `pd.get_dummies(X)` may be helpful as it determines which variables are categorical and converts them into dummy variables.  

Assuming you have X and Y split, you can run the following:  
```
import boostaroota as br
import pandas as pd

#Specify the evaluation metric: can use whichever you like as long as recognized by XGBoost
  #EXCEPTION: multi-class currently only supports "mlogloss" so much be passed in as eval_metric
eval_metric = 'logloss' 
#OHE the Predictors
X = pd.getdummies(X)

#Run BoostARoota - will return a list of variables
BR_vars = br.BoostARoota(X, Y, metric=eval_metric)
  
#Reduce X to only include those deemed as import to BoostARoota
BR_X = X[BR_vars].copy()
```

It's really that simple!  Of course, as we build more functionality into it, it will get to be more difficult.  Keep in mind that since you are OHE, if you have a numeric variable that is imported by python as a character, pd.get_dummies() will convert those numeric into many columns.  This can cause your DataFrame to explode in size, giving unexpected results and high run times.


## How it works  
Similar in spirit to Boruta, BoostARoota creates shadow features, but modifies the removal step a little mored

1. One-Hot-Encode the feature set
2. Double width of the data set, making a copy of all features in original dataset
3. Randomly shuffle the new features created in (2).  These duplicated and shuffled features are referred to as "shadow features"
4. Run XGBoost classifier on the entire data set ten times.  Running it ten times allows for random noise to be smoothed, resulting in more robust estimates of importance.
5. Obtain importance values for each feature.  This is a simple importance metric that sums up how many times the particular feature was split on in the XGBoost algorithm.
6. Compute "cutoff": the average feature importance value for all shadow features and divide by four.  Shadow importance values are divided by four to make it more difficult for the variables to be removed.  With values lower than this, features are removed at too high of a rate.
7. Remove features with average importance across the ten iterations that is less than the cutoff specified in (6)
8. Go back to (2) until the number of features removed is less than ten percent of the total.
9. Method returns the features remaining once completed.


## Algorithm Performance  

BoostARoota is shorted to BAR and the below table is utilizing the LSVT dataset from the UCI datasets.  The algorithm has been tested on other datasets.   If you are interested in the specifics of the testing please take a look at the testBAR.py script.  The basics are that it is run through 5-fold CV, with the model selection performed on the training set and then predicting on the heldout test set.  It is done this way to avoid overfitting the feature selection process.

All tests are run on a 12 core Intel i7. - Future iterations will compare run times on a 28 core Xeon, 120 cores on Spark, and running xgboost on a GPU.

|Data Set | Target | Boruta Time| BoostARoota Time |BoostARoota LogLoss|Boruta LogLoss|All Features LogLoss| BAR >= All |
| ------- | ------ | -----------| ---- | ---- | ---- | ---- | ---- |
|[LSVT](https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation)  |0/1|50.289s  | 0.487s   | 0.5617 | 0.6950 | 0.7311 | Yes |
|[HR](https://www.kaggle.com/ludobenistant/hr-analytics) |0/1| 33.704s  | 0.485s   | 0.1046 | 0.1003 | 0.1047 | Yes |
|[Fraud](https://www.kaggle.com/dalpozz/creditcardfraud) |0/1| 38.619s  | 1.790s   | 0.4333 | 0.4353 | 0.4333 | Yes |

As can be seen, the speed up from BoostARoota is around 100x with substantial reductions in log loss.  Part of this speed up is that Boruta is running single threaded, while BoostARoota (on XGB) is running on all 12 cores.  Not sure how this time speed up works with larger datasets as of yet.  

This has also been tested on [Kaggle's House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/submissions?sortBy=date&group=all&page=1).  With nothing done except running BoostARoota and evaluated on RMSE, all features scored .15669, while BoostARoota scored 0.1560. 

## Future Functionality (i.e. Current Shortcomings)
The text file `FS_algo_basics.txt` details how I was thinking through the algorithm and what additional functionality was thought about during the creation.
 * Currently, you must pass in a OHE DataFrame. Would like to add more "smarter" reductions
   * Ex/ Pass in categorical variables and keeps/drops all levels of the variable (rather than just dropping some dummy variable/levels)
   * Ex/ Pass in categorical, drops some levels and returns dataframe in the same form
     * Have run into problems with dimensions and names differing - would like to fix this
 * Preprocessing Steps - Need some first pass filters for reducing dimensionality right off the bat
   * Check and drop _identical_ features, leaving option to drop highly correlated variables
   * Drop variables with near-zero-variance to target variable (creating threshold will be difficult)
   * LDA, PCA, PLS rankings 
     * Challenge with these is they remove based on linear relationships whereas trees are able to pick out the non-linear relationships and a variable with a low linear dependency may be powerful when combined with others.
   * t-SNE - Has shown some promise in high-dimensional data
 * Algorithm needs a better stopping criteria
   * Next step is to test it against Y and the eval_metric to see when it is falling off.
 * Expand compute to handle larger datasets (if user has the hardware)
   * Run on PySpark: make it easy enough that can just pass in SparkContext - will require some refactoring
   * Run XGBoost on GPU - although may run into memory issues with the shadow features.
   
## Updates
* 9/22/17 - Uploaded to PyPI and expanded tests
* 9/8/17 - Added Support for multi-class classification, but only for logloss.  Need to pass in eval="mlogloss"
* 9/6/17 - have implemented in BoostARoota2() a stopping criteria specifying that at least 10% of features need to be dropped to continue.
* 8/25/17 - The testBAR.py testing framework was just completed.

## Want to Contribute?

This project has found some initial successes and there are a number of directions it can head.  It would be great to have some additional help if you are willing/able.  Whether it is directly contributing to the codebase or just giving some ideas, any help is appreciated.  The goal is to make the algorithm as robust as possible.  The primary focus right now is on the components under Future Implementations, but are in active development.  Please reach out to see if there is anything you would like to contribute in that part to make sure we aren't duplicating work.  
