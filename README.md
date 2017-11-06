# BoostARoota  
A Fast XGBoost Feature Selection Algorithm  

## Why Create Another Algorithm?  
Automated processes like Boruta showed early promise as they were able to provide superior performance with Random Forests, but has some deficiencies including slow computation time: especially with high dimensional data. Regardless of the run time, Boruta does perform well on Random Forests, but performs poorly on other algorithms such as boosting or neural networks. Similar deficiencies occur with regularization on LASSO, elastic net, or ridge regressions in that they perform well on linear regressions, but poorly on other modern algorithms.

I am proposing and demonstrating a feature selection algorithm (called BoostARoota) in a similar spirit to Boruta utilizing XGBoost as the base model rather than a Random Forest. The algorithm runs in a fraction of the time it takes Boruta and has superior performance on a variety of datasets.  While the spirit is similar to Boruta, BoostARoota takes a slightly different approach for the removal of attributes that executes much faster.

## Installation 
Easiest way is to use `pip`:
```
$ pip install boostaroota
```

## Usage

This module is built for use in a similar manner to sklearn with `fit()`, `transform()`, etc.  In order to use the package, it does require X to be one-hot-encoded(OHE), so using the pandas function `pd.get_dummies(X)` may be helpful as it determines which variables are categorical and converts them into dummy variables.  This package does rely on pandas under the hood so data must be passed in as a pandas dataframe.

Assuming you have X and Y split, you can run the following:  
```
from boostaroota import BoostARoota
import pandas as pd

#OHE the variables - BoostARoota may break if not done
x = pd.getdummies(x)
#Specify the evaluation metric: can use whichever you like as long as recognized by XGBoost
  #EXCEPTION: multi-class currently only supports "mlogloss" so much be passed in as eval_metric
br = BoostARoota(metric='logloss')

#Fit the model for the subset of variables
br.fit(x,y)

#Can look at the important variables - will return a pandas series
br.keep_vars_

#Then modify dataframe to only include the important variables
br.transform(x)

```

It's really that simple!  Of course, as we build more functionality there may be a few more Keep in mind that since you are OHE, if you have a numeric variable that is imported by python as a character, pd.get_dummies() will convert those numeric into many columns.  This can cause your DataFrame to explode in size, giving unexpected results and high run times.

You can also view a complete demo [here.](https://github.com/chasedehan/BoostARoota/blob/master/odsc_west/demo.py)

## Usage - Choosing Parameters

The default parameters are optimally chosen for the widest range of input dataframes.  However, there are cases where other values could be more optimal.

* cutoff [default=4] - float (cutoff > 0)
  * Adjustment to removal cutoff from the feature importances
    * Larger values will be more conservative - if values are set too high, a small number of features may end up being removed.
    * Smaller values will be more aggressive; as long as the value is above zero (can be a float)
* iters [default=10] - int (iters > 0)
  * The number of iterations to average for the feature importances
    * While it will run, don't want to set this value at 1 as there is quite a bit of random variation
    * Smaller values will run faster as it is running through XGBoost a smaller number of times
    * Scales linearly. iters=4 takes 2x time of iters=2 and 4x time of iters=1
* max_rounds [default=100] - int (max_rounds > 0)
  * The number of times  the core BoostARoota algorithm will run.  Each round eliminates more and more features
    * Default is set high enough that it really shouldn't be reached under normal circumstances
    * You would want to set this value low if you felt that it was aggressively removing variables.
* delta [default=0.1] - float (0 < delta <= 1)
  * Stopping criteria for whether another round is started
    * Regardless of this value, will not progress past max_rounds
    * A value of 0.1 means that at least 10% of the features must be removed in order to move onto the next round
    * Setting higher values will make it more difficult to move to follow on rounds (ex. setting at 1 guarantees only one round)
    * Setting too low of a delta may result in eliminating too many features and would be constrained by max_rounds
* silent [default=False] - boolean
  * Set to True if don't want to see the BoostARoota output printed. Will still show any errors or warnings that may occur.

## How it works  
Similar in spirit to Boruta, BoostARoota creates shadow features, but modifies the removal step.

1. One-Hot-Encode the feature set
2. Double width of the data set, making a copy of all features in original dataset
3. Randomly shuffle the new features created in (2).  These duplicated and shuffled features are referred to as "shadow features"
4. Run XGBoost classifier on the entire data set ten times.  Running it ten times allows for random noise to be smoothed, resulting in more robust estimates of importance. The number of repeats is a parameter than can be changed.
5. Obtain importance values for each feature.  This is a simple importance metric that sums up how many times the particular feature was split on in the XGBoost algorithm.
6. Compute "cutoff": the average feature importance value for all shadow features and divide by four.  Shadow importance values are divided by four (parameter can be changed) to make it more difficult for the variables to be removed.  With values lower than this, features are removed at too high of a rate.
7. Remove features with average importance across the ten iterations that is less than the cutoff specified in (6)
8. Go back to (2) until the number of features removed is less than ten percent of the total.
9. Method returns the features remaining once completed.


## Algorithm Performance  

BoostARoota is shorted to BAR and the below table is utilizing the LSVT dataset from the UCI datasets.  The algorithm has been tested on other datasets.   If you are interested in the specifics of the testing please take a look at the testBAR.py script.  The basics are that it is run through 5-fold CV, with the model selection performed on the training set and then predicting on the heldout test set.  It is done this way to avoid overfitting the feature selection process.

All tests are run on a 12 core (hyperthreaded) Intel i7. - Future iterations will compare run times on a 28 core Xeon, 120 cores on Spark, and running xgboost on a GPU.

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
 * Algorithm could use a better stopping criteria
   * Next step is to test it against Y and the eval_metric to see when it is falling off.
 * Expand compute to handle larger datasets (if user has the hardware)
   * Run on PySpark: make it easy enough that can just pass in SparkContext - will require some refactoring
   * Run XGBoost on GPU - although may run into memory issues with the shadow features.
   
## Updates
* 10/26/17 - Modified Structure to resemble sklearn classes and added tuning parameters.
* 9/22/17 - Uploaded to PyPI and expanded tests
* 9/8/17 - Added Support for multi-class classification, but only for the logloss eval_metric.  Need to pass in eval="mlogloss"
* 9/6/17 - have implemented in BoostARoota2() a stopping criteria specifying that at least 10% of features need to be dropped to continue.
* 8/25/17 - The testBAR.py testing framework was just completed and ran through a number of datasets

## Want to Contribute?

This project has found some initial successes and there are a number of directions it can head.  It would be great to have some additional help if you are willing/able.  Whether it is directly contributing to the codebase or just giving some ideas, any help is appreciated.  The goal is to make the algorithm as robust as possible.  The primary focus right now is on the components under Future Implementations, but are in active development.  Please reach out to see if there is anything you would like to contribute in that part to make sure we aren't duplicating work.  

## Current Contributors
* Chase DeHan
* Zach Riddle

A special thanks to [Progressive Leasing](http://progleasing.com) for sponsoring this research.
