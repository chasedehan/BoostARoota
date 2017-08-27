# BoostARoota  
A Fast XGBoost Feature Selection Algorithm  

## Why Create Another Algorithm?  
Automated processes like Boruta showed early promise as they were able to provide superior performance with Random Forests, but has some deficiencies including slow computation time: especially with high dimensional data. Regardless of the run time, Boruta does perform well on Random Forests, but performs poorly on other algorithms such as boosting or neural networks. Similar deficiencies occur with regularization on LASSO, elastic net, or ridge regressions in that they perform well on linear regressions, but poorly on other modern algorithms.

I am proposing and demonstrating a feature selection algorithm (called BoostARoota) in a similar spirit to Boruta utilizing XGBoost as the base model rather than a Random Forest. The algorithm runs in a fraction of the time it takes Boruta and has superior performance on a variety of datasets.  While the spirit is similar to Boruta, BoostARoota takes an approach 

## How to use the code  
The code is currently not in a package to `pip install` so you will need to manually run the BoostARoota.py file.  

The only function you _have_ to have is `BoostARoota(X, Y, metric)`.  In order to use the function, it does require X to be one-hot-encoded(OHE), so `GetCatDummies(X, ReplaceNA=True)` may be helpful as it determines which variables are categorical and converts them into dummy variables.  

Assuming you have X and Y split, you can run the following:  
```
#Specify the evaluation metric: can use whichever you like as long as recognized by XGBoost
eval_metric = 'logloss' 
#OHE the Predictors
X = GetCatDummies(X)

#Run BoostARoota - will return a list of variables
BR_vars = BoostARoota(X, Y, metric=eval_metric)

#Reduce X to only include those deemed as import to BoostARoota
BR_X = X[BR_vars].copy()
```

It's really that simple!  Of course, as we build more functionality into it, it will get to be more difficult.  Keep in mind that since you are OHE, if you have a numeric variable that is imported by python as a character, GetCatDummies() will convert those numeric into many columns.  This can cause your DataFrame to explode in size, giving unexpected results and high run times.


## How it works  

This section is coming - the approach is evolving quickly.  You can definitely look at the code to see how it is working.

## Algorithm Performance  

BoostARoota is shorted to BAR and the below table is utilizing the LSVT dataset from the UCI datasets.  The algorithm has been tested on other datasets.  The testBAR.py testing framework was just completed (8/25/2017) and will be running many others through.  If you are interested in the specifics of the testing please take a look at the testBAR.py script.  The basics are that it is run through 5-fold CV, with the model selection performed on the training set and then predicting on the heldout test set.  It is done this way to avoid overfitting the feature selection process.

All tests are run on a 12 core Intel i7. - Future iterations will compare run times on a 28 core Xeon, 120 cores on Spark, and running xgboost on a GPU.

Iteration | Boruta Time| BAR Time |Boruta LogLoss|BAR LogLoss|All Features LogLoss|
| ------- | -----------| ---- | ---- | ---- | ---- |
|      1  | 47.17s  | 0.52s   | 0.69 | 0.46 | 0.73 |
|      2  | 47.08s  | 0.69s   | 0.69 | 0.48 | 0.73 |
|      3  | 46.77s  | 0.47s   | 0.69 | 0.49 | 0.73 |

As can be seen, the speed up from BoostARoota is around 100x with substantial reductions in log loss.  Part of this speed up is that Boruta is running single threaded, while BoostARoota (on XGB) is running on all 12 cores.  Not sure how this time speed up works with larger datasets as of yet.

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
 * Algorithm is a single pass through 10 iterations
   * This needs to be more flexible and have more iterations to reduce dimensions further than it currently is. 
   * Haven't been able to work out a solid stopping criteria yet.
 * Expand compute to handle larger datasets (if user has the hardware)
   * Run on PySpark: make it easy enough that can just pass in SparkContext - will require some refactoring
   * Run XGBoost on GPU - although may run into memory issues with the shadow features.
   


## Want to Contribute?

This project has found some initial successes and there are a number of directions it can head.  It would be great to have some additional help if you are willing/able.  Whether it is directly contributing to the codebase or just giving some ideas, any help is appreciated.  The goal is to make the algorithm as robust as possible.  The primary focus right now is on the components under Future Implementations, but are in active development.  Please reach out to see if there is anything you would like to contribute in that part to make sure we aren't duplicating work.  
