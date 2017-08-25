# BoostARoota
A Fast XGBoost Feature Selection Algorithm

## Why Create Another Algorithm?
Automated processes like Boruta showed early promise as they were able to provide superior performance with Random Forests, but has some deficiencies including slow computation time: especially with high dimensional data. Regardless of the run time, Boruta does perform well on Random Forests, but performs poorly on other algorithms such as boosting or neural networks. Similar deficiencies occur with regularization on LASSO, elastic net, or ridge regressions in that they perform well on linear regressions, but poorly on other modern algorithms.

I am proposing and demonstrating a feature selection algorithm (called BoostARoota) in a similar spirit to Boruta utilizing XGBoost as the base model rather than a Random Forest. The algorithm runs in a fraction of the time it takes Boruta and has superior performance on a variety of datasets.  While the spirit is similar to Boruta, BoostARoota takes an approach 

## How to use the code



## How it works

This section is coming - the approach is evolving quickly.

## Algorithm Performance

BoostARoota is shorted to BAR and the below table is utilizing the LSVT dataset from the UCI datasets.  The algorithm has been tested on other datasets.  The testBAR.py testing framework was just completed (8/25/2017) and will be running many others through.  If you are interested in the specifics of the testing please take a look at the testBAR.py script.  The basics are that it is run through 5-fold CV, with the model selection performed on the training set and then predicting on the heldout test set.  It is done this way to avoid overfitting the feature selection process.

All tests are run on a 12 core Intel i7.

Iteration | Boruta Time| BAR Time |Boruta LogLoss|BAR LogLoss|All Features LL|
| ------- | -----------| ---- | ---- | ---- | ---- |
|      1  | 47.17s  | 0.52s   | 0.69 | 0.46 | 0.73 |
|      2  | 47.08s  | 0.69s   | 0.69 | 0.48 | 0.73 |
|      3  | 46.77s  | 0.47s   | 0.69 | 0.49 | 0.73 |

As can be seen, the speed up from BoostARoota is around 100x with substantial reductions in log loss.  Part of this speed up is that Boruta is running single threaded, while BoostARoota (on XGB) is running on all 12 cores.  Not sure how this time speed up works with larger datasets as of yet.

## Future Implementations (i.e. Current Shortcomings)
 


## Want to Contribute?

This project has found some initial successes and there are a number of directions it can head.  It would be great to have some additional help if you are willing/able.  Whether it is directly contributing to the codebase or just giving some ideas, any help is appreciated.  The goal is to make the algorithm as robust as possible.  The primary focus right now is on the components under Future Implementations, but are in active development.  Please reach out to see if there is anything you would like to contribute in that part to make sure we aren't duplicating work.  
