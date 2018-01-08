#dask_testing.py
from boostaroota import BoostARoota
from dask.distributed import Client, progress
import dask.dataframe as dd #using dd rather than pandas
import dask_xgboost as dxgb
from dask import compute, persist
import pandas as pd
import urllib

########################################################################################################################
#
#  Set up dask client
#
########################################################################################################################
client = Client()
client

########################################################################################################################
#
#  Bring in the data
#
########################################################################################################################
#Download data from UCI as pandas then convert it to a dask dataframe

train = dd.read_csv("train.csv")
y = train['SalePrice']
del train['SalePrice']

train, y = persist(train, y)
progress(train, y)

df2 = dd.get_dummies(train.categorize()).persist()


########################################################################################################################
#
#  Run an XGBoost using dask
#
########################################################################################################################
params = {'eval_metric': 'rmse'}  # use normal xgboost params

bst = dxgb.train(client, params, df2, y)



