#testTimings.py
"""
This script is to develop the timings for a variety of dataframes
Should be conducted in a similar spirit to that described in the original Boruta Paper


OLD - has the older call to BR
"""

import pandas as pd
import numpy as np
import urllib
from boostaroota import BoostARoota
import time

########################################################################################################################
#
#Bring in data
#
########################################################################################################################
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
# Run the timing
#
########################################################################################################################
#just to check and see that it works
#br.BoostARoota(train,labels,metric='logloss')

attributes = np.arange(500,4001,500)
objects = np.arange(250,2001,250)
combos = [(x,y) for x in attributes for y in objects]

# attributes = np.arange(1000,1001,500)
# objects = np.arange(250,501,250)
# combos = [(x,y) for x in attributes for y in objects]
def oneIteration():
    timings = []
    for att, obj in combos:
        #subsample the objects/observations
        this_df = train.sample(n=obj).copy()
        this_y = labels.loc[this_df.index.values].copy()
        # this_df.reset_index(inplace=True, drop=True)
        # this_y.reset_index(inplace=True, drop=True)
        #extend the attributes
        if att != 500:
            extraAttLen = att - 500
            extraAttributes = pd.DataFrame(np.random.randint(0, 100, size=(obj, extraAttLen)),
                                           columns=['Ext'+ str(x) for x in range(extraAttLen)])
            #Bind the dataframes together
            bound = pd.concat([this_df, extraAttributes],axis=1,ignore_index=True)

        else:
            bound = this_df.copy()
        start = time.time()
        br = BoostARoota(metric='logloss')
        br.fit(bound,this_y)
        timings.append(time.time()-start)
    return timings


timingDF = pd.DataFrame({'Attributes': [x for x,y in combos],
                         'Objects': [y for x,y in combos]})

#Run through the timing structure 3 times
for i in range(10):
    timingDF['N'+str(i)] = oneIteration()

timingDF.to_csv('timings.csv')



