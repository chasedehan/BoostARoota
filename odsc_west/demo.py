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
print("Original training set has " + str(train.shape) + " dimensions. \n" +\
"BoostARoota with .fit() and .transform() reduces to " + str(new_train.shape) + " dimensions. \n" +\
"BoostARoota with .fit_transform() reduces to " + str(new_train2.shape) + " dimensions.\n" +\
"The two methods may give a slightly different dimensions because of random variation as it is being refit")


########################################################################################################################
#
#  Test that its working with any classifier
#
########################################################################################################################
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier()
br = BoostARoota(metric=None, clf=clf)
new_train = br.fit_transform(train, labels)