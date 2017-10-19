import pandas as pd
import boostaroota as br

#################
#LSVT Voice Rehab data
    # Data came from here: https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation
lsvt = pd.read_csv('//pf.stormwind.local/DDE/Chase/Data/lsvt_VR.csv')

#Split and make appropriate transformations
lsvt_X = lsvt[lsvt.columns[1:lsvt.shape[1]]].copy()
lsvt_Y = lsvt[lsvt.columns[0]].copy() - 1
lsvt_X = pd.get_dummies(lsvt_X)
del lsvt

#Run BoostARoota
names = br.BoostARoota(lsvt_X, lsvt_Y, metric='logloss')

#Reduce the data frame
new_X = lsvt_X[names].copy()

#Dimension Reduction
print("lsvt_X has " + str(lsvt_X.shape) + " dimensions.")
print("BoostARoota reduces to " + str(new_X.shape) + " dimensions.")


