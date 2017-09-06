data_path = "~/DDE/Chase/Data/"
#################
#LSVT Voice Rehab data
    # Data came from here: https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation
lsvt = pd.read_csv(data_path + "LSVT_VR.csv")

#Split and make appropriate transformations
lsvt_X = lsvt[lsvt.columns[1:lsvt.shape[1]]].copy()
lsvt_Y = lsvt[lsvt.columns[0]].copy() - 1
lsvt_X = GetCatDummies(lsvt_X)
del lsvt

names, i  = BoostARoota2(lsvt_X, lsvt_Y, metric='logloss')

lsvt_results = testBARvSelf(lsvt_X, lsvt_Y, eval='class')
print(lsvt_results)

