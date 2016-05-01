import pickle
import pandas as pd
import sys


df_1 = pickle.load(open(sys.argv[1], "rb"))
#df_2 = pickle.load(open(sys.argv[2], "rb"))

for i in range(15):
    df_tmp1 = df_1.ix[i*200 :(i+1)*200]
    #df_tmp2 = df_2.ix[i*200 :(i+1)*200]
    #print df_tmp.mean()
    print df_tmp1.var()

