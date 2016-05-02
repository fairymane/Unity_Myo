#!/usr/bin/python
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import time
import datetime 
import sys
import threading
import random
from sklearn import preprocessing, svm, tree
from sknn.mlp import Classifier, Layer
#from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from hmmlearn import hmm
import math
import pickle
import OSC
import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def count_mean_crossing(df):
    """
    count_mean_crossing: count number of mean crossing on each dimension of ndarrary or DataFrame
    Parameters:
        df: DataFrame
    Return:
        array type, number of mean crossing on each dimension
    """
    tmp_df = df - df.mean(0)
    d = df.shape[1]
    res = np.zeros(d)
    for i in xrange(d) :
        col = df.ix[:, i].values
        res[i] = np.count_nonzero(np.diff(np.sign(col))) 
    return res
          
def handler_realtime(addr, tags, data, client_address):
    #txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    #vtime = datetime.datetime.now()
    #stime = vtime.strftime("%Y-%m-%d %H:%M:%S ") 
    #stime = vtime.strftime("%H:%M:%S.%f") 
    #txt += stime
    #txt += str(data)
    #print time.time()
    #print(txt)
    return

def IMGHandler_realtimg(addr, tags, data, client_address):
    #global count_img
    #count_img += 1
    #if count_img > 300:
    #    sys.exit()
    #txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    #vtime = datetime.datetime.now()
    #stime = vtime.strftime("%H:%M:%S.%f") 
    #txt += stime
    #txt += str(data)
    #imgdf.ix[stime] = data;
    #print(txt)
    return

def EMGHandler_realtime(addr, tags, data, client_address):
    global step_size
    global start
    global end
    global emgdf
    global pca_
    global model_
    global index_label
    global prediction_bin
    global bin_count
    #txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    vtime = datetime.datetime.now()
    stime = vtime.strftime("%H:%M:%S.%f")  
    #txt += stime
    #txt += str(data)
    

    emgdf.ix[stime] = data
    if end < emgdf.shape[0]:
        tmp = np.concatenate((emgdf.ix[start:end].mean(0), emgdf.ix[start:end].median(0), emgdf.ix[start:end].var(0),  count_mean_crossing(emgdf.ix[start:end]) ))
        tmp_pca =  pca_.transform(tmp)
        test_res = model_.predict(tmp_pca)
        prediction_bin[bin_count] = int(test_res[0] )
        bin_count += 1
        if bin_count>5:
            bin_count = 0
        res = np.argmax(np.bincount(prediction_bin))    
        print bcolors.OKGREEN + 'predict: ' + index_label[res] + bcolors.ENDC
        start += step_size
        end += step_size


def get_stream():
    s = OSC.OSCServer(('127.0.0.1', 8888))  # listen on localhost, port 57120
    s.addMsgHandler('/myo/IMG', IMGHandler_realtimg) 
    s.addMsgHandler('/myo/emg', EMGHandler_realtime)     # call handler() for OSC messages received with the /startup address
    s.addMsgHandler('/myo/pose', handler_realtime)     # call handler() for OSC messages received with the /startup address
    s.serve_forever()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print 'require 2~4 arguments. e.g.: \n python myoRecogintion.py model_file(pickle) ws(recoginition window size of ws second, with 200 freqency, default by 1 if not specified) ss (step size, defult by 0.25 if not specified)'
        sys.exit()
    model_file   = sys.argv[1]
    window_size = 200 
    step_size = 50
    if len(sys.argv) > 2:
        window_size = int(sys.argv[2]) * 200 # frequency is 200
    if len(sys.argv) > 3:
        step_size = int(sys.argv[3]) * 200

    label_index = {'emg_index_test': 1, 'emg_middle_test':2, 'emg_ring_text' :3, 'emg_little_test': 4, 'emg_spread_test':5, 'emg_idle_test':6} 
    index_label = {1 : 'emg_index_test', 2: 'emg_middle_test', 3: 'emg_ring_test', 4: 'emg_little_test', 5: 'emg_spread_test', 6: 'emg_idle_test'}
    emg_header = ['em1', 'em2', 'em3', 'em4', 'em5', 'em6', 'em7', 'em8']
    stat_feature = ['_mean', '_median', '_var', '_meanCrossCount']
    emgdf = pd.DataFrame(columns= emg_header)
    
    start = 0
    end = window_size
    prediction_bin = np.array([6, 6, 6, 6, 6, 6])
    bin_count = 0
    [pca_, model_] = pickle.load( open(model_file, "rb" ) )

    get_stream()


    


