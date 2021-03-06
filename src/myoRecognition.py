#!/usr/bin/python
import pandas as pd
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

def power_bit_length(x):
    return 2**(x-1).bit_length()

def fft_feature(df, ws):
    hamm_wind = np.hamming(ws)
    tmp = np.absolute(np.fft.fft(df.multiply(hamm_wind, axis=0)) / sum(hamm_wind)  )
    return tmp 

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
        col = df[:, i]
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
    global count_imu
    global dx_
    global dy_
    global roll_
    global pitch_
    global yaw_
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
    global OSC_Client

    dx_ += float(data[-2])
    dy_ += float(data[-1])
    roll_ += float(data[6])
    pitch_ += float(data[7])
    yaw_ += float(data[8])
    count_imu += 1
    if count_imu == 8:
        
        oscmsg = OSC.OSCMessage()
        oscmsg.setAddress("/imu")

        oscmsg.append([dx_/1200, dy_/1200, (roll_-68)/68, (pitch_-68)/68, (yaw_-68)/68] )
        OSC_Client.send(oscmsg)

        # oscmsg = OSC.OSCMessage()
        # oscmsg.setAddress("/dy")
        # oscmsg.append(dy_/4)
        # OSC_Client.send(oscmsg)

        # oscmsg = OSC.OSCMessage()
        # oscmsg.setAddress("/roll")
        # oscmsg.append(roll_/4)
        # OSC_Client.send(oscmsg)

        # oscmsg = OSC.OSCMessage()
        # oscmsg.setAddress("/pitch")
        # oscmsg.append(pitch_/4)
        # OSC_Client.send(oscmsg)

        # oscmsg = OSC.OSCMessage()
        # oscmsg.setAddress("/yaw")
        # oscmsg.append(yaw_/4)
        # OSC_Client.send(oscmsg)

        count_imu =0
        dx_ = 0
        dy_ = 0
        roll_ = 0
        pitch_ = 0
        yaw_ = 0
    return

def EMGHandler_realtime(addr, tags, data, client_address):
    global step_size
    global start
    global end
    global emgdf
    global pca_
    global model_
    global index_label
    global OSC_Client
    global flag    
    #txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    vtime = datetime.datetime.now()
    stime = vtime.strftime("%H:%M:%S.%f")  
    #txt += stime
    #txt += str(data)

    emgdf.ix[stime] = data
    if end < emgdf.shape[0]:
        # tmp = np.concatenate((emgdf.ix[start:end].mean(0), emgdf.ix[start:end].median(0), emgdf.ix[start:end].var(0),  count_mean_crossing(emgdf.ix[start:end]) ))
        tmp = emgdf.ix[start:end]
        tmp_fft = fft_feature(tmp, window_size)

        tmp_sample = np.concatenate((tmp.mean(0), tmp.median(0), tmp.var(0), count_mean_crossing(tmp.as_matrix()), tmp_fft.mean(0), np.median(tmp_fft, axis=0), tmp_fft.var(0),  count_mean_crossing(tmp_fft) ))
        tmp_sample = np.array(tmp_sample).reshape(1, tmp_sample.shape[0] )
        tmp_pca =  pca_.transform(tmp_sample)
        test_res = model_.predict(tmp_pca)
        res = int(test_res[0])

        flag = not flag

        if flag:
            oscmsg = OSC.OSCMessage()
            oscmsg.setAddress("/gesture")
            oscmsg.append(res)
            OSC_Client.send(oscmsg)


        #print 'test_res: ', test_res
        print bcolors.OKBLUE + '\npredict: ' + index_label[res] + bcolors.ENDC

        # prediction_bin[bin_count] = int(test_res[0] )
        # bin_count += 1
        # if bin_count>5: 
        #     bin_count = 0
        # print 'prediction_bin: ', prediction_bin    
        # res = np.argmax(np.bincount(prediction_bin)) 
        # print 'res: ', res    
        # print bcolors.OKGREEN + 'predict: ' + index_label[res] + bcolors.ENDC
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
    window_size = 256 
    step_size = 64
    if len(sys.argv) > 2:
        window_size = int(float(sys.argv[2]) * 200) # frequency is 200
    if len(sys.argv) > 3:
        step_size = int(float(sys.argv[3]) * 200)

    count_imu = 0
    dx_ = 0
    dy_ = 0
    roll_ = 0
    pitch_ = 0
    yaw_ = 0

    flag = False

    window_size = power_bit_length(window_size)
    step_size  = power_bit_length(step_size)
        
    label_index = {'emg_idle_s':0, 'emg_index_s': 1, 'emg_index_d': 1, 'emg_ring_s' :3, \
    'emg_spread_s':5, 'emg_wavein_s' :6, \
    'emg_waveout_s' :7, 'emg_fist_s' :8, 'emg_fist_d' :8,  'emg_doubleTapping_d' :9}  # label_index


    index_label = dict()
    for key, val in label_index.items():
        index_label[val] = key 

    print 'window_size: ', window_size, ' step_size, ', step_size   
    emg_header = ['em1', 'em2', 'em3', 'em4', 'em5', 'em6', 'em7', 'em8']
    stat_feature = ['_mean', '_median', '_var']
    emgdf = pd.DataFrame(columns= emg_header)
    
    start = 0
    end = window_size
    [pca_, model_] = pickle.load( open(model_file, "rb" ) )

    OSC_Client = OSC.OSCClient()
    OSC_Client.connect(('127.0.0.1', 8889))

    get_stream()


    


