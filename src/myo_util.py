#!/usr/bin/python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import threading
import random
import Queue
from sklearn import preprocessing, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import math
import pickle
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




def generate_stream(logfile, df):
    """
    generate_stream generate stream from a log file in real time manner
    Parameters:
        logfile: the input file that convert to DataFrame (df) and use as a real time stream,
                    the logfile could be an offline file or a file that constantly being written 
                    other program (e.g. hello-myo )
        df: the DataFrame that holds all the updates from logfile, df will be shared with other
                    thread(s)

    """
    print 'start stream+++++++++++'
    i = 0    
    while 1:
        where = logfile.tell()
        line = logfile.readline()
        if not line:
            print 'not line, stream waiting'
            time.sleep(0.5)
            logfile.seek(where)
            i += 1 
        else:
            ll = line.strip(' \r\n').split(',')
            df.loc[ll[0]] = ll[1:11]
            i = 0
            #print df.ix[-1]

        if i == 20:
            break
    
def normalization(arr):
    """
    normalization: scale arrary or DataFrame to mean zero and range from -1 to 1    
    Parameters:
        arr: numpy arrary or pandas DataFrame
    Return:
        scaled arrary value
    """
    return (arr - arr.mean(0)) / (arr.max(0) - arr.min(0))


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




def calculate_stat(df):
    """
    calculate_stat: calculate different statistics of given DataFrame, this function is used to generate 
                    features (list of feature as a sample point) for support vector machine
    Parameter:
        df: DataFrame type, usually df is a sliding window in this function 
    Return:
        right now for each input sliding window of 9 dimension, calculate_stat will calculate 9*4 = 36 features
        i.e. [mean, median, var, number_of_mean_crossing ] for each dimension [accel_xyz, gyro_xyz, row, pitch, yaw]   

    """
    #df = normalization(df)
    mean = df.mean(0).values
    median = df.median(0).values
    var = df.var(0).values
    mean_crossing_count = count_mean_crossing(df)
    #print 'mean_crossing_count: ', mean_crossing_count
    res = np.concatenate((mean, median, var,  mean_crossing_count))
    #print 'feature size', res.shape, 'feature: \n', res
    return res


def get_sliding_window(df, ws, get_traning_data = False, label = -1,  \
                        n_sample = 45, pattern_name = 'NA', output_format = 'hdf', \
                        save_data = 'train', aws = 120, ss = None, fs = 20):
    """
    get_sliding_window: 1. generate sliding window from datastream in real time manner
                        2. generate training feature samples for SVM (when get_traning_data is True) or testing the 
                            real time stream by recognizing the current unit patterns
                        3. store the generated feature sampels (store in HDF5 when output_format = 'hdf', in csv file when output_format = 'csv')
    Parameters:
    df is the datastream
    ws windows size, here us is the time duration, e.g. 2 seconds
    if get_traning_data set True, training phase. get_sliding_window() generates sliding window from datastream, calculate statistics (features) for each sliding window,
                those samples are used as training sample to train a PCA model and a SVM model, and store training feature samples in either 'hdf5' or 'csv'
    if get_traning_data set False, testing phase. get_sliding_window() generates sliding window from datastream, calculate statistics (features) for each sliding window,
                those samples are used as testing samples that apply to PCA and SVM model generated above for predicting unit_patterns in real time manner
                right now we apply every accumulated 15 testing samples with PCA and SVM due to the consideration of scaling generated statistics (features)
    label: the label corresponding to the unit_patterns (for training purpose), effective only in the training phase
    n_sample: number of training samples to generate, effective only in the training phase, default to 45
    output_format: right now support save as 'hdf5' or 'csv'
    save_data:
        'train' save collected training samples, effective only in the training phase, right now only support 'hdf5'  
        'train_append' if user want to append training samples if the corresponding DataFrame already exists,
             effective only in the training phase, right now only support 'hdf5'
        'test'  save collected testing samples, effective only in the training phase, right now only support 'hdf5'
        'test_simple' save collected testing samples with only time index with label and label_index, 
                effective only in the training phase, right now only support 'hdf5'
    aws: activity window size, default 120, i.e. the system will predict the current activity every every 120 sliding_window         
    ss is the step size in time duration, e.g. when ws = 4 seconds, ss = 1 second, sliding windows will have 75 percent overlapping
    fs is the sample frequency, right now it is 20 samples/second generated by hello-myo.cpp

    """

    print '+++++++++++++++++++start sliding windows+++++++++++++++++'
    if ss is None:
    # ss was not provided. the windows will not overlap in any direction.
        ss = ws * 0.25

    ss = int(ss * fs)
    ws = int(ws * fs)    
    argv_list = ['roll_w', 'pitch_w', 'yaw_w','accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    start = 0
    end = start + ws
    #print 'ss: ', ss
    #print 'ws: ', ws


    #if get_traning_data:
    feature_header = ['label_index', 'label']
    stat_feature = ['_mean', '_median', '_var', '_meanCrossCount'] 
    #feature_list = np.zeros(len(argv_list)* len(stat_feature) )
    if n_sample <= 0:
        n_sample = 45

    for fe in stat_feature:
        for arg in argv_list:
            feature_header.append(arg+fe)

    #print 'feature_header: ', feature_header
    n_feature = len(feature_header)
    feature_header.append('estimate_label')
    feature_header.append('estimate_label_index')

    feature_list =  pd.DataFrame(columns= feature_header).dropna();
    feature_list.loc[0] = 0

    win_n = 0
    i = 0
    real_stream_start = 0
    real_stream_end = 0
    real_ws = 5
    epoch_time = 0
    format_time_t = ''


    if save_data == 'test_simple':
        act_COUNT = 0
        act_wind_count = 0
        bow_features = pd.DataFrame(columns= ['start_time', 'end_time'] + label_index.keys()  )
        bow_features.loc[act_COUNT] = 0
    unit_pattern_list = []
    res_label = 'NA'    
    
    f_gui = open('to_gui.csv', 'w+')
    gui_string_ =  'time,estimated_unit_pattern,estimated_activity\n'
    f_gui.write(gui_string_)
    #try:
    if True:
        while 1:
            if end < df.shape[0]:
                
                sliding_window = df.iloc[start:end]

                #epoch_time = df.index[start]
                format_time_t = df['formatted_time'][start]

                if save_data == 'test_simple':

                    if act_wind_count == 0 :
                        bow_features.loc[act_COUNT]['start_time']  = format_time_t
                        act_wind_count +=1

                    elif act_wind_count == aws:
                        bow_features.loc[act_COUNT]['end_time'] = format_time_t

                        file_name =  'stream_' + pattern_name + '_.txt'

                        tmp_test = bow_features.loc[act_COUNT]
                        #print 'before: tmp_test :', tmp_test, '\n', tmp_test.shape
                        tmp_test = tmp_test.ix[2:].values
                        #print 'tmp_test :', tmp_test
                        test_label = -1

                        #print 'act_pca_: ', act_pca_

                        res_label_index = testing_accuracy(tmp_test, test_label, act_pca_, act_model_, activity = True )
                        #print 'res_label_index: ', res_label_index
                        res_label = act_index_label[res_label_index[0]] 

                        with open(file_name, 'wa') as f:
                            for item in unit_pattern_list:

                                f.write("%s, " % item)
                            f.write("\n\n")
                        unit_pattern_list = []
                        act_COUNT +=1
                        print '*************************act_COUNT: ', act_COUNT
                        bow_features.loc[act_COUNT] = 0
                        act_wind_count = 0
                    
                    else:
                        act_wind_count +=1

        
                #print '#####################act_COUNT: ', act_COUNT
                #print '#####################act_wind_count: ', act_wind_count
                sliding_window = sliding_window[argv_list].astype(np.float).dropna()
                
                print '$sliding_window number: $', win_n, ' index start: ', start, ' end: ', end#,  ':\n', sliding_window
                
                ### withou normlaization of the sliding window
                feature = calculate_stat(sliding_window)
                    #print 'first, feature size: ', feature.shape, '   feature: ', feature
                    #feature = np.insert(feature, 0, label)
                    #print 'feature size: ', feature.shape, '   feature: ', feature
                    #feature_list = np.vstack((feature_list, feature))
                    #print 'pattern_name: ', pattern_name

                if  pattern_name in label_index.keys() : 
                    feature_list.ix[win_n, 0:2] = [label_index[pattern_name], pattern_name]
                else:
                    #print 'feature_list: ', feature_list
                    #print 'feature_list.ix[win_n, 0:2]:   ', feature_list.ix[win_n, 0:2]
                    feature_list.ix[win_n, 0:2] = [-1, pattern_name]

                feature_list.ix[win_n, 2:n_feature] = feature

                #print 'feature_list.index: ', feature_list.index
                #print 'feature_list.index[win_n]: ', feature_list.index[win_n]

                #feature_list.index = format_time_t  

                win_n += 1

                if get_traning_data:
                    feature_list.ix[win_n, -2:] = 'NA'
                    if win_n == n_sample:
                        break

                else:
                    real_stream_end = win_n

                    if  real_stream_end - real_stream_start >= real_ws:

                        df_tmp = feature_list.ix[real_stream_start: real_stream_end ].convert_objects()

                        #print 'df_tmp&&&&&&&&&&&&&&&&&&&&&: ', df_tmp

                        [df_test, test_label] = get_sample_label( df_tmp )

                        #### double check wether you need scale df_test

                        est_label_k =  testing_accuracy(df_test, test_label, pca_, model_ )

                        #print '$$$$$$$$$$$ df_test shape: ', df_test.shape


                        """
                        ### K Means model
                        kmeans_test_pred = cluster_model_.predict(df_test) + 1                       
                        kmeans_distance = kmeans_dist(cluster_model_.cluster_centers_, df_test)
                        print '$$$$$$$$$$$$$$$$$$$$$$$$kmeans_distance: \n', kmeans_distance
                        kmeans_test_min_distance_index = np.argmin(kmeans_distance , axis=1) + 1
                        print '$$$$$$$$$$$$$$$$$$$$$$$kmeans_test_pred:', kmeans_test_pred
                        print '$$$$$$$$$$kmeans_test_min_distance_index ', kmeans_test_min_distance_index
                        """
                        ### GMM Model
                        #c_test_pred_prob =  cluster_model_.predict_proba(df_test)
                        #c_test_prob_label = np.argmax(c_test_pred_prob, axis=1) + 1
                        #c_test_prob_max = np.max(c_test_pred_prob, axis=1)
                        #print '$$$$$$$$$$$$$$$$$$ c_test_pred:          ', c_test_pred
                        #print '$$$$$$$$$$$$$$$$$$ c_test_prob_label:    ', c_test_prob_label
                        #print '$$$$$$$$$$$$$$$$$$ c_test_prob_max       ', c_test_prob_max 
                        #print '$$$$$$$$$$$$$$$$$$ c_test_pred_prob:   \n', c_test_pred_prob
                        #min_max = np.min(c_test_prob_max)
                        #print '&&&&&&&&&&&&&&&&&& min_max: ', min_max
                        #if min_max < 0.99:
                            #print '########################## NEW PATTERNS DETECTED ################################################################' 

                        print 'real_stream_start: real_stream_end+++++:', real_stream_start, ' : ', real_stream_end
                        #print 'time- before', format_time_t, ' est_label_k: ', est_label_k

                        if len(est_label_k) != real_ws :
                            print '!!!wrong lable size is suppose to have the same as real_ws = ', real_ws, ' len(est_label_k) :', len(est_label_k) 
                        #break
                        est_label_v = [index_label[x] for x in est_label_k]
                        print  'time', format_time_t,'\nEstimated label: \n', est_label_v
                        print '$$$$$$$$$$$$$$$$$$ Estimated label index ', est_label_k
                        
                        est_label_index_ = np.argmax(np.bincount(est_label_k.astype(int) ))
                        est_label_ = index_label[est_label_index_]

                        est_string = '@@@@@@@@@@@@@@@@@@@@@@Estimated unit_pattern: [' + est_label_ + '] \n \n\n@@@@@@@@@@@@@@@@@@@@@@@@@@Estimated activity: [' + res_label + ']\n' 

                        print '##########################################\n'
                        print bcolors.OKGREEN + est_string +  bcolors.ENDC
                        print '##########################################'

                        gui_string_ =  format_time_t+','+est_label_ +',' + res_label + '\n'
                        f_gui.write(gui_string_)


                        if save_data == 'test_simple':
                            #print 'bow_features: ', bow_features
                            #print 'act_COUNT: ', act_COUNT
                            for unit_pattern in est_label_v:
                                #print 'unit_pattern: ', unit_pattern
                                bow_features.loc[act_COUNT][unit_pattern] += 1
                            #print '&&&&&&&&&&&&&&&&&&&aha bow_features: \n', bow_features 

                        feature_list.ix[real_stream_start: real_stream_end+1, -2 ]= est_label_v
                        feature_list.ix[real_stream_start: real_stream_end+1, -1 ]= est_label_k
                        unit_pattern_list += est_label_v

                        real_stream_start = real_stream_end
                #print 'feature_list header: ', feature_list.columns

                start += ss
                end += ss
                i = 0
            else:
                print 'sliding window waiting for incomming datastream'
                time.sleep(0.5)
                i += 1
                if i == 25:
                    break      
            
        feature_list = feature_list.ix[3:]
        if save_data == 'test_simple':
            if bow_features.loc[act_COUNT]['end_time'] == 0:
                bow_features.loc[act_COUNT]['end_time'] = format_time_t

        if output_format == 'csv':
            if get_traning_data:
                file_name = pattern_name + '_train_sample.csv'
            else:
                file_name = pattern_name + '_bow_features.csv'
                
            #with open(file_name, 'a') as f:
            with open(file_name) as f:
                feature_list.to_csv(f)


        if output_format == 'hdf' :
            hdf = pd.HDFStore('../data/unit_patterns1.h5')
            print 'pattern_name: ', pattern_name
            print 'hdf.keys: ', hdf.keys() 

                
            #print  'feature_list size: ', feature_list.shape, '\nfeature_list columns$$$$$$$$$ \n', feature_list
            #print '\n\n\n +++++++++\n\n\n\n'

            print '+++++++++++++++saving+++++++++++++++'
                 
            if get_traning_data: 
                #pattern_name_ =  'train_' + pattern_name  
                if (pattern_name in hdf) and save_data == 'train_append':
                    feature_list = feature_list.append(hdf[pattern_name],ignore_index = True)
                    #print  '&&&&&&feature_list size:&&&&& ', feature_list.shape, 'feature_list columns$$$$$$$$$ \n', feature_list
 
                if save_data in ['train', 'train_append']:
                    feature_list = feature_list.convert_objects()
                    hdf.put(pattern_name, feature_list, format='table', data_columns=True)

            else:

                print 'aha stream_test_to_hdf5'

                if save_data == 'test_simple':

                    bow_features = bow_features.convert_objects()

                    file_name =  'BOW_FEATURE_' + pattern_name

                    """
                    dir_ = os.path.dirname(file_name)

                    print 'dir_: ', dir_
                    if not os.path.exists(dir_):
                        os.makedirs(dir_)
                    print 'aha dir_: ', dir_
                    """
                    with open(file_name + '.csv', 'w') as f:
                        bow_features.to_csv(f)


                    feature_list_simple = feature_list[['estimate_label', 'estimate_label_index'] ].convert_objects()
                    hdf.put(pattern_name + '_stream_simple', feature_list_simple, format='table', data_columns=True)

                    print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& bow_features: \n', bow_features


                    hdf.put(file_name, bow_features, format='table', data_columns=True)



                    

                if save_data == 'test':
                    feature_list = feature_list.convert_objects()
                    hdf.put(pattern_name + '_stream_all', feature_list, format='table', data_columns=True)

            hdf.close()
    f_gui.close() 

    return feature_list
    """        
    except StandardError, e:
        print 'sliding_window problem: \n', e
    """


def enthread(target, thread_name, args):
    """
    enthread: encapsulate a thread to capture the output when the thread terminates
    Parameters:
        target: target function for the thread
        thread_name: string type, any thread name
        args: argument lsit for the target function
    Return:
        q: output when the thread terminates
    """
    print 'enthread!0'
    q = Queue.Queue()
    def wrapper():
        q.put(target(*args))
    t = threading.Thread(name = thread_name, target=wrapper)
    t.start()
    return q


def real_time_stream(file_stream, ws, ss = None, label = -1, get_traning_data = False):
    """
    real_time_stream: main function to combine other function. Creating 2 threads, one for generating real time data stream
                        the other one to generate sliding window either for training or testing
    Parameters:
    file_stream is the input csv file, which could get real time updates from Myo
    ws: window size in time duration
    ss: step size in time duration
    label: the label corresponding to the unit patterns, default to be -1
    get_traning_data set to True when user want to collect training samples, set to False when user want to use real time sliding_window for recognition
    """

    logfile = open(file_stream,"r")
    header = logfile.readline().strip(' ').split(',')

    df_stream = pd.DataFrame(index = [header[0]], columns=header[1:11]).dropna();
    #print 'df_stream :', df_stream 

    t1 = threading.Thread(name='saving stream', target = generate_stream, args=(logfile, df_stream,)  )
    #t2 = threading.Thread(name='get_sliding_window', target = get_sliding_window, args = (df_stream, 2, get_traning_data = get_traning_data ) )
    if ss is None:
        ss = ws * 0.25
    try:
        #thread.start_new_thread( generate_stream, (sys.argv[1], df,) )

        t1.start()
        time.sleep(1)
        try:
            #get_sliding_window(df, ws, get_traning_data = False, label = -1, n_sample = 45, pattern_name = 'NA', output_format = 'hdf', save_data = 'train_append', ss = None, fs = 20):
            if get_traning_data:
                ### get training sample
                q2 = enthread(target = get_sliding_window, thread_name='get_sliding_window', args=(df_stream, ws, get_traning_data, label, 120, sys.argv[2], 'hdf', 'train'  )   )
            else:
                ### only get sliding window
                q2 = enthread(target = get_sliding_window, thread_name='get_sliding_window', args=(df_stream, ws, get_traning_data, label, 120, sys.argv[2], 'hdf', 'test_simple' ))

        except:
            print "Error: thread- saving stream "

        time.sleep(1)
 
    except:
        print "Error: unable to start thread"


def shuffle_data1(hdf_file, df_name_list):
    store = pd.HDFStore(hdf_file)

    df = store[ df_name_list[0] ].ix[:,2:]
    df['label_index'] = 1

    for i in xrange(1, len(df_name_list ) ):
        df_tmp = store[ df_name_list[i] ].ix[:,2:]
        df_tmp['label_index'] = i+1
        df = df.append(df_tmp)


    df.index = range(df.shape[0])
    #print '\n df size, before: ', df.shape

    rows =  random.sample(df.index, df.shape[0] )
    df = df.ix[rows]
    #print '$$$$$$$$df: \n', df, '\n df size: ', df.shape
    store.close()

    return df





def shuffle_data2(hdf_file, label_index):
    """
    shuffle_data: combine and shuffle (randomnize by rows) DataFrame(s) to generate traing and testing samples
    Parameters:
        hdf_file: hdf file which stores the DataFrame(s)
        label_index: label_index contains the class label(s) and label_index index (e.g. {shooting1: 1, walking1:2 }) 
                    for unit_patterns: e.g. walking, jumping, 
    Return:
        train_data_set and test_data_set extracted from shuffle_data() function  
    """
    store = pd.HDFStore(hdf_file)

    df_init = False

    for key, val in label_index.items() :
        try:
            tmp_df = store[key]
            tmp_df['label_index'] = val
            if not df_init:
                df_train = pd.DataFrame(columns = tmp_df.columns) 
                df_test = pd.DataFrame(columns = tmp_df.columns)
                df_init = True

            test_sample = int(tmp_df.shape[0] * 0.25)
            rows = random.sample(tmp_df.index, test_sample )
            df_test = df_test.append(tmp_df.ix[rows])
            df_train = df_train.append(tmp_df.drop(rows))

        except StandardError, e:
            print 'The DataFrame corresponding to ", key, " may not exists: \n', e

    train_rows =  random.sample(df_train.index, df_train.shape[0] )
    df_train = df_train.ix[train_rows]

    test_rows =  random.sample(df_test.index, df_test.shape[0] )
    df_test = df_test.ix[test_rows]
    store.close()

    return [df_train, df_test]




def kmeans_dist(arrs1, arrs2):
    """
    kmeans_dist: calculate distance measure between two arrs1
    Parameters:
        arrs1: k means cluster centers
        arrs2: testing sliding window
    Return:
        res: array or matrix that keep the distance of arrs1 and arrs2
    """

    n1, m1 = arrs1.shape
    n2, m2 = arrs2.shape
    res = np.zeros( (n1, n2) )
    #print 'n1: ', n1, ' m1: ', m1
    #print 'n2: ', n2, ' m2: ', m2

    print 'arrs2 type: ', type(arrs2)
    if type(arrs2) == pd.core.frame.DataFrame:
        arrs2 = arrs2.values
    if m1== None or m2 == None :
        print "wrong dimention"
        return
    for i in xrange(n1):
        for j in xrange(n2):
            #print 'arrs1[i, :]: ', arrs1[i, : ]
            #print 'arrs2[j, :]: ', arrs2[j, :]

            res[i, j] = np.linalg.norm( arrs1[i, :] - arrs2[j, :] )
    #print 'res: ', res
    return res



def shuffle_data(hdf_file, arg_lst):
    """
    shuffle_data: combine and shuffle (randomnize by rows) DataFrame(s) to generate traing and testing samples
    Parameters:
        hdf_file: hdf file which stores the DataFrame(s)
        arg_lst: arg_lst contains the class label(s) for unit_patterns: e.g. walking, jumping, 
    Return:
        train_data_set and test_data_set extracted from shuffle_data() function  
    """
    store = pd.HDFStore(hdf_file)
    cs = store[arg_lst[0]]
    cp = store[arg_lst[1]]
    cb = store[arg_lst[2]]
    cr = store[arg_lst[3]]
    crun = store[arg_lst[4]]

    #print 'cs: ', cs
    cs['label_index'] = label_index[arg_lst[0]]
    cp['label_index'] = label_index[arg_lst[1]]
    cb['label_index'] = label_index[arg_lst[2]]
    cr['label_index'] = label_index[arg_lst[3]]
    crun['label_index'] = label_index[arg_lst[4]]

    test_sample = int(cs.shape[0] * 0.25)

    rows = random.sample(cs.index, test_sample )

    df_test = cs.ix[rows]
    df_train = cs.drop(rows)

    cp_test_sample = int(cp.shape[0] * 0.25)
    rows = random.sample(cs.index, cp_test_sample )
    df_test = df_test.append(cp.ix[rows])
    df_train = df_train.append(cp.drop(rows))

    cb_test_sample = int(cb.shape[0] * 0.25)
    rows = random.sample(cb.index, cb_test_sample )
    df_test = df_test.append(cb.ix[rows])
    df_train = df_train.append(cb.drop(rows))

    cr_test_sample = int(cr.shape[0] * 0.25)
    rows = random.sample(cr.index, cr_test_sample )
    df_test = df_test.append(cr.ix[rows])
    df_train = df_train.append(cr.drop(rows))

    crun_test_sample = int(crun.shape[0] * 0.25)
    rows = random.sample(crun.index, crun_test_sample )
    df_test = df_test.append(crun.ix[rows])
    df_train = df_train.append(crun.drop(rows))


    train_rows =  random.sample(df_train.index, df_train.shape[0] )
    df_train = df_train.ix[train_rows]

    test_rows =  random.sample(df_test.index, df_test.shape[0] )
    df_test = df_test.ix[test_rows]
    store.close()

    return [df_train, df_test]


def get_sample_label(df):
    """
    get_sample_label: seperate DataFrame into data_set and labels
    parameter: 
        df: DataFrame
    return: [data_samples, label]

    """
    #print 'df_train : ', df_train ,'\n df_test', df_test
    samples = df
    if 'estimate_label' in samples.columns:
        samples = samples.drop('estimate_label', axis=1)
    if 'estimate_label_index' in samples.columns:
        samples = samples.drop('estimate_label_index', axis=1)
    #else:
    samples = samples.dropna()
    label = samples['label_index']
    samples = samples.drop('label_index', axis=1)
    #print 'samples:&&&&&&: ', samples 
    if 'label' in samples.columns:
        samples = samples.drop('label', axis=1)

    #print 'samples:&&&&&&: ', samples

    #print 'get_sample_label --------- label: ', label    
    #if scale:    
        #samples= preprocessing.scale(samples)

    return [samples, label]


def training_svm(df, kernel_ = 'rbf', C = 1.0, h = .02 ):
    """
    training_svm training an SVM (kernel = rbf by defult)
    parameter: 
        df: DataFrame
        kernel_ : kernel function, 'rbf' by default
        C:  SVM regularization parameter
        h: step size in the mesh
    Return:
        [PCA_model, SVM_model]

    """
    [df_train, train_label] = get_sample_label(df)
    df_train = preprocessing.scale(df_train)
    num_component = int(math.ceil(df_train.shape[1] * 0.4))
    pca_ = PCA(n_components= num_component)
    pca_.fit(df_train)
    df_train_pca =  pca_.transform(df_train)
         
    ### we create an instance of SVM and fit out data. We do not scale our
    ### data since we want to plot the support vectors
    svm_train = svm.SVC(kernel= kernel_, gamma=0.7, C=C).fit(df_train_pca, train_label )
    return [pca_, svm_train]


def training_decision_tree(df):
    [df_train, train_label] = get_sample_label(df)
    num_component = int(math.ceil(df_train.shape[1] * 0.4))
    pca_ = PCA(n_components= num_component)
    pca_.fit(df_train)
    df_train_pca =  pca_.transform(df_train)
    tree_train = tree.DecisionTreeClassifier().fit(df_train_pca, train_label )
    return [pca_, tree_train]

def training_random_forest(df, pca_comp = 0.4):
    [df_train, train_label] = get_sample_label(df)
    num_component = int(math.ceil(df_train.shape[1] * pca_comp))
    pca_ = PCA(n_components= num_component)
    pca_.fit(df_train)
    df_train_pca =  pca_.transform(df_train) 
    rf_train = RandomForestClassifier(n_estimators=100).fit(df_train_pca, train_label )
    return [pca_, rf_train]

    


def testing_accuracy(test_data, test_label, pca_train, model_train, get_accuracy = False, activity = False):
    """
    testing_accuracy: test the accuracy given the test_data corresponding to the trained pca, svm model
    Parameters:
        test_data:  testing data
        test_label: testing labels
        get_accuracy: boolen variable
    Return:
        if get_accuracy = False, return predicted labels corresponding to test_data
        if get_accuracy = True, return calculated accuracy
    """

    df_test_pca =  pca_train.transform(test_data)

    test_res = model_train.predict(df_test_pca)

    n_test_sample = len(test_res)
    #print 'test_label type: ', test_label
    if activity:
        n_error = np.count_nonzero( test_res - test_label )
    else: 
        n_error = np.count_nonzero( test_res - test_label.values )
    accuracy = float(n_test_sample- n_error ) / n_test_sample

    if get_accuracy:

        print 'accuracy: ', accuracy
    if activity:
        print 'test_res VS test_label\n', test_res, '\n', test_label
    else:    
        print 'test_res VS test_label\n', test_res, '\n', test_label.values

    #if get_accuracy:
        
    #    return accuracy
    #else:
        #print 'test_res lable',  test_res
    return test_res


if __name__ == '__main__':


    #real_time_stream(sys.argv[1], get_traning_data = True)


    #logfile = open(file_stream,"r")
    #header = logfile.readline().strip(' ').split(',')

    #argv_list = ['timestamp', 'roll_w', 'pitch_w', 'yaw_w','accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

    #df_stream = pd.read_csv(sys.argv[1], index_col = 'timestamp', usecols = argv_list ).dropna();

    #get_sliding_window(df_stream, ws =3 , get_traning_data = True, ss = 2, fs = 20)

    #testing_accuracy(df_test, test_label, pca_, svm_, True)
    

    
    #arg_lst = ['shooting1', 'walking1', 'dribbling1', 'reading1', 'running1']
    #label_index = {'running1': 5, 'reading1': 4, 'dribbling1': 3, 'walking1': 2, 'shooting1': 1}
    
    #09 label_index = {'idle_sitting': 9, 'idle_standing': 8, 'guitar_foot_on_chair': 7, 'guitar_standing': 6,'running1': 5, 'guitar_sitting': 4, 'dribbling': 3, 'walking': 2, 'shooting': 1}
    
    label_index = {'idle_sitting':4, 'dribbling': 3, 'walking': 2, 'shooting': 1}
    index_label = {v: k for k, v in label_index.items()}

    act_label_index = {'LIVE_CONCERT' : 1, 'PLAY_BASKETBALL' :2, 'GUITA_PRACTICE' : 3}
    act_index_label = {v: k for k, v in act_label_index.items()}

    


    hdf_file = '../data/unit_patterns1.h5'
    global pca_
    global model_
    global act_model_
    global act_pca_
    global cluster_model_
        
    """     
    [df_train, df_test] = shuffle_data2(hdf_file, label_index)

 

    #model_file = 'kmeans_models.pickle'

    #cluster_model_  = pickle.load( open( model_file, "rb" ) )

    activity_file = 'act_model.pickle'

    [act_pca_, act_model_]  = pickle.load( open( activity_file, "rb" ) )

    #print 'cluster_model_.cluser_centers_: ', cluster_model_.cluster_centers_
    #n1, m1 = cluster_model_.cluster_centers_.shape
    #print 'shape: ', n1, ' ', m1

    #print 'cluser_models .keys() : ', cluser_models.keys()
    #cluster_model_ = cluser_models['full'] 

    #print 'support vector machine !'
    #[pca_, model_] = training_svm(df_train, kernel_ = 'rbf', C = 1.0, h = .02 )

    [df_test, test_label] = get_sample_label(df_test)

    #test_data = preprocessing.scale(df_test)

    #testing_accuracy(test_data, test_label, pca_, model_, get_accuracy = False)


    ### Decision Tree
    #print 'decision tree!'
    #[pca1_, model1_] = training_decision_tree(df_train)
    #testing_accuracy(df_test, test_label, pca1_, model1_, get_accuracy = False)



    ### Random Forest
    print 'random forest'
    [pca_, model_] = training_random_forest(df_train)

    testing_accuracy(df_test, test_label, pca_, model_, get_accuracy = True)
    


    #with open('model_list.pickle', 'wb') as f:
    #    pickle.dump([pca1_, model1_, pca2_, model2_], f)
    
    #hdf = pd.HDFStore('../data/unit_patterns1.h5')

    #test_reading = hdf['reading1_stream_test']
    #test_shooting = hdf['shooting1_stream_test']
    #hdf.close()

    """ 

    label_ = -1
    if len(sys.argv) > 3:
        label_ =  sys.argv[3]


    """
     
    ### Train Activity:
    activity_hdf_file = '../data/unit_patterns1.h5'

    df_train = shuffle_data1(activity_hdf_file, ['BOW_FEATURE_live_concert_1', 'BOW_FEATURE_basketball_1', 'BOW_FEATURE_guitar_practicing_1' ])
    df_test = shuffle_data1(activity_hdf_file,  ['BOW_FEATURE_live_concert_2', 'BOW_FEATURE_basketball_2', 'BOW_FEATURE_guitar_practicing_2' ])

    [train_data, train_label] = get_sample_label(df_train)

    [test_data, test_label] = get_sample_label(df_test)

    [act_pca_, act_model_] = training_random_forest(df_train)

    print '!!!!!test data shape: ', test_data.shape

    testing_accuracy(test_data, test_label, act_pca_, act_model_, get_accuracy = True)


    with open('act_model.pickle', 'wb') as f:
        pickle.dump([act_pca_, act_model_], f)
    time.sleep(0.1)    

    """
    
    
    #real_time_stream(sys.argv[1], 2, label = label_,  get_traning_data = False)
    real_time_stream(sys.argv[1], 2, label = label_, get_traning_data = True)



    






    
