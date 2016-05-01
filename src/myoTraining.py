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
from sknn.mlp import Classifier, Layer
#from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from hmmlearn import hmm
import math
import pickle
import os




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
   
def generate_window_feature(hdf_file, ws, ss = None, fs = 200, pattern_name = 'NA'):
    hdf = pd.HDFStore('../data/gesture.h5')
    hdf_keys = hdf.keys() 
    stat_feature = ['_mean', '_median', '_var', '_meanCrossCount']
    if ss is None:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws * 0.25
    ss = int(ss * fs)
    ws = int(ws * fs)

    print 'hdf_keys: ', hdf_keys
    for key in hdf_keys:
        if 'raw_' in key:
            #key = key.strip('/ ')
            df = hdf[key]    
            df_header = df.columns
            header = []
            for elm in stat_feature:
                for df_elm in df_header:
                    header.append(df_elm + elm)
            feature_df = pd.DataFrame(columns= header)
            feature_df.loc[0] = 0
            feature_df_count = 0
            print '+++++++++++++++++++start sliding windows+++++++++++++++++'
            df_len = df.shape[0]
            start = 0
            end = ws
            while end < df_len:
                #print 'feature_df.ix[feature_df_count]: ', feature_df.ix[feature_df_count]
                feature_df.ix[feature_df_count] = np.concatenate((df.ix[start:end].mean(0), df.ix[start:end].median(0), df.ix[start:end].var(0),  count_mean_crossing(df.ix[start:end]) ))
                feature_df_count +=1

                start += ss
                end += ss
            feature_df = feature_df.convert_objects()    
            hdf.put(key[5:], feature_df, format='table', data_columns=True)
    hdf.close()        

def shuffle_data(hdf_file, label_index):
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

    print 'store.keys: ', store.keys()
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

def get_sample_label(df):
    """
    get_sample_label: seperate DataFrame into data_set and labels
    parameter: 
        df: DataFrame
    return: [data_samples, label]
    """
    samples = df
    samples = samples.dropna()
    label = samples['label_index']
    samples = samples.drop('label_index', axis=1)
    #print 'samples:&&&&&&: ', samples 
    if 'label' in samples.columns:
        samples = samples.drop('label', axis=1)

    return [samples, label]


def testing_accuracy(test_data, test_label, pca_train, model_train, extract_test_res = False, get_accuracy = False, activity = False):
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
    #print test_data
    df_test_pca =  pca_train.transform(test_data)
    #print 'df_test_pca.shape: ', df_test_pca.shape
    test_res = model_train.predict(df_test_pca)
    n_test_sample = len(test_res)
    if extract_test_res:
        tmp = []
        for i in range(n_test_sample):
            tmp.append(int(test_res[i][0]) )
        test_res = tmp
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

    return test_res

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

def training_NN(df, alpha_val, hidden_layer ):
    [df_train, train_label] = get_sample_label(df)
    num_component = int(math.ceil(df_train.shape[1] * 0.4))
    pca_ = PCA(n_components= num_component)
    pca_.fit(df_train)
    df_train_pca =  pca_.transform(df_train)
    #nn_train = Classifier(layers=[Layer("Maxout", units=80), Layer("Softmax")], learning_rate = alpha_val, n_iter=80).fit(df_train_pca, train_label)
    nn_train = Classifier(
    layers=[
        Layer("Maxout", units= hidden_layer[0] ),
        Layer("Softmax")],
    learning_rate=alpha_val,
    n_iter=25).fit(df_train_pca, train_label)
    #nn_train = neural_network.MLPClassifier(algorithm='l-bfgs', alpha = alpha_val, hidden_layer_sizes = hidden_layer, random_state=1).fit(df_train_pca, train_label)
    return [pca_, nn_train]


def training_random_forest(df, pca_comp = 0.4):
    [df_train, train_label] = get_sample_label(df)
    num_component = int(math.ceil(df_train.shape[1] * pca_comp))
    pca_ = PCA(n_components= num_component)
    pca_.fit(df_train)
    df_train_pca =  pca_.transform(df_train) 
    rf_train = RandomForestClassifier(n_estimators=100).fit(df_train_pca, train_label )

    return [pca_, rf_train]

def training_hmm(df, pca_comp = 0.6):
    [df_train, train_label] = get_sample_label(df)
    num_component = int(math.ceil(df_train.shape[1] * pca_comp))
    pca_ = PCA(n_components= num_component)
    pca_.fit(df_train)
    df_train_pca =  pca_.transform(df_train)
    remodel = hmm.GaussianHMM(n_components=6, n_iter=200).fit(df_train_pca)
    print 'remodel.monitor_ :', remodel.monitor_ 
    print 'remodel.monitor_.converged :', remodel.monitor_.converged
    print 'initial prob :, ', remodel.startprob_
    print 'transition matrix: ', remodel.transmat_
    return [pca_, remodel]


def run_random_forest(hdf_file, label_index):
    print 'training random forest'
    [df_train, df_test] = shuffle_data(hdf_file, label_index)
    [train_data, train_label] = get_sample_label(df_train)

    print 'train_data shape: ', train_data.shape
    [test_data, test_label] = get_sample_label(df_test)
    print 'test_data shape: ', test_data.shape
    [pca_, model_] = training_random_forest(df_train)
    with open('../data/random_forest.p', 'wb') as f:
        pickle.dump([pca_, model_], f)
    testing_accuracy(test_data, test_label, pca_, model_, get_accuracy = True)

def run_hmm(hdf_file, label_index):
    print 'training random forest'
    [df_train, df_test] = shuffle_data(hdf_file, label_index)
    [train_data, train_label] = get_sample_label(df_train)

    print 'train_data shape: ', train_data.shape
    [test_data, test_label] = get_sample_label(df_test)
    print 'test_data shape: ', test_data.shape
    [pca_, model_] = training_hmm(df_train)
    testing_accuracy(test_data, test_label, pca_, model_, get_accuracy = True)


def run_svm(hdf_file, label_index):
    print 'training svm'
    [df_train, df_test] = shuffle_data(hdf_file, label_index)
    [train_data, train_label] = get_sample_label(df_train)
    print 'train_data shape: ', train_data.shape
    [test_data, test_label] = get_sample_label(df_test)
    print 'test_data shape: ', test_data.shape
    [pca_, model_] = training_svm(df_train)
    testing_accuracy(test_data, test_label, pca_, model_, get_accuracy = True)

def run_decision_tree(hdf_file, label_index):
    print 'training decision tree'
    [df_train, df_test] = shuffle_data(hdf_file, label_index)
    [train_data, train_label] = get_sample_label(df_train)
    print 'train_data shape: ', train_data.shape
    [test_data, test_label] = get_sample_label(df_test)
    print 'test_data shape: ', test_data.shape
    [pca_, model_] = training_decision_tree(df_train)
    testing_accuracy(test_data, test_label, pca_, model_, get_accuracy = True)

def run_NN(hdf_file, label_index, alpha_val, hidden_layer):
    print 'training neural network'
    [df_train, df_test] = shuffle_data(hdf_file, label_index)
    [train_data, train_label] = get_sample_label(df_train)
    print 'train_data shape: ', train_data.shape
    [test_data, test_label] = get_sample_label(df_test)
    print 'test_data shape: ', test_data.shape
    [pca_, model_] = training_NN(df_train, alpha_val, hidden_layer)
    testing_accuracy(test_data, test_label, pca_, model_, extract_test_res = True, get_accuracy = True)


if __name__ == "__main__":

    label_index_ = {'emg_index_test': 1, 'emg_middle_test':2, 'emg_ring_text' :3, 'emg_little_test': 4, 'emg_spread_test':5, 'emg_idle_test':6}  # label_index
    hdf_file_ = '../data/gesture.h5'

    #file  = sys.argv[1] 
    #df = pickle.load(open(file, "rb"))
    #ws = 1
    #generate_window_feature(hdf_file_, ws)


    ## shuffle data and seperate to 
    #run_random_forest(hdf_file_, label_index_)
    run_random_forest(hdf_file_, label_index_)
    #run_hmm(hdf_file_, label_index_)
    #run_svm(hdf_file_, label_index_)
    #run_decision_tree(hdf_file_, label_index_)
    #run_NN(hdf_file_, label_index_, alpha_val = 1e-3, hidden_layer = (60, 5) )
    

