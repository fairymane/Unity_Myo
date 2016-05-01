# (*) To communicate with Plotly's server, sign in with credentials file
# import plotly.plotly as py  
 
# # (*) Useful Python/Plotly tools
# import plotly.tools as tls   
 
# # (*) Graph objects to piece together plots
# from plotly.graph_objs import *
 
import numpy as np  # (*) numpy for math functions and arrays
import pandas as pd
import matplotlib.pyplot as plt
import datetime 
import time
import sys 
import OSC
# import threading
import pickle

def handler(addr, tags, data, client_address):
    txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    vtime = datetime.datetime.now()
    #stime = vtime.strftime("%Y-%m-%d %H:%M:%S ") 
    stime = vtime.strftime("%H:%M:%S.%f") 
    txt += stime
    txt += str(data)
    #print time.time()
    #print(txt)

def EMGHandler(addr, tags, data, client_address):
    global count_emg
    global window
    global emgdf
    global sampleLen 
    count_emg += 1

    # if count_emg  > sampleLen * sys.argv[2] :
    #     sys.exit()

    txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    vtime = datetime.datetime.now()
    #stime = vtime.strftime("%Y-%m-%d %H:%M:%S.%f")
    stime = vtime.strftime("%H:%M:%S.%f")  
    txt += stime
    txt += str(data)
    emgdf.ix[stime] = data
    # countdown visual signal, every 0.1s
    if count_emg % 20 == 0:
        print("".join(["#" for i in range(10 - int((count_emg-(window-1)*sampleLen)/20))]))

    if count_emg % sampleLen == 0:
        print "\n\n\n######################## window number:  ", window, " count_emg = ",count_emg
        window +=1

    if count_emg == int(sampleLen*int(sys.argv[2])):
        #pickle.dump(emgdf, open( sys.argv[1], "wb" ) )
        hdf = pd.HDFStore('../data/gesture.h5')
        emgdf = emgdf.convert_objects()
        hdf.put('raw_' + sys.argv[1], emgdf, format='table', data_columns=True)
        print 'hdf keys: ', hdf.keys()
        hdf.close()
        sys.exit()

    #print(txt)

def IMGHandler(addr, tags, data, client_address):
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

def get_stream():
    s = OSC.OSCServer(('127.0.0.1', 8888))  # listen on localhost, port 57120
    s.addMsgHandler('/myo/IMG', IMGHandler) 
    s.addMsgHandler('/myo/emg', EMGHandler)     # call handler() for OSC messages received with the /startup address
    s.addMsgHandler('/myo/pose', handler)     # call handler() for OSC messages received with the /startup address

    s.serve_forever()

if __name__ == "__main__":
    count_emg = 0
    count_img = 0
    window = 1
    sampleLen = 200 # 1s for each sample
    img_header = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'roll', 'pitch', 'yaw', 'quat_x', 'quat_y', 'quat_z', 'quat_w' ] 
    emg_header = ['em1', 'em2', 'em3', 'em4', 'em5', 'em6', 'em7', 'em8'] 
    emgdf = pd.DataFrame(columns= emg_header)
    imgdf = pd.DataFrame(columns= img_header )
    #global count_img
    #t_emg = threading.Thread(name='ploting_emg', target= plot_emg_1)
    #t_emg.start()

    #t_img = threading.Thread(name='ploting_img', target= plot_img)
    #t_img.start()
    print 'filename: ', sys.argv[1],' sample count: ', sys.argv[2]
    #offline_plot(sys.argv[1])
    get_stream()
