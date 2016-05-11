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
    # global count_emg
    # global window
    # global emgdf
    # global sampleLen 
    # count_emg += 1

    # if window  > int(sys.argv[2]) :
    #    sys.exit()

    # txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    # vtime = datetime.datetime.now()
    # #stime = vtime.strftime("%Y-%m-%d %H:%M:%S.%f")
    # stime = vtime.strftime("%H:%M:%S.%f")  
    # txt += stime
    # txt += str(data)
    # emgdf.ix[stime] = data
    # # countdown visual signal, every 0.1s
    # if count_emg % 20 == 0:
    #     print("".join(["#" for i in range(10 - int((count_emg-(window-1)*sampleLen)/20))]))

    # if count_emg % sampleLen == 0:
    #     print "\n\n\n########################EMG window number:  ", window, " count_emg = ",count_emg
    #     window +=1

    # if count_emg == int(sampleLen*int(sys.argv[2])):
    #     #pickle.dump(emgdf, open( sys.argv[1], "wb" ) )
    #     hdf = pd.HDFStore('../data/gesture.h5')
    #     emgdf = emgdf.convert_objects()
    #     hdf.put('raw_' + sys.argv[1], emgdf, format='table', data_columns=True)
    #     print 'hdf keys: ', hdf.keys()
    #     hdf.close()
    #     sys.exit()

    # print(txt)
    return

def IMUHandler(addr, tags, data, client_address):
    global count_imu
    global window_imu
    global imudf
    global sampleLen 
    count_imu += 1

    request_window = int(sys.argv[2])
    if window_imu  > request_window:
        sys.exit()

    #txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    vtime = datetime.datetime.now()
    #stime = vtime.strftime("%Y-%m-%d %H:%M:%S.%f")
    stime = vtime.strftime("%H:%M:%S.%f")  
    #txt += stime
    #txt += str(data)
    imudf.ix[stime] = data
    # countdown visual signal, every 0.1s
    if count_imu % 20 == 0:
        print("".join(["#" for i in range(10 - int((count_imu-(window_imu-1)*sampleLen)/20))]))

    if count_imu % sampleLen == 0:
        print "\n\n\n#####################IMU window number:  ", window_imu, " count_imu = ",count_imu
        window_imu +=1

    if count_imu == int(sampleLen*request_window):
        pickle.dump(imudf, open( sys.argv[1], "wb" ) )
        hdf = pd.HDFStore('../data/gesture.h5')
        imudf = imudf.convert_objects()
        hdf.put('raw_imu' + sys.argv[1], imudf, format='table', data_columns=True)
        print 'hdf keys: ', hdf.keys()
        hdf.close()
        sys.exit()

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
    s.addMsgHandler('/myo/IMG', IMUHandler) 
    s.addMsgHandler('/myo/emg', EMGHandler)     # call handler() for OSC messages received with the /startup address
    s.addMsgHandler('/myo/pose', handler)     # call handler() for OSC messages received with the /startup address

    s.serve_forever()

if __name__ == "__main__":
    count_emg = 0
    count_imu = 0
    window = 1
    window_imu = 1
    sampleLen = 200 # 1s for each sample
    imu_header = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'roll', 'pitch', 'yaw', 'quat_x', 'quat_y', 'quat_z', 'quat_w' ] 
    emg_header = ['em1', 'em2', 'em3', 'em4', 'em5', 'em6', 'em7', 'em8'] 
    emgdf = pd.DataFrame(columns= emg_header)
    imudf = pd.DataFrame(columns= imu_header )
    #global count_img
    #t_emg = threading.Thread(name='ploting_emg', target= plot_emg_1)
    #t_emg.start()

    #t_img = threading.Thread(name='ploting_img', target= plot_img)
    #t_img.start()
    print 'filename: ', sys.argv[1],' sample count: ', sys.argv[2]
    #offline_plot(sys.argv[1])
    get_stream()
