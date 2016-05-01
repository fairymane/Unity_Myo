# (*) To communicate with Plotly's server, sign in with credentials file
import plotly.plotly as py  
 
# (*) Useful Python/Plotly tools
import plotly.tools as tls   
 
# (*) Graph objects to piece together plots
from plotly.graph_objs import *
 
import numpy as np  # (*) numpy for math functions and arrays
import pandas as pd
import matplotlib.pyplot as plt
import datetime 
import time
import sys 
import OSC
import threading
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
    count_emg = count_emg + 1
    if count_emg  >5000 :
        sys.exit()

    txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    vtime = datetime.datetime.now()
    #stime = vtime.strftime("%Y-%m-%d %H:%M:%S.%f")
    stime = vtime.strftime("%H:%M:%S.%f")  
    txt += stime
    txt += str(data)
    emgdf.ix[stime] = data
    if count_emg % 200 == 0:
        print "\n\n\n ###########################################window number:  ", window
        window +=1
    if count_emg == 5000:
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
def plot_img():
    global count_img
    tls.set_credentials_file(username="fairymane", api_key="x4kt8y1smu")

    stream_ids = tls.get_credentials_file()['stream_ids']

    #print 'pring IMG!!!!'

    streams = []
    traces = [] 
    for i in xrange(9):
        streams.append(Stream( token=stream_ids[i],  maxpoints=20))
        traces.append(Scatter(
        x=[],
        y=[],
        name = img_header[i],
        showlegend=True,
        visible=True,
        yaxis='y2',
        mode='lines+markers',
        stream=streams[i]        # (!) embed stream id, 1 per trace
    ) ) 
    data = Data(traces)

    layout = Layout(
        title='Time Series over 9 IMU streams',
        showlegend=True,
        autosize=True,
        width=1400,
        height=1000,
        xaxis=XAxis(
            title='x Axis - timestamp',
            titlefont=Font(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            ),
            autorange=True,
        ),
        yaxis=YAxis(
            title='accel_x_y_z',
            titlefont=Font(
                family='Courier New, monospace',
                size=14,
                color='#7f7f7f'
            ),
            domain=[0, 0.32],
        ),

        yaxis2=YAxis(
            title='gyro_x_y_z',
            titlefont=Font(
                family='Courier New, monospace',
                size=14,
                color='#7f7f7f'
            ),
            domain=[0.33, 0.66],
            autorange=True,
            showgrid=True,
        ),

            yaxis3=YAxis(
            title='roll_pitch_yaw',
            titlefont=Font(
                family='Courier New, monospace',
                size=12,
                color='#7f7f7f'
            ),
            domain=[0.68, 1],
            autorange=True,
            showgrid=True,
        )
    )

    fig = Figure(data=data, layout=layout ) 
    #plot_url = py.plot(fig, filename='tools-get-subplots')

    # (@) Send fig to Plotly, initialize streaming plot, open new tab
    unique_url = py.plot(fig, filename='legend-labels')

    s = []
    for i in xrange(9):
        s.append(py.Stream(stream_ids[i]) )
        s[i].open()
    jstream = 0

    while jstream < count_img:
        #print 'count_img: ', count_img, ' --jstream: ', jstream,
        #print 'imgdf.ix[jstream]: ', imgdf.ix[jstream]
        for i in xrange(9):
            s[i].write(dict(x= imgdf.index[jstream]  , y=imgdf.ix[jstream][i]))
        #time.sleep(0.08)
        jstream +=1

    for i in xrange(9):
        s[i].close()


    """    
    s1 = py.Stream(stream_id1)
    s2 = py.Stream(stream_id2)
    s3 = py.Stream(stream_id3)
    s4 = py.Stream(stream_id4)
    s5 = py.Stream(stream_id5)
    s6 = py.Stream(stream_id6)
    s7 = py.Stream(stream_id7)
    s8 = py.Stream(stream_id8)
    s9 = py.Stream(stream_id9)
    # (@) Open the stream
    s1.open()
    s2.open()
    s3.open()
    s4.open()
    s5.open()
    s6.open()
    s7.open()
    s8.open()
    s9.open()
    i = 0
    N = len(yy1);

    while i<N:
        x = xx[i];
        y1 = yy1[i];
        y2= yy2[i];
        y3= yy3[i];

        gx_ =  gx[i];
        gy_ =  gy[i];
        gz_ =  gz[i];

        r = roll[i]
        p = pitch[i]
        y = yaw[i]

        s1.write(dict(x=x, y=y1))
        s2.write(dict(x=x, y=y2))
        s3.write(dict(x=x, y=y3))  

        s4.write(dict(x=x, y=gx_))
        s5.write(dict(x=x, y
        =gy_))
        s6.write(dict(x=x, y=gz_))

        s7.write(dict(x=x, y=r))
        s8.write(dict(x=x, y=p))
        s9.write(dict(x=x, y=y)) 

        time.sleep(0.08)
        i +=1

    s1.close() 
    s2.close() 
    s3.close() 
    s4.close()
    s5.close() 
    s6.close()
    s7.close()
    s8.close() 
    s9.close()   
    """
def offline_plot(pickle_file):
    df = pickle.load(open(pickle_file, "rb"))
    plot_emg_1(df)

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
    img_header = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'roll', 'pitch', 'yaw', 'quat_x', 'quat_y', 'quat_z', 'quat_w' ] 
    emg_header = ['em1', 'em2', 'em3', 'em4', 'em5', 'em6', 'em7', 'em8'] 
    emgdf = pd.DataFrame(columns= emg_header)
    imgdf = pd.DataFrame(columns= img_header )
    #global count_img
    #t_emg = threading.Thread(name='ploting_emg', target= plot_emg_1)
    #t_emg.start()

    #t_img = threading.Thread(name='ploting_img', target= plot_img)
    #t_img.start()
    print sys.argv[1]
    #offline_plot(sys.argv[1])
    get_stream()
