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


count_emg = 0
count_img = 0
img_header = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'roll', 'pitch', 'yaw', 'quat_x', 'quat_y', 'quat_z', 'quat_w' ] 
emg_header = ['em1', 'em2', 'em3', 'em4', 'em15', 'em6', 'em7', 'em8'] 
emgdf = pd.DataFrame(columns= emg_header)
imgdf = pd.DataFrame(columns= img_header )



def plot_emg_1():
    #tls.set_credentials_file(username="fairymane", api_key="x4kt8y1smu")
    global count_emg

    stream_ids = tls.get_credentials_file()['stream_ids']

    
    stream_id1 = stream_ids[0]
    stream_id2 = stream_ids[1]
    stream_id3 = stream_ids[2]

    stream_id4 = stream_ids[3]
    stream_id5 = stream_ids[4]
    stream_id6 = stream_ids[5]

    stream_id7 = stream_ids[6]
    stream_id8 = stream_ids[7]

    # Make instance of stream id object 
    #for accelometer
    stream1 = Stream( token=stream_id1,  maxpoints=20)

    stream2 = Stream(
        token=stream_id2,  # (!) link stream id to 'token' key
        maxpoints=40      # (!) keep a max of 80 pts on screen 
    )
    stream3 = Stream(
        token=stream_id3,  # (!) link stream id to 'token' key
        maxpoints=40      # (!) keep a max of 80 pts on screen
    )
    stream4 = Stream(
        token=stream_id4,  # (!) link stream id to 'token' key
        maxpoints=40      # (!) keep a max of 80 pts on screen
    )
    stream5 = Stream(
        token=stream_id5,  # (!) link stream id to 'token' key
        maxpoints=40      # (!) keep a max of 80 pts on screen
    )
    stream6 = Stream(
        token=stream_id6,  # (!) link stream id to 'token' key
        maxpoints=40      # (!) keep a max of 80 pts on screen
    )
    stream7 = Stream(
        token=stream_id7,  # (!) link stream id to 'token' key
        maxpoints=40      # (!) keep a max of 80 pts on screen
    )
    stream8 = Stream(
        token=stream_id8,  # (!) link stream id to 'token' key
        maxpoints=40      # (!) keep a max of 80 pts on screen
    )
    trace1 = Scatter(
        x=[],
        y=[],
        name = 'emg1',
        showlegend=True,
        visible=True,
        yaxis='y2',
        mode='lines+markers',
        stream=stream1         # (!) embed stream id, 1 per trace
    )
    trace2 = Scatter(
        x=[],
        y=[],
        name = 'emg2',
        showlegend=True,
        visible=True,
        yaxis='y2',
        mode='lines+markers',
        stream=stream2         # (!) embed stream id, 1 per trace
    )

    trace3 = Scatter(
        x=[],
        y=[],
        name = 'emg3',
        showlegend=True,
        visible=True,
        yaxis='y2',
        mode='lines+markers',
        stream=stream3         # (!) embed stream id, 1 per trace
    )

    ## trace4-6 for gyro
    trace4 = Scatter(
        x=[],
        y=[],
        name = 'emg4',
        showlegend=True,
        visible=True,
        yaxis='y2',
        mode='lines+markers',
        stream=stream4         # (!) embed stream id, 1 per trace
    )

    trace5 = Scatter(
        x=[],
        y=[],
        name = 'emg5',
        showlegend=True,
        visible=True,
        mode='lines+markers',
        stream=stream5         # (!) embed stream id, 1 per trace
    )

    trace6 = Scatter(
        x=[],
        y=[],
        name = 'emg6',
        showlegend=True,
        visible=True,
        mode='lines+markers',
        stream=stream6         # (!) embed stream id, 1 per trace
    )


    trace7 = Scatter(
        x=[],
        y=[],
        name = 'emg7',
        showlegend=True,
        visible=True,
        mode='lines+markers',
        stream=stream7         # (!) embed stream id, 1 per trace
    )

    trace8 = Scatter(
        x=[],
        y=[],
        name = 'emg8',
        showlegend=True,
        visible=True,
        mode='lines+markers',
        stream=stream8         # (!) embed stream id, 1 per trace
    )
    """

    streams = []
    traces = [] 
    for i in xrange(8):
        streams.append(Stream( token=stream_ids[i],  maxpoints=20))
        traces.append(Scatter(
        x=[],
        y=[],
        name = emg_header[i],
        showlegend=True,
        visible=True,
        yaxis='y2',
        mode='lines+markers',
        stream=streams[i]        # (!) embed stream id, 1 per trace
    ) ) 
    """
    #data = Data([traces[0], traces[1], traces[2], traces[3], traces[4], traces[5], traces[6], traces[7]])
    data = Data([trace1, trace2, trace3, trace4,trace5, trace6, trace7,trace8])

    # Make a figure object
    #fig = tls.make_subplots(rows=1, cols=1)
    layout = Layout(
        title='Time Series over 8 EMG streams',
        
        autosize=True,
        width=1200,
        height=1200,
        showlegend=True,
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
            title='emgs 5-8',
            titlefont=Font(
                family='Courier New, monospace',
                size=14,
                color='#7f7f7f'
            ),    
            autorange=True,
            domain=[0, 0.5],
        ),

        yaxis2=YAxis(
            title='emg 1-4',
            titlefont=Font(
                family='Courier New, monospace',
                size=14,
                color='#7f7f7f'
            ),
            domain=[0.5, 1],
            autorange=True,
            showgrid=True,
        )

    )

    fig = Figure(data=data, layout=layout ) 
    #plot_url = py.plot(fig, filename='tools-get-subplots')

    # (@) Send fig to Plotly, initialize streaming plot, open new tab
    unique_url = py.plot(fig, filename='legend-labels')
    """
    s1 = py.Stream(stream_ids[0])
    s2 = py.Stream(stream_ids[1])
    s3 = py.Stream(stream_ids[2])
    s4 = py.Stream(stream_ids[3])
    s5 = py.Stream(stream_ids[4])
    s6 = py.Stream(stream_ids[5])
    s7 = py.Stream(stream_ids[6])
    s8 = py.Stream(stream_ids[7])

    """
    
    s1 = py.Stream(stream_id1)
    s2 = py.Stream(stream_id2)
    s3 = py.Stream(stream_id3)
    s4 = py.Stream(stream_id4)
    s5 = py.Stream(stream_id5)
    s6 = py.Stream(stream_id6)
    s7 = py.Stream(stream_id7)
    s8 = py.Stream(stream_id8)
    

    # (@) Open the stream
    s1.open()
    s2.open()
    s3.open()
    s4.open()
    s5.open()
    s6.open()
    s7.open()
    s8.open()

    jstream  = 0

    while True:
        if jstream < len(emgdf) and jstream%3 == 0:
            print 'emgdf lenth: ', emgdf.shape[0], ' --jstream: ', jstream
            print 'time: ', emgdf.index[jstream]
            #print 'emgdf.ix[i]: ', emgdf.ix[i]
            x  = emgdf.index[jstream]
            e1 = emgdf.ix[jstream][0]
            e2 = emgdf.ix[jstream][1]
            e3 = emgdf.ix[jstream][2]
            e4 = emgdf.ix[jstream][3]
            e5 = emgdf.ix[jstream][4]
            e6 = emgdf.ix[jstream][5]
            e7 = emgdf.ix[jstream][6]
            e8 = emgdf.ix[jstream][7]

            s1.write(dict(x=x, y=e1))
            s2.write(dict(x=x, y=e2))
            s3.write(dict(x=x, y=e3))
            s4.write(dict(x=x, y=e4))
            s5.write(dict(x=x, y=e5))
            s6.write(dict(x=x, y=e6))
            s7.write(dict(x=x, y=e7))
            s8.write(dict(x=x, y=e8))

            #time.sleep(0.08)
            jstream +=1

    s1.close() 
    s2.close() 
    s3.close() 
    s4.close()
    s5.close() 
    s6.close()
    s7.close()
    s8.close() 


def plot_emg():
    global count_emg
    tls.set_credentials_file(username="fairymane", api_key="x4kt8y1smu")
    stream_ids = tls.get_credentials_file()['stream_ids']
    streams = []
    traces = [] 
    for i in xrange(8):
        streams.append(Stream( token=stream_ids[i],  maxpoints=20))
        traces.append(Scatter(
        x=[],
        y=[],
        name = emg_header[i],
        showlegend=True,
        visible=True,
        yaxis='y2',
        mode='lines+markers',
        stream=streams[i]        # (!) embed stream id, 1 per trace
    ) ) 
    data = Data(traces)

    # Make a figure object
    #fig = tls.make_subplots(rows=1, cols=1)
    layout = Layout(
        title='Time Series over 8 EMG streams',
        
        autosize=True,
        width=1200,
        height=1200,
        showlegend=True,
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
            title='emgs 5-8',
            titlefont=Font(
                family='Courier New, monospace',
                size=14,
                color='#7f7f7f'
            ),    
            autorange=True,
            domain=[0, 0.5],
        ),

        yaxis2=YAxis(
            title='emg 1-4',
            titlefont=Font(
                family='Courier New, monospace',
                size=14,
                color='#7f7f7f'
            ),
            domain=[0.5, 1],
            autorange=True,
            showgrid=True,
        )
    )
    fig = Figure(data=data, layout=layout ) 

    # (@) Send fig to Plotly, initialize streaming plot, open new tab
    unique_url = py.plot(fig, filename='legend-labels')
    s = []
    for i in xrange(8):
        s.append(py.Stream(stream_ids[i]) )
        s[i].open()
    jstream = 0

    while jstream < count_emg and jstream % 3 == 0:
        if jstream < len(emgdf.index):
            print 'table lenth: ', emgdf.shape[0] , ' --jstream: ', jstream,
            print 'emgdf.ix[jstream]: ', emgdf.ix[jstream]
            for i in xrange(8):
                s[i].write(dict(x= emgdf.index[jstream]  , y=emgdf.ix[jstream][i]))
            #time.sleep(0.08)
            jstream +=1

    for i in xrange(8):
        s[i].close()

def handler(addr, tags, data, client_address):
    txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    vtime = datetime.datetime.now()
    #stime = vtime.strftime("%Y-%m-%d %H:%M:%S ") 
    stime = vtime.strftime("%H:%M:%S.%f") 
    txt += stime
    txt += str(data)
    print time.time()
    print(txt)

def EMGHandler(addr, tags, data, client_address):
    global count_emg
    count_emg = count_emg + 1
    #if count_emg  >1000 :
    #    sys.exit()

    txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    vtime = datetime.datetime.now()
    #stime = vtime.strftime("%Y-%m-%d %H:%M:%S.%f")
    stime = vtime.strftime("%H:%M:%S.%f")  
    txt += stime
    txt += str(data)
    emgdf.ix[stime] = data;
    #print(txt)

def IMGHandler(addr, tags, data, client_address):
    global count_img
    count_img += 1
    if count_img > 300:
        sys.exit()
    txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    vtime = datetime.datetime.now()
    stime = vtime.strftime("%H:%M:%S.%f") 
    txt += stime
    txt += str(data)
    imgdf.ix[stime] = data;
    #print(txt)

def plot_img():
    global count_img
    tls.set_credentials_file(username="fairymane", api_key="x4kt8y1smu")

    stream_ids = tls.get_credentials_file()['stream_ids']

    print 'pring IMG!!!!'

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
        print 'count_img: ', count_img, ' --jstream: ', jstream,
        print 'imgdf.ix[jstream]: ', imgdf.ix[jstream]
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
        s5.write(dict(x=x, y=gy_))
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
if __name__ == "__main__":
    count_emg = 0
    count_img = 0
    global count_emg
    global count_img
    t_emg = threading.Thread(name='ploting_emg', target= plot_emg_1)
    t_emg.start()

    #t_img = threading.Thread(name='ploting_img', target= plot_img)
    #t_img.start()

    s = OSC.OSCServer(('127.0.0.1', 8888))  # listen on localhost, port 57120


    s.addMsgHandler('/myo/IMG', IMGHandler) 
    s.addMsgHandler('/myo/emg', EMGHandler)     # call handler() for OSC messages received with the /startup address
    s.addMsgHandler('/myo/pose', handler)     # call handler() for OSC messages received with the /startup address

    s.serve_forever()
    if count_emg  > 1000:
        print 'imgdc  &&&&&&&&& shape: ', imgdf.shape
        print imgdf
        s.close()
        sys.exit()
