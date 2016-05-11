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
import pickle

tls.set_credentials_file(username="fairymane", api_key="x4kt8y1smu")



stream_ids = tls.get_credentials_file()['stream_ids']

stream_id1 = stream_ids[0]
stream_id2 = stream_ids[1]
stream_id3 = stream_ids[2]

stream_id4 = stream_ids[3]
stream_id5 = stream_ids[4]
stream_id6 = stream_ids[5]

stream_id7 = stream_ids[6]
stream_id8 = stream_ids[7]
stream_id9 = stream_ids[8]

# Make instance of stream id object 
#for accelometer
stream1 = Stream( token=stream_id1,  maxpoints=20)

stream2 = Stream(
    token=stream_id2,  # (!) link stream id to 'token' key
    maxpoints=20      # (!) keep a max of 80 pts on screen
    
)

stream3 = Stream(
    token=stream_id3,  # (!) link stream id to 'token' key
    maxpoints=20      # (!) keep a max of 80 pts on screen
)


## for gyro scope
stream4 = Stream(
    token=stream_id4,  # (!) link stream id to 'token' key
    maxpoints=20      # (!) keep a max of 80 pts on screen
)

stream5 = Stream(
    token=stream_id5,  # (!) link stream id to 'token' key
    maxpoints=20      # (!) keep a max of 80 pts on screen
)

stream6 = Stream(
    token=stream_id6,  # (!) link stream id to 'token' key
    maxpoints=20      # (!) keep a max of 80 pts on screen
)

## for row pitch yaw
stream7 = Stream(
    token=stream_id7,  # (!) link stream id to 'token' key
    maxpoints=20      # (!) keep a max of 80 pts on screen
)

stream8 = Stream(
    token=stream_id8,  # (!) link stream id to 'token' key
    maxpoints=20      # (!) keep a max of 80 pts on screen
)

stream9 = Stream(
    token=stream_id9,  # (!) link stream id to 'token' key
    maxpoints=20      # (!) keep a max of 80 pts on screen
)


file_ = 'raw_imu_' + sys.argv[1]
#df = pd.read_csv(file, index_col = 'formatted_time').dropna()
print 'file_: ', file_
hdf = pd.HDFStore('../data/gesture.h5')
print 'hdf_keys: ', hdf.keys()

df = hdf[file_] 
#df = pickle.load(open(file_, "rb"))

print df.columns
#xx = df['timestamp'].values;
xx = df.index.values;
yy1 = df['accel_x'].values
yy2 = df['accel_y'].values
yy3 = df['accel_z'].values

gx = df['gyro_x'].values
gy = df['gyro_y'].values
gz = df['gyro_z'].values

roll = df['roll'].values
pitch = df['pitch'].values
yaw = df['yaw'].values


## trace1-3 for accel
trace1 = Scatter(
    x=[],
    y=[],
    name = 'accel_x',
    showlegend=True,
    visible=True,
    mode='lines+markers',
    stream=stream1         # (!) embed stream id, 1 per trace
)

trace2 = Scatter(
    x=[],
    y=[],
    name = 'accel_y',
    showlegend=True,
    visible=True,
    mode='lines+markers',
    stream=stream2         # (!) embed stream id, 1 per trace
)

trace3 = Scatter(
    x=[],
    y=[],
    name = 'accel_z',
    showlegend=True,
    visible=True,
    mode='lines+markers',
    stream=stream3         # (!) embed stream id, 1 per trace
)

## trace4-6 for gyro
trace4 = Scatter(
    x=[],
    y=[],
    name = 'gyro_x',
    showlegend=True,
    visible=True,
    mode='lines+markers',
    yaxis='y2',
    stream=stream4         # (!) embed stream id, 1 per trace
)

trace5 = Scatter(
    x=[],
    y=[],
    name = 'gyro_y',
    showlegend=True,
    visible=True,
    mode='lines+markers',
    yaxis='y2',
    stream=stream5         # (!) embed stream id, 1 per trace
)

trace6 = Scatter(
    x=[],
    y=[],
    name = 'gyro_z',
    showlegend=True,
    visible=True,
    mode='lines+markers',
    yaxis='y2',
    stream=stream6         # (!) embed stream id, 1 per trace
)


## trace7-9 for roll pitch yaw
trace7 = Scatter(
    x=[],
    y=[],
    name = 'roll',
    showlegend=True,
    visible=True,
    mode='lines+markers',
    yaxis='y3',
    stream=stream7         # (!) embed stream id, 1 per trace
)

trace8 = Scatter(
    x=[],
    y=[],
    name = 'pitch',
    showlegend=True,
    visible=True,
    mode='lines+markers',
    yaxis='y3',
    stream=stream8         # (!) embed stream id, 1 per trace
)

trace9 = Scatter(
    x=[],
    y=[],
    name = 'yaw',
    showlegend=True,
    visible=True,
    mode='lines+markers',
    yaxis='y3',
    stream=stream9         # (!) embed stream id, 1 per trace
)




data = Data([trace1, trace2, trace3, trace4,trace5, trace6, trace7,trace8, trace9])

# Make a figure object
#fig = tls.make_subplots(rows=1, cols=1)
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
#tls.embed('streaming-demos','12')
