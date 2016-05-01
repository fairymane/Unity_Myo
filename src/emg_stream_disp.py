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
#py.sign_in(username="fairymane", api_key="x4kt8y1smu")
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

stream7 = Stream(
    token=stream_id7,  # (!) link stream id to 'token' key
    maxpoints=20      # (!) keep a max of 80 pts on screen
)

stream8 = Stream(
    token=stream_id8,  # (!) link stream id to 'token' key
    maxpoints=20      # (!) keep a max of 80 pts on screen
)



#file_ = '/Users/taofeng/Dropbox/git/LayeredSensing/source/data/emg_data.csv'
file_ = sys.argv[1]
#df = pd.read_csv(file_, index_col = 'formatted_time').dropna()
df = pickle.load(open(file_, "rb"))
print df.columns
#df = df1[['timestamp','emg1','emg2','emg3','emg4','emg5','emg6','emg7']]
#df.columns = ['emg1','emg2','emg3','emg4','emg5','emg6','emg7','emg8']

#xx = df['timestamp'].values;
xx = df.index.values;
emg1 = df['em1'].values
emg2 = df['em2'].values
emg3 = df['em3'].values

emg4 = df['em4'].values
emg5 = df['em5'].values
emg6 = df['em6'].values

emg7 = df['em7'].values
emg8 = df['em8'].values



## trace1-3 for accel
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

i = 0
N = len(emg1);

while i<N:
    x  = xx[i]
    e1 = emg1[i]
    e2 = emg2[i]
    e3 = emg3[i]
    e4 = emg4[i]
    e5 = emg5[i]
    e6 = emg6[i]
    e7 = emg7[i]
    e8 = emg8[i]

    s1.write(dict(x=x, y=e1))
    s2.write(dict(x=x, y=e2))
    s3.write(dict(x=x, y=e3))
    s4.write(dict(x=x, y=e4))
    s5.write(dict(x=x, y=e5))
    s6.write(dict(x=x, y=e6))
    s7.write(dict(x=x, y=e7))
    s8.write(dict(x=x, y=e8))

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
  
#tls.embed('streaming-demos','12')
