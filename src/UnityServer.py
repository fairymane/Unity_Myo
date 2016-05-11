
import time
import sys 
import OSC

def handler(addr, tags, data, client_address):
    txt = "OSCMessage '%s' from %s: " % (addr, client_address)

    txt += str(data)
    print(txt)

def get_stream():
    s = OSC.OSCServer(('127.0.0.1', 8889))  # listen on localhost, port 8889
    # s.addMsgHandler('/dx', handler) 
    # s.addMsgHandler('/dy', handler)     # call handler() for OSC messages received with the /startup address
    # s.addMsgHandler('/roll', handler)     # call handler() for OSC messages received with the /startup address
    # s.addMsgHandler('/pitch', handler) 
    # s.addMsgHandler('/yaw', handler)
    s.addMsgHandler("/imu", handler) 
    s.addMsgHandler("/gesture_label", handler) 
    s.serve_forever()

if __name__ == "__main__":
	get_stream()
