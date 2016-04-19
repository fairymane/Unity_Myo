import OSC

def handler(addr, tags, data, client_address):
    txt = "OSCMessage '%s' from %s: " % (addr, client_address)
    txt += str(data)
    print(txt)

if __name__ == "__main__":
    s = OSC.OSCServer(('127.0.0.1', 8888))  # listen on localhost, port 57120
    #s = OSC.OSCServer(('192.168.70.53', 8000))  # listen on localhost, port 57120
    s.addMsgHandler('/myo/accel', handler)     # call handler() for OSC messages received with the /startup address
    s.addMsgHandler('/myo/gyro', handler)     # call handler() for OSC messages received with the /startup address
    s.addMsgHandler('/myo/emg', handler)     # call handler() for OSC messages received with the /startup address
    s.addMsgHandler('/myo/pose', handler)     # call handler() for OSC messages received with the /startup address
    s.addMsgHandler('/myo/orientation', handler)     # call handler() for OSC messages received with the /startup address
    s.serve_forever()
