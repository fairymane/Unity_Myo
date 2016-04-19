import os
from socket import *
host = "127.0.0.1" # set to IP address of target computer
#host = "192.168.70.53" # set to IP address of target computer
port = 8887
addr = (host, port)
UDPSock = socket(AF_INET, SOCK_DGRAM)
while True:
    data = raw_input("Enter message to send or type 'exit': ")
    UDPSock.sendto(data, addr)
    if data == "exit":
        break
UDPSock.close()
os._exit(0)
