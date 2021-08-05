"""
 Tutorial found at https://www.youtube.com/watch?v=-_CXA8SZsOs
"""

import zmq
from time import sleep

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind('tcp://127.0.0.1:2000')

messages = [100, 200, 300]
curMsg = 0

while(True):
    sleep(1)
    socket.send_pyobj({curMsg:messages[curMsg]})
    if(curMsg == 2):
        curMsg = 0
    else:
        curMsg = curMsg + 1



