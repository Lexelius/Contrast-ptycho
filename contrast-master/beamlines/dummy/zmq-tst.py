import zmq
import numpy as np
import matplotlib.pyplot as plt

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('tcp://127.0.0.1:5556')
socket.setsockopt(zmq.SUBSCRIBE, b"")


x = []
y = []
diff = []
status = []
np.seterr(divide = 'ignore') # ignore error from taking the log of zero. Default is 'warn'
# alternatevly you could instead plot:
# plt.matshow(np.log(np.where(diff[-1] > 1.0e-10, diff[-1], np.nan)), 0)
while True:
    message = socket.recv_pyobj()
    if len(list(message.values())) == 4:
        x.append(list(message.values())[0])
        y.append(list(message.values())[1])
        diff.append(list(message.values())[2])
        status.append(list(message.values())[3])
        plt.matshow(np.log(diff[-1]), 0) #### Wont plot anything in real time..
        ##print(f"x = {x[-1]}, y = {y[-1]}")
    ##print(message)
