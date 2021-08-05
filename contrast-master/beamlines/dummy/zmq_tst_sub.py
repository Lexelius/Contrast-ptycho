import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('tcp://127.0.0.1:2000')
socket.setsockopt(zmq.SUBSCRIBE, b"")

method = 1

if method == 1:
    while(True):
        message = socket.recv_pyobj()
        print(message.get(1))
        ##print(message)


elif method == 2:
    listeners = [1,2]
    while(True):
        message = socket.recv_pyobj()
        msgIndex = message.keys()[0]
        if(msgIndex in listeners):
            print(message.get(msgIndex))

