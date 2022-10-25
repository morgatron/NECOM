import numpy as np
import DPM
import pyqtgraph as pg
from time import sleep
from PyQt5.QtWidgets import QApplication
from box import Box
from async_components import ZMQSubReadChunked, ChunkedSourcePlotter

import zmq, pickle

glbSOCKET_PUBSUB =None
glbPORT_PUBSUB = 5560
def subscribe():
    global glbSOCKET_PUBSUB
    glbSOCKET_PUBSUB = zmq.Context().socket(zmq.SUB)
    glbSOCKET_PUBSUB.set_hwm(5)
    glbSOCKET_PUBSUB.connect("tcp://localhost:%s" % glbPORT_PUBSUB)
    glbSOCKET_PUBSUB.setsockopt(zmq.SUBSCRIBE, b"raw")

def getData():
    if glbSOCKET_PUBSUB.poll(10):
        topic, msg = glbSOCKET_PUBSUB.recv().split(b' ', 1)
        return topic, pickle.loads(msg)

dpm = None
glb_RUN = False
glb_INITED = False
def init():
    global dpm
    global glb_INITED
    if dpm is not None:
        close()
    subscribe()
    dpm = DPM.DockPlotManager("scope")
    glb_INITED = True

data = None
t = None

pp = Box( 
        sleep = 0.1,
        func = lambda t, y: {'raw':{'x':t, 'y':y[0]}}
        )

def update():
    global data, t
    reply = getData()
    if reply is None:
        print(".", end="")
        return
    topic, datD=reply
    dt=datD['dt']
    data=datD['data']
    if len(data):
        try:
            data=np.vstack(data)
        except ValueError: # if they're not all the same length we'll trim them -- but this may be the sign of a problem
            print("clipping...")
            newMaxL=min([r.size for r in data])
            data=[r[:newMaxL] for r in data]
    t=np.arange(data[0].size)*dt
    pltDict = pp.func(t, data)
    if 'x' in pltDict:
        dpm.addData('raw', pltDict)
    else:
        for name, dataD in pltDict.items():
            dpm.addData(name, dataD)
    #plot
    #dpm.addData("raw", {'x':tPlt, 'y':yPlt})

def plotContinuously():
    while glb_RUN:
        update()
        print('.')
        sleep(0.1)

glb_RUN_THREAD =None
timer = None
def start(interval =0.2):
    global timer
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(interval*1000)

def stop():
    timer.stop()

def close():
    stop()
    #acq.close()
    glbSOCKET_PUBSUB.close()
    del dpm

def split_into_minus_plus(t, y):
    N = t.size//2
    y1, y2 = y[:N], y[-N:]
    t1 = t[:N]
    return {'m': {'x':t1, 'y':(y1 - y2)}, 'p': {'x':t1, 'y':(y1 + y2)}}


def processStream(data, metaData = {}, Nread = None):
    print(f"{data.keys}")
    tL = data['tL']
    y = data['datL']
    mnY = np.mean(y, axis=0)
    N = mnY.size
    t = np.arange(N)
    if 'dt' in metaData:
         t *= metaData['dt']

    data = {'raw': {'x': t, 'y': mnY}}
    data |= split_into_minus_plus(t, mnY)
    return data# {'data':data, "metaData":metaData, 'Nread': N}




IN_PORT = 5560
def main():
    reader = ZMQSubReadChunked(port = IN_PORT, topic = "raw")
    plotter = ChunkedSourcePlotter(
            inputF = lambda : reader.retrieve(10),
            preProcessF = lambda d: processStream(d['data'], d['metaData']) if d else None,
            label = 'raw',
            defaultPlotKwargs = dict(mode=0), # Replace mode
            poll_interval=0.2
            )
    return plotter



#pp.func= preProc
#def wrap(func):
#    def wrapped(t, data):
#        d = func(t, data)
#        return {'nam'}
#    return wrapped


import time
if __name__ == "__main__":
    app = pg.mkQApp("Plotting Example")
    #pg.exec(u)
    #app = QApplication([])
    plotter = main()
    def stop():
        plotter.stop()
    plotter.start()
    #init()
    #start()
