""" Online signature calculator, plotter, server

Inputs: 
* scope data
* Modulation parameters (shared params ok)

* MAY read fitted data too (not clear yet)
* MAY read other slow modulation params (not clear yet)

Outputs:
* A signature dictionary
    * Maybe saved to disk, maybe served via ZMQ?
* Perhaps a graph of the latest sigs

"""
import numpy as np
import DPM
import pyqtgraph as pg
from time import sleep
import threading
from box import Box
import shared_parameters
import zmq, pickle

glbP = shared_parameters.SharedParams("NECOM")

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
    else:
        print(" no");
dpm = None
glb_RUN = False
glb_INITED = False
def init():
    global dpm
    global glb_INITED
    if dpm is not None:
        close()
    subscribe()
    dpm = DPM.DockPlotManager("signatures")
    glb_INITED = True

data = None
t = None

pp = Box( 
        sleep = 0.1,
        func = lambda t, y: {'raw':{'x':t, 'y':y[0]}}
        )

def calc_square_mod_vals(cycle, modParamD):
    mod_valD = {}
    for key, [amp,period] in modParamD:
        mod_valD[key] = amp*(1.0 if cycle%period-period/2 > 0 else -1.0)
    return mod_valD


def update():
    global data, t
    reply = getData()
    if reply is None:
        print("Nothing recieved")
        return
    topic, datD=reply
    dt=datD['dt']
    data=datD['data']

    if len(data): # Not sure if this is actually necessary- since this may be done on the sending end already.
        try:
            data=np.vstack(data)
        except ValueError: # if they're not all the same length we'll trim them -- but this may be the sign of a problem
            print("clipping...")
            newMaxL=min([r.size for r in data])
            data=[r[:newMaxL] for r in data]
    t=np.arange(data[0].size)*dt

    # Need a mechanism to sync
    mod_valD = calc_square_mod_vals(cycle, modParamD)
    sig_calculator.update(data, mod_valD)

    new_sigs = sig_calculator.signatures
    glbP.p.signatures = new_sigs

    #Plot
    for name, dataD in pltDict.items():
        dpm.addData(name, dataD)
    #plot
    #dpm.addData("raw", {'x':tPlt, 'y':yPlt})

def runContinuously():
    while glb_RUN:
        update()
        print('.')
        sleep(0.1)

glb_RUN_THREAD =None
timer = None
def start(interval =0.1):
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