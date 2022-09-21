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




## Synchronisation
It seems likely this library will be started wihout being precisely 
synchronised with the start of hte modulation. So the modulations 
won't be properly in phase.

A 'simple' (if not efficient) approach is to always calculate both
quadratures of each signal, and keep track of the 'phase' of each 
modulation response. A simple metric would be the size of DC resposne.
A problem with this method is that we don't a-priori know the sign of
each response, so there's a pi radian ambiguity.
However, given that we know the nominal phase between each signal, 
perhaps there's a method to solve for time zero and extract the 
correct phase? This would be a function called only to begin with.

This might need some drawing out...

Alternatively, it _will_ get the right shape of the signatures
with out correcting for this phase, just not always the right sign.
It _could_ be corrected manually by adding slow offsets, perhaps
doing everything from control.py to get it in roughly the right...
but of course then we'd have to synchronise manually by looking at 
the signals anyway!

The only question is whether to do it all here, or to assume we're already close.


## Functional Overview

Parameters:

@start... starts the serving
@on_timer is run regularly to check for new data from scope. 
    It'll check several conditions, and run other functions if necessary including:
@on_mod_params_updated when new modulation parameters are found using shared_parameters
@find_t0 can be run after running for a while to try and get signature signs right. It may
be run automatically some time after startup/parameter change if phases aren't all zero.


"""
import numpy as np
import DPM
import pyqtgraph as pg
from time import sleep
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

def calc_square_mod_vals(cycle, modParamD, phase_lag=0):
    param2val = {}
    for key, [amp,period] in modParamD:
        cycle_offset = int(phase_lag*period)
        param2val[key] = amp*(1.0 if (cycle+cycle_offset)%period-period/2 > 0 else -1.0)
    return param2val


def new_sigs(t_cycle, dataL, freqs, phis, old_sigs, N_cycles2ave = 100 ):
    pass

def new_phases(t_cycles, dataL, freqs, phis, old_phases, N_cycles2ave=100 ):
    pass

state = Box( t_cycles = 0,
            b_monitor_phases = True,
            t_last_change = 0,
            sigs = None,
            sigs_90 = None,
            lags = None)
def on_timer():
    global data, t_cycles
    reply = getData()
    if reply is None:
        print("Nothing recieved")
        return
    topic, datD=reply
    dt=datD['dt']
    newData=datD['data']
    if len(data): # Not sure if this is actually necessary- since this may be done on the sending end already.
        try:
            data=np.vstack(data)
        except ValueError: # if they're not all the same length we'll trim them -- but this may be the sign of a problem
            print("clipping...")
            newMaxL=min([r.size for r in newData])
            newData=[r[:newMaxL] for r in newData]
    #t_cycle=np.arange(newData[0].size)*dt

    # Actually let's just for-loop this!
    {label:calc_new_sig(t_cycle, state.sigs[label], p.period, p.Nave) for label, p in state.mod_paramsD }
    state.sigs = calc_new_sigs()
    if state.b_monitor_phases:
        state.sigs_90 = calc_new_sigs(... lag = 0.5)
        lags = ...
        if not all( int(lags)==0 for new_lag in lags):
            print(lags)

        if (t_cycles - state.t_last_change > 1000) and 0:
            state.b_monitor_phases = False
        

    # Need a mechanism to sync
    mod_valD = calc_square_mod_vals(cycle, modParamD)
    sig_calculator.update(data, mod_valD)

    new_sigs = sig_calculator.signatures
    glbP.p.signatures = new_sigs
    t_cycles += len(data)

    #plot
    #dpm.addData("raw", {'x':tPlt, 'y':yPlt})

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