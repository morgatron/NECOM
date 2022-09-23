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

    def saveArray(self, **arrs):
        dirName = os.path.dirname(self.curParFilename)
        for name, arr in arrs.items():
            filePath = os.path.join(dirName, name)
            if isinstance(arr, Mapping):
                np.savez(filePath, **arr)
            else:
                np.savez(filePath, only = arr )

    def loadArray(self, name):
        dirName = os.path.dirname(self.curParFilename)
        if not name.endswith(".npz"):
            name = name +".npz"
        filePath = os.path.join(dirName, name)
        loaded = dict(np.load(filePath))
        if "only" in loaded:
            return loaded['only']
        return loaded


"""
import numpy as np
import DPM
import pyqtgraph as pg
from time import sleep
from box import Box
import shared_parameters
import zmq, pickle
from math import atan2, atan, pi, sin, cos
import util

glbP = shared_parameters.SharedParams("NECOM")

glbSOCKET_PUBSUB =None
glbPORT_PUBSUB = 5560
def subscribe():
    global glbSOCKET_PUBSUB
    glbSOCKET_PUBSUB = zmq.Context().socket(zmq.SUB)
    glbSOCKET_PUBSUB.set_hwm(5)
    glbSOCKET_PUBSUB.connect("tcp://localhost:%s" % glbPORT_PUBSUB)
    glbSOCKET_PUBSUB.setsockopt(zmq.SUBSCRIBE, b"raw")


tLast = 0
dt_last = 0
def getData(Nmin = 20):
    global tLast, dt_last
    datL = []
    tL = []
    while len(datL) < Nmin:
        while glbSOCKET_PUBSUB.poll(10):
            topic, msg = glbSOCKET_PUBSUB.recv().split(b' ', 1)

            datD = pickle.loads(msg)
            t = datD['t']
            dt=datD['dt']
            dts = np.diff(t)
            newData=datD['data']
            datL.extend(newData)
            tL.extend(t)
    try:
        newData=np.vstack(datL)
    except ValueError: # if they're not all the same length we'll trim them -- but this may be the sign of a problem
        print("clipping...")
        newMaxL=min([r.size for r in newData])
        newData=[r[:newMaxL] for r in newData]
    
    dt_last = dts.mean()
    dts = np.diff([tLast]+tL)
    if(not all( (dts-dt_last)/dt_last < 0.1 )):
        print(dts)
    #print(dts)
    tLast = tL[-1]
    return tL, dt, newData


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
    dpm.addDockPlot("sigs")
    dpm.addDockPlot("sigs_90")
    glb_INITED = True

data = None
t = None


def calc_square_mod_vals(cycle, modParamD, phase=0):
    param2val = {}
    for key, [amp,period] in modParamD:
        cycle_offset = int(phase/pi*period)
        param2val[key] = amp*(1.0 if (cycle+cycle_offset)%period-period/2 > 0 else -1.0)
    return param2val
def calc_new_sig(t_cycle, old_sig, new_data, period, amp = 1, periods2ave = 100, phase=0):
    amp = 1
    if old_sig is None:
        old_sig = np.zeros(new_data.shape[1], dtype='f8')
    N_new = len(new_data)
    cycle_offset = int(phase/(2*pi)*period)
    tAx = (t_cycle + cycle_offset) + np.arange(N_new)
    amps = np.where( tAx%period-period/2 > 0, amp, -amp )
    #print(f"amps.shape: {amps.shape}")
    new_sig_contribution = (amps[:,None]*new_data).sum(axis=0)
    Nave = periods2ave*period
    #print(f"Nave: {Nave}, N_new: {N_new}")
    new_sig = ((Nave - N_new)* old_sig + new_sig_contribution)/Nave
    return new_sig


state = Box( t_cycles = 0,
            b_monitor_phases = True,
            t_last_change = 0,
            sigs = {},
            sigs_90 = {},
            lags = {},
            last_write = 0,
            last_plot = 0,
            periods2ave = 100,
            mod_paramD = None,
            )

def find_t0(t_cur, periods, phis, go_back = 30000, go_forward=1000):
    valsL = []
    errL = []
    ts2search = np.arange(t_cur+go_forward, t_cur-go_back,-1)
    for t in ts2search:
        vals= [ (t/period + phi/pi)%1 for period,phi in zip(periods,phis) ]
        valsL.append(vals)
        errL.append(sum(vals))
    t0 = ts2search[np.argmin(errL)]
    new_t = t_cur + (t_cur-t0)
    print(f"new_t, old_t: {new_t}, {t_cur}")
    return new_t

datL = []
t_L = []
tLast = 0
def on_timer():
    if glbP.changedSinceLoad() or state.mod_paramD is None:
        state.mod_paramD = glbP.P.modsSynced.copy()
    resp = getData()
    if resp is None:
        #print("Nothing recieved")
        return
    tL, dt, newData = resp
    #datL.extend(newData)
    #t_L.extend(tL)
    #t_cycle=np.arange(newData[0].size)*dt

    for label, params in state.mod_paramD.items():
        old_sig = state.sigs[label] if label in state.sigs else None
        state.sigs[label] = calc_new_sig(state.t_cycles, old_sig = old_sig, new_data=newData, period=params.period_cycles, amp = params.amp, periods2ave = state.periods2ave)

    if state.b_monitor_phases:
        phase_lags = {}
        for label, params in state.mod_paramD.items():
            old_sig = state.sigs_90[label] if label in state.sigs_90 else None
            state.sigs_90[label] = calc_new_sig(state.t_cycles, old_sig = old_sig, new_data=newData, period=params.period_cycles, amp = params.amp, periods2ave = state.periods2ave, phase=pi/2)

            xSig = util.smooth(state.sigs[label], 20)
            ySig = util.smooth(state.sigs_90[label], 20)
            sgn =  1 if np.sum(xSig*ySig)>0 else -1
            x = abs(xSig).sum()
            y = abs(ySig).sum()
            phase_lags[label] = atan(sgn*y/x)
        #print({label: "{:.3f}".format(val) for label, val in phase_lags.items()})
        state.lags = phase_lags
        #if not all( int(lags)==0 for new_lag in lags):
            #print(lags)

        if (state.t_cycles - state.t_last_change > 1000) and 0:
            state.b_monitor_phases = False
        

    # Need a mechanism to sync
    #mod_valD = calc_square_mod_vals(cycle, modParamD)
    #sig_calculator.update(data, mod_valD)

    if state.t_cycles > state.last_write + 1000:
        if state.last_write != 0:
            print("saving signatures")

            #state.t_cycles = find_t0(state.t_cycles, [p.period_cycles for _,p in mod_paramD.items()], 
            #            state.lags.values())
            sigAs = {label: cos(state.lags[label])*state.sigs[label] + sin(state.lags[label])*state.sigs_90[label]}
            glbP.saveArray(signatures = sigAs)
        state.last_write = state.t_cycles

    #plot
    if state.t_cycles > state.last_plot + 200:
        xAx = np.arange(newData.shape[-1])
        for label, sig in state.sigs.items():
            dpm.dockD['sigs'].addData(label, {'x':xAx, 'y': sig})
        for label, sig in state.sigs_90.items():
            dpm.dockD['sigs_90'].addData(label, {'x':xAx, 'y': sig})
        state.last_plot = state.last_plot + 200
    state.t_cycles += len(newData)

timer = None
def start(interval =0.5):
    init()
    global timer
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(on_timer)
    timer.start(interval*1000)

def stop():
    timer.stop()

def close():
    stop()
    #acq.close()
    glbSOCKET_PUBSUB.close()
    del dpm