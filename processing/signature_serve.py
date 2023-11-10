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
from async_components import ZMQSubReadChunked, ZMQSubReadData, ZMQPubSendData, AsyncTransformer, SimpleSourcePlotter

glbP = shared_parameters.SharedParams("NECOM")

trigCommIdxD = {
    "pump_Phi":2,
    'pump_Theta':3,
    'Bx': 4,
    'Bz': 5,
    'Bx_1': 7,
    'Bz_1': 6,
    'Nx': 8,
    'Nz': 9,
}

def calc_square_mod_vals(cycle, modParamD, phase=0):
    param2val = {}
    for key, [amp,period] in modParamD:
        cycle_offset = int(phase/pi*period)
        param2val[key] = amp*(1.0 if (cycle+cycle_offset)%period-period/2 > 0 else -1.0)
    return param2val

def calc_new_sig_quads(t_cycle, old_quads,  new_data, period, partially_calced_quads= None, amp = 1, periods2ave = 100, phase=0):
    """
    t_cycle is number of reps since this mod was last flagged with a rising edge
    """
    if amp == 0:
        print("Amp == 0, returning nothin'")
        return old_quads,None
    zeroArr = lambda: np.zeros(new_data.shape[1], dtype='f8')
    #cycle_offset = int(phase/(2*pi)*period)

    N_new = len(new_data)
    N_partial = (t_cycle - N_new)%period # Number of traces already averaged

    if old_quads is None:
        quads = [zeroArr(), zeroArr()]
        N_partial = 0
    else:
        quads = [el.copy() for el in old_quads]
    if partially_calced_quads is None:
        partially_calced_quads = [zeroArr(), zeroArr()]

    tAx = t_cycle  + np.arange(N_new)
    #tAx90 = tAx + period/2
    amps = 1/amp*np.sin(tAx/period*2*pi - 0*pi/4)
    amps90 = 1/amp*np.cos(tAx/period*2*pi -0*pi/4)
    #amps = np.where( tAx%period-period/2 > 0, 1/amp, -1/amp )
    #add up until end 
    ind = 0
    ind_next_period = period - N_partial 
    periods2ave = periods2ave/period
    aveF = lambda cur, new: (cur*(periods2ave-1) + new/period)/periods2ave
    #print("N_partial: ", N_partial)
    while 1:
        slc = slice(ind, ind_next_period )
        #print("slice: ", slc)
        partially_calced_quads[0] += (amps[slc,None]*new_data[slc]).sum(axis=0)
        partially_calced_quads[1] += (amps90[slc,None]*new_data[slc]).sum(axis=0)

        ind = ind_next_period
        ind_next_period += period
        if ind <=N_new:
            #print("adding period: periods2ave = {}".format(periods2ave))
            quads= [aveF(quads[0], partially_calced_quads[0]), 
                    aveF(quads[1], partially_calced_quads[1])]
            partially_calced_quads[0] = zeroArr()
            partially_calced_quads[1] = zeroArr()
        if ind >= N_new:
            #print("breaking: N_new = {}, N_partial".format(N_new))
            break

    return quads, partially_calced_quads


#Unused
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
    return new_t

def check_period_and_phase(t_idx0, flag_idx, newFlags, lastFlag, prevEdge=None):
    allFlags = np.bitwise_and(np.hstack([[lastFlag], newFlags]), 1 << flag_idx)
    rising_idxs = np.where(np.diff(allFlags) > 0)[0]

    newPeriod = None
    finalEdge = None
    if len(rising_idxs) >0:
        rising_idxs += t_idx0
        if prevEdge:
            newPeriod = rising_idxs[0] - prevEdge
        elif len(rising_idxs) > 1:
            newPeriod = rising_idxs[1] - rising_idxs[0]
        finalEdge = rising_idxs[-1]

    print(f"finalEdge: {finalEdge}, newPeriod: {newPeriod}, prevEdge: {prevEdge}")
    return finalEdge, newPeriod

def update_state(new_data, state):
    mod_params = state['mod_paramD']
    Nave = state['Nave']
    #t,dt, newData = new_data 
    newData = np.array(new_data['data']['datL'])
    t = np.array(new_data['data']['tL'])
    flags = np.array(new_data['data']['flagsL'])

    if state['ref'] is None:
        state['ref'] = np.zeros(newData.shape[1])
    state['ref'] = ((Nave-1)*state['ref'] + newData.mean(axis=0))/Nave
    newData = newData - 1*state['ref'][None,:]

    #Calculate both quadratures at once, but only using integer number of periods.
    #Remainder data is multiplied/summed a usual, but not added to the running average.
    # It's instead returned so the summing can be finished when more data comes in
    new_sig_quads_and_partials = {
        label : calc_new_sig_quads(state.t_cycles - state.lastEdges[label], 
            old_quads = state.sig_quads[label] if label in state.sig_quads else None,
            new_data=newData, 
            period=params.period_cycles, 
            partially_calced_quads = state.partial_quads[label] if label in state.partial_quads else None,
            amp = params.amp, 
            periods2ave = state.periods2ave
            )
        for label, params in mod_params.items()
        if label in state.lastEdges and params.amp>0
    }
    partial_quads = {label: val[1] for label, val in new_sig_quads_and_partials.items()}
    sig_quads = {label: val[0] for label, val in new_sig_quads_and_partials.items()}

    #check_period_and_phase.lastFlag = 0b111111111
    #check_period_and_phase.prevEdges = {}#6*[None]
    #check_period_and_phase.prevPeriods = {}#6*[None]

    # Get phase lag sync if needed
    # For anything unsynced, we see if there's enough data available now to sync
    # (i.e. at least to period starts)
    # If yes, we get the lag value and compare it to the expected period
    # If not, we record any
    for label in state.yet_to_sync:
        lastLastEdge = state.lastEdges[label] if label in state.lastEdges else None
        lastEdge, measuredPeriod = check_period_and_phase(state.t_cycles, 
                            trigCommIdxD[label], newFlags =flags, 
                            lastFlag = state.lastFlag, prevEdge = lastLastEdge )
        if measuredPeriod is not None:
            if measuredPeriod != mod_params[label].period_cycles:
                print(f"WARNING: GOT PERIOD {measuredPeriod} FOR SIGNAL {label}, EXPECTED {mod_params[label].period_cycles}")
            state.lags[label] = lastEdge % measuredPeriod
            state.yet_to_sync.remove(label)
        if lastEdge:
            state.lastEdges[label] = lastEdge

    #print(state.lags)
    #======================
    #def calc_angle(sig_x, sig_y):
    #    sig_x_sm = util.smooth(sig_x, 20)
    #    sig_y_sm = util.smooth(sig_y, 20)
    #    sgn =  1 if np.sum(sig_x_sm*sig_y_sm)>0 else -1
    #    x = abs(sig_x_sm).sum()
    #    y = abs(sig_y_sm).sum()
    #    return atan(sgn*y/x)

    #phase_lags = {label: calc_angle(*sig_quads[label]) for label in mod_params}
    #if not all( int(lags)==0 for new_lag in lags):
        #print(lags)
    if glbP.changedSinceLoad() and 0:
        state.mod_paramD = glbP.P.modsSynced.copy()
        print("realoding mod_paramD")

    t_cycles = state.t_cycles + len(newData)
    do_write = t_cycles - state.last_write > 200
    last_write =  t_cycles if do_write else state.last_write
    #rint(state.t_cycles)
    state.update({"sig_quads": sig_quads,
            "partial_quads": partial_quads,
            "t_cycles": state.t_cycles + len(newData),
            "do_write": do_write,
            "last_write": last_write,
            "lastFlag" : flags[-1]
            })
    return state
    #return Box({"sigs": new_sigs,
    #        "sigs_90": new_sigs_90,
    #        "lags":phase_lags,
    #        "t_cycles": state.t_cycles + len(newData),
    #        "do_write": do_write,
    #        "last_write": last_write,
    #        "mod_paramD": state['mod_paramD'],
    #        "periods2ave": state['periods2ave']
    #        })


def transformF(new_data, state):

    if state.do_write:
        #state.t_cycles = find_t0(state.t_cycles, [p.period_cycles for _,p in mod_paramD.items()], 
        #            state.lags.values())
        def calcNorm(sig0, sig90, lag):
            sigA = cos(lag)*sig0 + sin(lag)*sig90
            return sigA
        #sigAs = {label: calcNorm(*state.sig_quads[label], state.lags[label]) for label in state.sig_quads }
        sigAs0 = {label: state.sig_quads[label][0] for label in state.sig_quads}
        sigAs90 = {label: state.sig_quads[label][1] for label in state.sig_quads}
        magnitudes = {label: np.sqrt(np.mean(sigA0**2)+ np.mean(sigA90**2)) for label, sigA0, sigA90 in zip(sigAs0.keys(),sigAs0.values(), sigAs90.values() )}
        print("MAGNITUDES: {} ".format([f"{label}: {mag:.3f}:" for label, mag in magnitudes.items()  ]))
        phases = {label: atan2(np.sqrt(np.mean(sigA0**2)), np.sqrt(np.mean(sigA90**2))) for label, sigA0, sigA90 in zip(sigAs0.keys(),sigAs0.values(), sigAs90.values() )}
        print("PHASES: {} ".format([f"{label}: {phase:.3f}:" for label, phase in phases.items()  ]))
        #sigAs = {label: state.sig_quads[label][0] for label in state.sig_quads}
        #sigAs.update({str(label) + '_90': state.sig_quads[label][1] for label in state.sig_quads})

        sigAs = {label+'_0': sigAs0[label]/(magnitudes[label]) for label in sigAs0}
        sigAs |= {label+'_90': sigAs90[label]/(magnitudes[label]) for label in sigAs90}

        glbP.saveArray(signatures = sigAs)
        return sigAs

    else:
        return None
    ##plot
    #if state.t_cycles > state.last_plot + 200:
    #    xAx = np.arange(newData.shape[-1])
    #    for label, sig in state.sigs.items():
    #        dpm.dockD['sigs'].addData(label, {'x':xAx, 'y': sig})
    #    for label, sig in state.sigs_90.items():
    #        dpm.dockD['sigs_90'].addData(label, {'x':xAx, 'y': sig})
    #    state.last_plot = state.last_plot + 200
    #state.t_cycles += len(newData)
    

def getTestModParamD():
    p = glbP.P.modsSynced.copy()
    trimmed = Box (Bz = p.Bz, Bx = p.Bx)
    return trimmed

IN_PORT = 5560
def main():

    state0 = Box({
        "sig_quads": {},
        "partial_quads": {},
        "lags": {},
        "t_cycles": 0,
        "do_write": False,
        "mod_paramD" : glbP.P.modsSynced.copy(),
        #"mod_paramD" : getTestModParamD(),#glbP.P.modsSynced.copy(),
        "last_write": 0,
        "periods2ave": 256*60,
        "Nave" : 40000,
        'ref': None,
        'yet_to_sync': list(glbP.P.modsSynced.keys()),
        "lastFlag": 0b111111111,
        "lastEdges": {},
    })
    reader = ZMQSubReadChunked(port = IN_PORT, topic = "raw")
    sender = ZMQPubSendData(port = IN_PORT+1, topic = "signatures")
    transformer = AsyncTransformer(
            inputF = reader.retrieve,
            transformF = transformF,
            state_updateF = update_state,
            outputF = lambda args: sender.send(**args),
            run_interval = 0.2,
            state = state0,
            )
    # Plotter
    reader0 = ZMQSubReadData(port = IN_PORT+1, topic = "signatures")
    def getData0():
        new_dat = reader0.retrieve()
        if new_dat is not None:
            new_dat = {key:new_dat[key] for key in new_dat if not key.endswith('90')}
        return new_dat
    reader90 = ZMQSubReadData(port = IN_PORT+1, topic = "signatures")
    def getData90():
        new_dat = reader90.retrieve()
        if new_dat is not None:
            new_dat= {key:new_dat[key] for key in new_dat if key.endswith('90')}
        return new_dat
    plotter = SimpleSourcePlotter(
            inputF = getData0,
            label = 'signatures0',
            poll_interval= 1.0,
            )
    plotter2 = SimpleSourcePlotter(
            inputF = getData90,
            label = 'signatures90',
            poll_interval= 1.0,
            )
    transformer.start()
    plotter.start()
    plotter2.start()
    return plotter, plotter2, transformer

timer = None
def start(interval =0.5):
    init()
    global timer
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(on_timer)
    timer.start(int(interval*1000))

def stop_old():
    timer.stop()

def close():
    stop()
    #acq.close()
    glbSOCKET_PUBSUB.close()
    del dpm



if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    #start()
    plotter, plotter2, transformer = main()
    def stop():
        transformer.stop()
        plotter.stop()
        plotter2.stop()

