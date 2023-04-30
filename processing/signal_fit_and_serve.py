"""
Transforms the stream of acquired traces to signal values: ideally generically.
Then outputs it on another ZMQ socket.

To know what to fit to, it reads 'signatures' from somewhere: from disk or served by 'sginature_calculator.py'

inputs:
* data source
* pre-processing function (masking out areas etc)
* actual fitting function (traces, currentSignatures, fitParams)
* initialSignatures
* function to update signatures <- should prob. be external code.

* If no signatures are given, this should be able to work them out somehow.

"""

import numpy as np
import shared_parameters
#import frame_preprocessing

import zmq
from statsmodels import api as sm
from box import Box
import time
from async_components import ZMQSubReadChunked, ChunkedSourcePlotter, ZMQChunkedPublisher, AsyncTransformer

glbP = shared_parameters.SharedParams("NECOM")

#_---------------------------------------------------
# Redo:
def fitting_func(traces, signatures, bReturnObjs = False):

    resF=(lambda mod: mod.fit()) if bReturnObjs else (lambda mod: mod.fit().params)

    exog = np.array(list(signatures.values())).T
    exog = sm.add_constant(exog)
    model=sm.OLS(traces[0], exog)#, missing='drop')
    resL=[resF(model)]    

    if len(traces)>1:
        for trace in traces[1:]:
            model.endog[:]=trace #sig.stack()
            resL.append(resF(model))

    resA = np.array(resL)
    resD = {label: vals for label, vals in zip(['dc']+list(signatures.keys()), resA.T)}
    return resD

def transformF(new_data, state):
    signatures = {name:state.signatures[name] for name in state.to_fit }
    data = new_data['data']
    t = data['tL']
    yL= data['datL']
    resD = fitting_func(yL, signatures)
    resD['tL'] = t
    return resD
    
def update_state(new_data, state):
    if glbP.changedSinceLoad() or state.signatures is None:
        state.signatures = glbP.loadArray('signatures')
    Nsamps = len(new_data['data']['tL'])
    state.N_sent += Nsamps
    print(f"Sample rate: {state.N_sent/(time.time() - state.t_start)}")
    return state


IN_PORT = 5560
OUT_PORT = 5562
def main():
    state0 = Box(
        signatures = None,
        N_sent = 0,
        t_start = time.time(),
        to_fit = ["Bx_1", "Bz_1", "pump_Phi"]
    )
    reader = ZMQSubReadChunked(port = IN_PORT, topic = "raw")
    sender = ZMQChunkedPublisher(port = OUT_PORT, topic = "fitted")
    transformer = AsyncTransformer(
            inputF = reader.retrieve,
            transformF = transformF,
            state_updateF = update_state,
            outputF = lambda args: sender.send(**args),
            run_interval = 0.1,
            state = state0,
            )
    # Plotter
    reader = ZMQSubReadChunked(port = OUT_PORT, topic = "fitted")
    def getData():
        new_dat = reader.retrieve()
        if new_dat:
            print(new_dat['data'].keys())
        return new_dat
    plotter = ChunkedSourcePlotter(
            inputF = getData,
            label = 'fitted',
            poll_interval= 0.2,
            )
    transformer.start()
    plotter.start()
    return plotter, transformer


def set_max_samples(N):
    for itm in plotter.dpm.dataD.items():
        itm[1][0].max_samples=N

def set_to_fit(names):
    transformer.state.to_fit = names

def clearPlot():
    for itm in plotter.dpm.dataD.items():
        itm[1][0].clear()

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    #start()
    plotter, transformer = main()
    def stop():
        transformer.stop()
        plotter.stop()
    def start():
        transformer.start()
        plotter.start();
