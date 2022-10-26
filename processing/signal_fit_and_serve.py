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

class FitServer(object):
    PORT = "5561"
    streamFile=None

    def __init__(self, signalSource, fittingFunc, outputFunc, initialSignatures=None, signatureUpdateFunc=None):
        


        signalSource.init()

        #acq.subscribe(b'seg')
        self.updateGradProc(True)


        self.sampRate=sampRate
        Nhist=300
        self.signalHist=np.zeros((Nhist, 3))
        self.calFact=np.ones(3, dtype='f8')
        self.modAmpL=[0.05,0.05,0.05]
        self.modFreqL=[5,3,1]
        #self.tAx=arange(Nhist)/self.sampRate
        self.bStopStreaming=False
        self.tHist=[]
        self.tRate = 0
        self.tLast = 0
    def updateGradProc(self, justDoIt=False):
        #print("check for updated grad")
        if glbP.isCalUpdated() or justDoIt==True: 
            cal=glbP.loadCal()
            self.gradD= {key:cal['grad'][key]  for key in self.varsToFit}
            t=cal['t']
            self.p = glbP.p
            #self.gradD={key: zeroMiddle(t, self.gradD[key], 100e-6,cal['pars'].pulseTiming.tau-100e-6) for key in self.gradD}
            self.gradProcessedD=preProcessGrad(t, self.gradD, cal['pars'])
            self.gwGrad.updateFromDict(self.gradProcessedD)

    def update(self):
        # self.signatures = updateSignatures()
        # traces = self.signalSource.retrieveData()
        # signals = self.fitFunc(traces)
        # publish(signals)
        # if self.bSaveToFile:
        #   saveToFile(signals)
        #datL=acq.checkForPublished()
        #topic,(t0L,rawL,dt)=acq.checkForPublished()

        tL, newTraces = self.signalSource.getWaiting()
        fit_signals = self.fittingFunc(newTraces, self.signatures)
        self.outputFunc(tL, fit_signals)

        def update():
            #sigPre = preProcessSig(t, rawL, self.p, subRef=self.gradD['ref'])
            signal = fitSimp(sigPre, self.gradProcessedD, addDC=True)

            self.signal = np.array(signal)
            print("sent {} segs".format(len(signal)))
            mag2Send=signal
            #mag2Send[:,:3]/=self.calFact
            msg=b'mag '+ pickle.dumps( (datD['t'], mag2Send) )
            self.SOCKET.send(msg)

            n = len(rawL)
            self.tRate = (self.tRate + n/(datD['t'][-1]-self.tLast)*9 )/10
            print('tRate:', self.tRate)
            #print(datD['t'])

            self.tLast=datD['t'][-1]


        self.signatures = self.signatureUpdate(newTraces) #if necessary

        if self.streamFile is not None:
            self.signal.to_numpy().tofile(self.streamFile)
        #else:
        #    self.updateGradProc()
        Nnew=self.signal.shape[0]
        self.signalHist=np.roll(self.signalHist, -Nnew, axis=0)
        self.signalHist[-Nnew:]=self.mag[:,:3]

        if self.signalHist[0].mean()!=0:
            #pdb.set_trace()
            self.updateCals()
            #self.bStopStreaming=True

    def updateCals(self): # this will be moved to signature_calculator.py
        """Update calibrations using known modulatons"""
        tAx=np.arange(self.signalHist.shape[0])/self.sampRate
        self.signalHist-=self.signalHist.mean(axis=0)
        for k,f in enumerate(self.modFreqL):
            Tpts=1./f*self.sampRate
            N=int(np.floor(tAx.size/Tpts)*Tpts)
            sinQuad=((np.sin(2*np.pi*f*tAx[:N])*self.signalHist[:N,k]).mean())
            cosQuad=((np.cos(2*f*np.pi*tAx[:N])*self.signalHist[:N,k]).mean())
            self.calFact[k]=np.sqrt( sinQuad**2  + cosQuad**2)/self.modAmpL[k]*2
            amp=sinQuad + 1j*cosQuad
            print(self.calFact[k])

    def startSavingToFile(self, filename, notes=""):
        datDir="c:/PulseMagData/magStreams/"
        self.streamFile=open(datDir+filename+"_strm.npz", 'wb')
        np.savez(datDir+filename+"_meta.npz", notes=notes, grad=self.gradD, pars=glbP.p)

    def stopStreaming(self):
        self.streamFile.close()
        self.streamFile=None




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
        to_fit = ['Bx', 'By', 'pump_Theta']
    )
    reader = ZMQSubReadChunked(port = IN_PORT, topic = "raw")
    sender = ZMQChunkedPublisher(port = OUT_PORT, topic = "fitted")
    transformer = AsyncTransformer(
            inputF = reader.retrieve,
            transformF = transformF,
            state_updateF = update_state,
            outputF = lambda args: sender.send(**args),
            run_interval = 0.2,
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
            poll_interval= 0.1,
            )
    transformer.start()
    plotter.start()
    return plotter, transformer


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    #start()
    plotter, transformer = main()
    def stop():
        transformer.stop()
        plotter.stop()
