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

"""

import numpy as np
#import frame_preprocessing



import zmq
class FitServer(object):
    PORT = "5561"
    streamFile=None

    def __init__(self, signalSource, fittingFunc, initialSignatures=None, signatureUpdateFunc=None):
    def __init__(self, varsToFit = ['Bx', 'By', 'Bz'], sampRate=15):
        import pmAcquire as acq

        acq.init(bRemote=True)
        acq.subscribe(b'raw')                
        self.varsToFit = varsToFit

        #acq.subscribe(b'seg')
        self.updateGradProc(True)
        self.SOCKET= zmq.Context().socket(zmq.PUB)
        self.SOCKET.set_hwm(10)
        self.SOCKET.bind("tcp://*:%s" % self.PORT)


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
        self.updateGradProc() #if necessary
        #datL=acq.checkForPublished()
        #topic,(t0L,rawL,dt)=acq.checkForPublished()
        rep=acq.checkForPublished()
        if rep is None:
            #print("Nothing recieved")
            return
        topic,datD=rep
        dt=datD['dt']
        rawL=datD['data']

        if len(rawL):
            try:
                rawL=np.vstack(rawL)
            except ValueError: # if they're not all the same length we'll trim them -- but this may be the sign of a problem
                newMaxL=min([r.size for r in rawL])
                rawL=[r[:newMaxL] for r in rawL]
            t=np.arange(rawL[0].size)*dt

            sigPre = preProcessSig(t, rawL, self.p, subRef=self.gradD['ref'])
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


