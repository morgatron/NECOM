""" Local/Remote interfaces to the picoscope, mainly for streaming purposes.

Lots of historical baggage still here.

"""


from tkinter import W
from picoscope import ps5000a
import numpy as np
import zmq
from time import sleep
import numpy as np
import util
import sys
import dill as pickle
import threading
import time
from async_components import ZMQChunkedPublisher


def quickSetup(self, 
        chanAParams=dict(coupling="DC", VRange=10.0, VOffset=0, BWLimited=False, probeAttenuation=1.0),
        chanBParams=None,
        sampleRate=5e6, 
        acqTime=1e-3,
        triggerParams=dict(trigSrc="External", threshold_V=1.0, direction="Rising", delay=0, enabled=True, timeout_ms=100),
        nCaps=1, nMemorySegments=-1, 
        resolution=None,
        ):
    """Set most scope parameters in one function
    @chanA(BCD)Params: dictionary of arguments to setChannel. Note that if the dictionary is None, the channel will be disabled. 
    @sampleRate: the desired samples per second for the scope, in Hz
    @acqTime: the length of time (in seconds) to record after each trigger.
    @triggerParams: a dictionary of arguments to setSimpleTrigger
    @nCaps: The number of traces to store in memory each time runBlock is called (and by default the number that will be retrieved by getDataRawBulk)
    @nMermorySegments: the number of windows in which to divide the scope memory. If -1 (default), it will be set to @nCaps
    @resolution: calls setResolution, only for 5000 series devices.
    
    """
    self.setChannel("A",  enabled=True, **chanAParams) if chanAParams else self.setChannel("A", enabled=False)
    self.setChannel("B",  enabled=True, **chanBParams) if chanBParams else self.setChannel("B", enabled=False)
    self.setChannel("C",  enabled=False)    
    self.setChannel("D",  enabled=False)
    if resolution:
        self.setResolution(str(resolution));
    self.setNoOfCaptures(nCaps)
    self.memorySegments(nCaps if nMemorySegments==-1 else nMemorySegments)
    if nMemorySegments !=-1 and nMemorySegments<nCaps:
        raise ValueError("nMemorySegments needs to be equal to or greater than the number of captures")
    actSampleRate, maxN=self.setSamplingFrequency(sampleRate, sampleRate*acqTime)
    self.setSimpleTrigger(**triggerParams)
    if maxN < sampleRate*acqTime:
        raise ValueError("At sample rate {} with {} buffers, the maximum collection time is {}.".format(actSampleRate, nCaps, maxN/sampleRate))
setattr(ps5000a.PS5000a, 'quickSetup', quickSetup)


glbLOCAL=True
glbPS=None
glbSOCKET_PAIR=None
glbPORT_PAIR = "5558"
glbSOCKET_PUBSUB=None
glbPORT_PUBSUB = "5560"
bPLOT=False
glbSENDER = None


curve=None
win=None
def init(bRemote=True):
    global glbLOCAL
    if not bRemote:
        glbLOCAL=True
        initLocal()
    else:
        glbLOCAL=False
        initRemote()
def initLocal():
    global glbPS, glbSENDER
    try:
        try:
            glbPS.getAllUnitInfo()
        except (NameError, TypeError, AttributeError): #is closed or never opened
            try:
                glbPS=ps5000a.PS5000a()
            except OSError: #has been in use elsewhere, let's hope it's not a lost reference somewhere
                try:
                    del glbPS
                except NameError:
                    pass
                import gc
                gc.collect()
                glbPS=ps5000a.PS5000a()
        
        #glbPS.quickSetup(dict(coupling="DC", VRange=2.0), nCaps=100, sampleRate=3e6, acqTime=1/60.-50e-6, resolution=15)
        setScopeParams()
    except Exception as e:
        print("Not using picoscope, maybe it's in use elsewhere? Exception: {}".format(e))

    context = zmq.Context()
    global glbSOCKET_PAIR, glbSOCKET_PUBSUB
    glbSOCKET_PAIR = context.socket(zmq.PAIR)
    glbSOCKET_PAIR.set_hwm(2)
    glbSOCKET_PAIR.bind("tcp://*:%s" % glbPORT_PAIR)
    glbSENDER = ZMQChunkedPublisher(port = glbPORT_PUBSUB, topic = "raw")
    #glbSOCKET_PUBSUB = context.socket(zmq.PUB)
    #glbSOCKET_PUBSUB.set_hwm(5)
    #glbSOCKET_PUBSUB.bind("tcp://*:%s" % glbPORT_PUBSUB)
    if bPLOT:
        import pyqtgraph as pg
        pg.setConfigOptions(antialias=True)
        #plotWin=pg.PlotWindow()
        #curve=plotWin.plotItem.plot()
        global p1, p2, plotWin
        plotWin=pg.GraphicsWindow(title='acquisition')
        p1=plotWin.addPlot()
        p1.plot([0,1])
        plotWin.nextRow()
        p2=plotWin.addPlot()
        p2.plot([0,1])
        p2.setXLink(p1)


glbMONITORING=False
glbCONTINUOUS=False
glbSTREAMING=False
from numpy import random

def setHWM(newHWM): # "High Water Mark" of the publishing socket.
    global glbSOCKET_PUBSUB
    glbSOCKET_PUBSUB.set_hwm(newHWM)
    if not glbLOCAL:
        sendCmd('setHWM', dict(newHWM=newHWM))

def startMonitoring():
    """Just a loop to run continuously, while monitoring for remote commands"""
    global glbMONITORING, glbLOCAL, glbCONTINUOUS
    if not glbLOCAL:
        raise ValueError("Scope is remote")
    glbMONITORING=True
    k=0
    if glbCONTINUOUS:
        startAcquisition()
    elif glbSTREAMING:
        startStreaming()
    print("start monitoring...")
    while glbMONITORING:
        k+=1
        if k>20: #Just print a message so we know it's still working
            print(".", end='', flush=True)
            k=0
        if checkAndRecieveRemoteCommand():
            print('breaking')
            break

        print('.', end ='')
        sleep(0.2)

def initRemote():
    context = zmq.Context()
    global glbSOCKET_PAIR
    glbSOCKET_PAIR= context.socket(zmq.PAIR)
    glbSOCKET_PAIR.set_hwm(2)
    glbSOCKET_PAIR.connect("tcp://localhost:%s" % glbPORT_PAIR)
    sleep(0.2)
    #glbSOCKET_PAIR.setsockopt(zmq.HWM,1)
    subscribe(b'raw')

def subscribe(topicFilter=None):
    global glbSOCKET_PUBSUB
    glbSOCKET_PUBSUB=zmq.Context().socket(zmq.SUB)
    glbSOCKET_PUBSUB.set_hwm(5)
    glbSOCKET_PUBSUB.connect("tcp://localhost:%s" % glbPORT_PUBSUB)
    if topicFilter:
        glbSOCKET_PUBSUB.setsockopt(zmq.SUBSCRIBE, topicFilter)
    #glbSOCKET_PUBSUB.setsockopt(zmq.HWM,1)
    sleep(0.2)

def closeSockets():
    if glbLOCAL:
        glbPS.close()
        glbSENDER.close()
    glbSOCKET_PAIR.close()
    glbSOCKET_PUBSUB.close()


def setScopeParams(acqTime=None, VRange=None, 
        Ncaps=None, sampleRate=None, resolution=None):
    newD=dict(
        acqTime=acqTime,
        VRange=VRange,
        Ncaps=Ncaps,
        sampleRate=sampleRate,
        resolution=resolution
        )
    if setScopeParams.lastAcqD is not None and setScopeParams.lastAcqD==newD:
        #print("set scope params same as last time, not repeating")
        return;
    setScopeParams.lastAcqD=newD
    if glbLOCAL: # if we're using the scope directly
        setScopeParamsLocal(newD)
    else:
        setupScopeRemote(newD)
setScopeParams.lastAcqD=None

scopeDefaults=dict( 
    VRange=2.0,  # Should probably be 1.0
    Ncaps=100, 
    sampleRate=3e6, 
    acqTime=1./60.-20e-6, 
    resolution=15
    )
curAcqD=scopeDefaults.copy() 

def setScopeParamsLocal(parD):
    """
    This should do something a little different depending on whether we're streaming or in block mode
    """
    #update the non-None parameters
    curAcqD.update({k:v for k,v in parD.items() if v is not None})
    if glbSTREAMING:
        print("Not updating scope acquisition parameters while streaming")
        return;
    scpDict=dict(
            chanAParams={'coupling':'AC', 'VRange': curAcqD['VRange']},
            acqTime= curAcqD['acqTime']-20e-6,
            nCaps= curAcqD['Ncaps'],
            sampleRate= curAcqD['sampleRate'],
            resolution= curAcqD['resolution'],
            )
    glbPS.quickSetup(**scpDict)

def setupScopeRemote(paramD):
    print("setting scope parameters remotely")
    glbSOCKET_PAIR.send(b'scp ' +pickle.dumps(paramD), zmq.NOBLOCK)     
    if not glbSOCKET_PAIR.poll(10000):
        raise ValueError("No reply from server after 10 seconds")
    else:
        rep=glbSOCKET_PAIR.recv()
        st=rep.split(b' ', 1)
        if st[0]!= b'OK':
            print("err: {}".format(rep))

def checkForPublished(topicFilter=""):    
    #print("checking for published: ", end='')
    if glbSOCKET_PUBSUB.poll(10):
        print('.', end='')
        topic,msg= glbSOCKET_PUBSUB.recv().split(b' ', 1)
        return topic, pickle.loads(msg)
    else:
        print(' no')

def setRepeating(state=True):
    global glbCONTINUOUS
    if glbLOCAL:
        glbCONTINUOUS=state
    else:
        state=1 if state else 0
        sendCmd('repeat', state)
        #glbSOCKET_PAIR.send(b'repeat '+bytes(str(state), 'utf-8') )

def recvCommand():
    if glbSOCKET_PAIR.poll(10):
        st = glbSOCKET_PAIR.recv(zmq.NOBLOCK).split(b' ',1)
        cmd=st[0]
        print("... got {}".format(cmd))
        if len(st)>1:
            param=pickle.loads(st[1])
        else:
            param=None
        return cmd, param
    else:
        return None, None

def checkAndRecieveRemoteCommand():
    #print("checking for remote command: ",end="", flush=True)
    print(".")
    cmd,param=recvCommand()
    if cmd is not None:
        if cmd==b'scp': #Param is a new scope paramaters
            try:
                glbPS.waitReady()
            except OSError as e:
                if e.args[0].find("PICO_CANCELLED")>=0 or e.args[0].find("NO_SAMPLES")>=0:
                    pass
                else:
                    raise e
            
            try:
                bStreamingWasStopped = False
                if glbSTREAMING:
                    bStreamingWasStopped = True
                    stopStreaming()
                setScopeParamsLocal(param)
                print("Updated scope parameters: {}".format(param))
                if bStreamingWasStopped:
                    print("restarting streaming")
                    startStreaming(bKeepSampsPerSeg=True)
                #glbSOCKET_PAIR.send(b'OK')
                replyOk()
            except OSError as e:
                glbSOCKET_PAIR.send(bytes('err {}'.format(e), 'utf-8') )
                print("scope error: {}".format(e)) 
        elif cmd==b'raw': #Acquire some data. This needs updating
            glbPS.stop()
            acquireRaw(bPUBLISH=True, **param) #this should prob just be **param, not args
            replyOk()
        elif cmd==b'stop': # Stop acquring
            stopStreaming()
            replyOk()
        elif cmd==b'monitorStop':
            replyOk()
            return 1 # returning true kills the monitoring that is calling this function
        elif cmd==b'startStreaming':
            startStreaming(**param)
            replyOk()
        elif cmd==b'setHWM':
            setHWM(**param)
        else:
            glbSOCKET_PAIR.send(bytes('err {}'.format(cmd), 'utf-8'), zmq.NOBLOCK )
            raise ValueError("Unkown command '{}'".format(cmd))

        #self.ps.quickSetup(**D)
    #print("finished command checking", flush=True)


def startAcquisition():
    glbPS.runBlock(pretrig=0,segmentIndex=0) 



t=None
dat=None


def sendCmd(cmd, params=None):
    #print('sending: {}'.format(cmd))
    if glbLOCAL:
        raise ValueError("Shouldn't be sendind a command if it's local")
    glbSOCKET_PAIR.send(bytes(cmd, 'utf-8') +b' '+ pickle.dumps(params))
    if glbSOCKET_PAIR.poll(5000):
        rep=glbSOCKET_PAIR.recv()
        st=rep.split(b' ', 1)[0]
    else:
        raise ValueError("No reply")
    if st!=b'OK':
        raise ValueError("Got: '{}' instead of OK".format(rep))

def replyOk():
    print('replying ok')
    glbSOCKET_PAIR.send(b'OK', zmq.NOBLOCK)



def acquireRaw(bReacquire=False, bWait=True, bPUBLISH=True):
    if glbLOCAL== False:
        glbSOCKET_PAIR.send(b'raw ' + pickle.dumps(dict(bReacquire=bReacquire)))
        #print("polling")
        sendCmd('raw', dict(bReacquire=bReacquire))
        print("waiting for raw...")
        rep=glbSOCKET_PUBSUB.recv()
        topic, msg=rep.split(b' ', 1)
        if topic != b'raw':
            raise ValueError("Don't understand {}".format(str[0]))
        return pickle.loads(msg)
    if not bReacquire:
        glbPS.runBlock(pretrig=0,segmentIndex=0) 

    if bWait:
        glbPS.waitReady()
    else:
        if not glbPS.isReady():
            return None
    global t, data
    #tt=glbPS._lowLevelGetValuesTriggerTimeOffsetBulk(0,5)
    (data,nSamps,ovf)=glbPS.getDataVBulk()

    if bReacquire: #Restart the acquisition
        glbPS.runBlock(pretrig=0,segmentIndex=0) 
    
    t=np.arange(data.shape[0])*glbPS.sampleInterval*data.shape[-1]
    if bPUBLISH:
        publishRaw(data, Nds=1)
        glbSOCKET_PUBSUB.send(msg)
        #print("Publishing raw data")
    if bPLOT:
        p1.curves[0].setData(t, data.mean(axis=0))
        p2.curves[0].setData(t, np.std(util.smoothalong(data, 10,axis=1), axis=0))
    return t, data

def publishRaw(data, t=None, Nds=1):
    dt=glbPS.sampleInterval*Nds
    #chunkedPublisher.send(data=data, t=t)
    msg=b'raw '+ pickle.dumps(dict(data=data, dt=dt, t=t))
    #glbSOCKET_PAIR.send(msg)
    #glbSOCKET_PUBSUB.send(msg)
    glbSENDER.send(tL = t, datL =  data)

def startMonitorThread(*args, **kwargs):
    startMonitorThread.thread=threading.Thread(target=startMonitoring, args=args, kwargs=kwargs)
    startMonitorThread.thread.setDaemon(True)
    startMonitorThread.thread.start()

def startStreamingThread(*args, **kwargs):
    #if not glbSTREAMING:
        #startStreaming()
    def streamingRecv(*args, **kwargs):
        while glbSTREAMING:
            acquireStreamingLatest(bPUBLISH=True)
            sleep(0.1)
    startStreamingThread.thread= threading.Thread(target=streamingRecv, args=args, kwargs=kwargs)
    startStreamingThread.thread.setDaemon(True)
    startStreamingThread.thread.start()

def stopMonitoring():
    if glbLOCAL:
        global glbMONITORING
        glbMONITORING=False
    else:
        sendCmd('monitorStop')

def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
    global segmentizer, wasCalledBack
    wasCalledBack = True
    datA = bufferAMax[startIndex:sourceEnd]
    datB = bufferBMax[startIndex:sourceEnd]
    segmentizer.feedData(datA, datB)

from queueview import QueueView
from  queue import Queue

class PreprocessCallback(object):
    """
    * Downsample
    * split into segments
    """
    def __init__(self,incomingData, Npts2keep=4e6, Nds=1, NptsPerSegment=None, sampleInterval=1):
        self._incData=QueueView(incomingData)
        self.streamReady=False
        self.data=np.zeros(int(Npts2keep), dtype='i2')
        self.q=Queue()
        self.Nds=Nds
        self.curSaveInd=0
        self.NptsPerSegment=NptsPerSegment
        self.trigIndsL=[]
        self.sampleInterval=sampleInterval
        self.tLast=0
        self.totalSegments=0
        tcPars = dict (
            startT = 1000e-6,
            DT = 100e-6,
            thresh = 1000,
            N = 11,
            rate = 256, #Hz
        )
        tcStartInds = (tcPars['startT'] + np.arange(tcPars['N'])*tcPars['DT'])/sampleInterval + 1
        tcStopInds = (tcPars['startT'] + (1+np.arange(tcPars['N'])) * tcPars['DT'])/sampleInterval -1
        tcPars['t_slices'] = [slice(startInd, stopInd) for startInd, stopInd in zip(tcStartInds.astype('i4'), tcStopInds.astype('i4'))]
        self.tcPars = tcPars
        self.tracesRead = 0
        self.secondsRead = 0

    def findTrigs(self, dat, dumpN = 200, thresh = 10000):#3000):
        sm=util.smooth(dat[dumpN:],3) #smooth out noise a little
        inds=np.where(sm>thresh)[0]
        inds=inds[1:][np.diff(inds)>5] #look for a gap in the high values
        inds+=dumpN
        return inds #indices of the rising edges
    
    def readFlagsFromTrigTrace(self, trigTrace):
        trigCommFlags = sum([(trigTrace[slc].mean()> self.tcPars['thresh']) << k for k,slc in enumerate(self.tcPars['t_slices']) ])
        #trigCommFlags = [(trigTrace[slc].mean()) for k,slc in enumerate(self.tcPars['t_slices']) ]
        #print("trigCommFlags: ", trigCommFlags)
        #print(trigCommFlags)
        if trigCommFlags & 2: #indicates a PPS just came
            self.secondsRead += 1
            if self.secondsRead > 1:
                self.tcPars['rate'] = self.tracesRead/(self.secondsRead-1)
                #print(self.tcPars['rate'])
        if self.secondsRead>0:
            self.tracesRead += 1
        return trigCommFlags
        # somehow need to get sample interval and slice parameters in here. Probably in the init...
        
    def setTotalIntervalLength(self, newT):
        self.NptsPerSegment=int(newT/self.sampleInterval)

    def __call__(self, handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, parameter):
        self.streamReady=True
        noOfSamples=int(noOfSamples/self.Nds)*self.Nds #Only take a multiple of the down-sample factor
        #print("start index: {}, nSamps: {}, qvsize: {}".format(startIndex, noOfSamples, self._incData.view().shape))
        if noOfSamples:
            #print("doing things")
            self._incData.advanceEnd(noOfSamples)
            #print("qvsize: {}".format(self._incData.view().shape))
            newDat=self._incData.view()
            trigInds=self.findTrigs(newDat[1])
            if len(trigInds)>1:
                #print("trig0: {}".format(trigInds[0])) 
                if self.NptsPerSegment is None:
                    self.NptsPerSegment=int(np.mean(np.diff(trigInds)))-5
                diffs=np.diff(np.hstack([[0], trigInds]) ) - self.NptsPerSegment
                if any(abs(diffs)<0):
                    print("oh no! diffs: {}".format(diffs))
                #print("Npts per seg: {}".format(self.NptsPerSegment))
                #print("len(TrigInds) {} in {} samples".format(len(trigInds), noOfSamples))
                Nprocessed=trigInds[-1]
                self._incData.dumpStart(Nprocessed) #Dump points up until the last trigger
                tA= self.tLast + (1+np.arange(len(trigInds)))/self.tcPars['rate']#trigInds*self.sampleInterval+self.tLast 
                self.tLast=tA[-1]
                Nqueued=0
                for t,trgI in zip(tA, [0]+list(trigInds[:-1])):
                    #print(trgI, self.NptsPerSegment)
                    yTrace, trigTrace = newDat[:,trgI:trgI+self.NptsPerSegment]
                    trigCommFlags = self.readFlagsFromTrigTrace(trigTrace)
                    self.q.put_nowait((t, trigCommFlags, yTrace) )
                    Nqueued +=1
                    #self.q.put_nowait((t, yTrace) )
                self.totalSegments+=len(trigInds)
                if Nqueued!= len(trigInds):
                    errSt = f"{Nqueued} != {len(trigInds)} totalSegs: {self.totalSegments} \r\n"
                    open("failure.txt", 'a').write(errSt)
            else:
                Nprocessed=0
                
            print( f"{self.tracesRead} | {len(trigInds)} | totalSegs: {self.totalSegments} \r\n")
            Nsave=int(Nprocessed/self.Nds)

            if self.curSaveInd+Nsave< self.data.size and Nsave!=0:
                self.data[self.curSaveInd:self.curSaveInd+Nsave]=util.downsample_npts(newDat[0, :Nprocessed], self.Nds)
                self.curSaveInd+=Nsave
            #else:
            #    self.data[self.curInd:self.curInd+noOfSamples]=ps.data[1,0,startIndex:startIndex+noOfSamples]
            #self._incData.dumpStart(noOfSamples) #Dump points up until the last trigger
        else:
            print("no samples")



Nstreamed=0
def startStreaming(totalLength=None, bKeepSampsPerSeg=True):
    global glbSTREAMING
    glbSTREAMING=True
    if not glbLOCAL:
        sendCmd('startStreaming', dict(totalLength=totalLength, bKeepSampsPerSeg=bKeepSampsPerSeg))
        return
    glbPS.quickSetup(chanAParams=dict(coupling="DC", VRange=curAcqD['VRange']),
                   chanBParams=dict(coupling="DC", VRange=5.), 
                nCaps=1,
            sampleRate=3e6, acqTime=0.2,resolution=15,
            triggerParams=dict(trigSrc="B", threshold_V=1.0, 
                    direction="Rising", delay=0,enabled=False)
              )
    glbPS.Nds=3
    data=glbPS.allocateDataBuffers(channels=["A","B"],numSamples=int(8e6), downSampleMode=4)
    if bKeepSampsPerSeg and hasattr(glbPS,'cb'):
        NptsPerSeg=glbPS.cb.NptsPerSegment
    else:
        NptsPerSeg=None
    cb=PreprocessCallback(glbPS.data[:,0,:], 4e6, Nds=1, sampleInterval=glbPS.sampleInterval*glbPS.Nds, NptsPerSegment=NptsPerSeg)
    if totalLength is not None:
        cb.setTotalIntervalLength(totalLength)
    glbPS.runStreaming(bAutoStop=False, downSampleMode=4, downSampleRatio=glbPS.Nds)
    glbPS.cb=cb
    glbPS.cb.tStartStreaming=time.time()
    global Nstreamed
    Nstreamed=0


def acquireStreaming(bNoStale=False, Nmin=1):
    tStart=time.time()
    #print("acquire streaming start at: {}".format(tStart))
    finTL=[]
    finDatL=[]
    bStillStale=bNoStale
    throwOuts=0
    while len(finDatL)<Nmin:
        Nsofar=len(finDatL)
        tElapsed=(time.time()-tStart)
        if tElapsed==0:
            tElapsed=1
        print('acqStreaming got {}/{}, rate: {}'.format(Nsofar, Nmin, float(Nsofar)/tElapsed ), end='')
        out=acquireStreamingLatest()
        tL, datL, dt=out
        if bStillStale:
            for k,el in enumerate(tL):
                if el>tStart:
                    #print("throwout {} of {}".format(k, len(tL)))
                    bStillStale=False;
                    datL=datL[k:]
                    tL=tL[k:]
                    break;
            else:
                if len(tL)>0:
                    datL=[]
                    print("allStale", end='')
                    throwOuts+=1
                    #print("throwout all, most recent: {}".format(el))
                    if throwOuts>100:
                        raise Exception("Too much stale data- the acquirer is probably out of sync")
        
        finTL.extend(tL)
        finDatL.extend(datL)
    finDatL=util.pruneRagged(finDatL)
    print("acquired.", flush=True)
    return finTL, finDatL, dt
        

def acquireStreamingLatest(bPUBLISH=True):
    """ Just return the latest values from the callback
    """
    
    #print("s", end='')
    if not glbLOCAL:
        out=checkForPublished()
        if out is None:
            return [], [], None

        datD = out[1]['datD']
        t,flagsL, datL= datD['tL'], datD['flagsL'], datD['datL']
        
        #t=D['t']
        #datL=D['data']
        #dt=glbPS.sampleInterval*glbPS.Nds
        return t, datL, flagsL

    q=glbPS.cb.q
    try:
        glbPS.getStreamingLatestValues(callback=glbPS.cb)
    except OSError as e:
        if e.args[0].find("PICO_BUSY")<=0:
            print("An exception!", flush=True)
            raise e
        else:
            print("Busy...")
            
    if q.qsize():
        #print('getting from q', end='')
        retrL=[q.get() for k in range(q.qsize())]
        #print(', q emptied')


        #$print("time: {}, tLast: {}".format(time.time()-glbPS.cb.tStartStreaming, t[-1]))
        t, flagsL, datL=zip(*retrL)
        t=[el+glbPS.cb.tStartStreaming for el in t]
        #t+=glbPS.cb.tStartStreaming
    else:
        t=[]
        datL=[]
        print('-')
    if bPUBLISH and len(t)>0:
        #print('start publish... ', end='')
        #publishRaw(datL, t=t, flags= Nds=glbPS.Nds)
        glbSENDER.send(tL = t, flagsL=flagsL, datL =  datL)
        #msg=b'raw '+ pickle.dumps((datL, glbPS.sampleInterval*6))
        #glbSOCKET_PAIR.send(msg)
        #glbSOCKET_PUBSUB.send(msg)
        global Nstreamed
        Nstreamed+=len(datL)
        tElapsed=time.time()-glbPS.cb.tStartStreaming
        if tElapsed==0:
            tElapsed=1;
        lag=time.time()-t[-1]
        print("Pub: {} segments, last at {} (lag:{}), Rate:{}".format(len(datL), t[-1], lag, Nstreamed/tElapsed))
        if lag>10:
            print("lag too much, restarting stream")
            startStreaming(bKeepSampsPerSeg=True)

    dt=glbPS.sampleInterval*glbPS.Nds
    return t, datL, dt

def stopStreaming():
    if glbLOCAL:
        glbPS.stop()
        glbSTREAMING=False
    else:
        glbSTREAMING=False
        sendCmd('stop')
def close():
    stopStreaming()
    stopMonitoring()
    try:
        startStreamingThread.thread._stop()
        startMonitorThread.thread._stop()
    except:
        pass
    closeSockets()
    try:
        glbPS.close()
    except:
        pass

if __name__=="__main__":
    #import pyqtgraph as pg
    #import threading
    #from pylab import *
    #import time
    init(bRemote=False)
    startMonitorThread()
    startStreaming()
    startStreamingThread()
    #startMonitoring()
    print("initialised PS")
    #startStreaming()
    #print("Streaming started")
    #startMonitoring()
    #print("Monitoring")
    #monitorThread()

    #from time import sleep
    #while 1:
    #    sleep(1)
    
    if 0:
        tStart=time.time()
        datStart=glbPS.cb.data.copy()
        l=[]
        while time.time()-tStart<60:
            t,dat,dt=acquireStreamingLatest()
            l.append(array(dat).copy())
            sleep(0.1)

        l=[el for el in l if el.size>0]
        d=vstack(l)
        plot(d[:,100])
        #t,dat,dt=zip(*l)

        #datEnd=glbPS.cb.data.copy()
        #plot(datStart)
        #plot(datEnd)


        #monitorThread(bPostRaw=True)
        #startMonitoring(bPostRaw=True)
