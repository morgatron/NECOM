""" Top level hardware control. 
Functions to set them all up given experiment parameters, as well as stop() and clear() to stop and clear the experiments.

"""
import pdb
from box import Box
import util
import numpy as np
from .. import shared_parameters
from time import sleep
import os
from copy import deepcopy
from pulse_patterns import generateTriggerWaveforms, makePulseTrain
from rigol.rigolfg import RigolFG
from functools import lru_cache
#import fgAgilent
#from coils import Coils
#from oven import Oven




#Coils = util.DummyObj("Coils")

d = Box( #devices
    coils=None,
    pumpAndBzFG=None,
    oven = None,
    )

glbP=shared_parameters.SharedParams()
def retrievePresentOutputs():
    outputs=Box()
    outputs.fields=Box(
                    vx=d.coils.x.field(), 
                    vy=d.coils.y.field(),
                    vz=d.coils.z.field(),
                    )
    outputs.fg=Box(t=d.fg.t,
                    vx=d.fgCont.VX, 
                    vy=d.fgCont.VY,
                    vz=d.fgCont.VZ,
                    )
    return outputs

def init_comms():
    global d
    print("set up DC fields")
    d.coils = tldevice("COMX")
    #d.coils.setModFreqs(3,4,5)
    #print("trigTask inited")
    addr="USB0::0x1AB1::0x0643::DG9A210800150::INSTR" # Replace with the actual address
    d.pumpAndBzFG= RigolFG(addr)

    import tldevice
    d.oven = tldevice("COMX")
    #acq.init(bRemote=True)
    #acq.subscribe(b'raw')
    setupExperiment()
    print("Experiment setup")


@lru_cache(1)
def setupDCFields(Bx, By, Bz):
    d.coils.setFields(Bx, By, Bz)

@lru_cache(1)
def setupOven(set_temp, pid_params=None):
    d.oven.pid.setpoint(set_temp)


@lru_cache(1)
def setupPumpAndBzPulses(tTotal, tPumpStart, tPumpWidth, tMagStart, tMagWidth):
    sampRate = 1000000# ??
    #sampRate = pulseFG.sampRate 
    
    d.pumpAndBzFG.setRate(sampRate)

    pump_wvfm = makePulseTrain([tPumpStart, tTotal+tPumpStart],  pulseTimes = 2*[tPumpWidth], pulseHeights=2*[2.0], sampleRate=sampRate, Nsamples = 2*sampRate *(tTotal+5e-6) )
    d.pumpAndBzFG.uploadWaveform(pump_wvfm, chanNum = 0)
    d.pumpAndBzFG.setTriggerMode("int", chanNum=0)

    Bz_wvfm = makePulseTrain([tMagStart, tTotal+tMagStart],  pulseTimes = 2*[tMagWidth], pulseHeights=[1,-1], sampleRate=sampRate, Nsamples = 2*sampRate *(tTotal+5e-6) )
    d.pumpAndBzFG.uploadWaveform(Bz_wvfm, chanNum = 1)
    d.pumpAndBzFG.setTriggerMode("int", chanNum=1)
    #Set sync_out on



def setupExperiment():
    _setupExperiment(glbP.p)

def _setupExperiment(params):
    params=deepcopy(params)
    setupOven(**params.oven)
    setupPumpAndBzPulses(**params.pump, **params.Bz_pulse)
    setupDCFields(**params.dc_fields)

    #acq.setScopeParams(acqTime=params.totalTime)
    sleep(0.2)

def changeParams(bPermanent=False, **kwargs):
    if bPermanent:
        glbP.change(**kwargs)
        setupExperiment()
    else:
        _setupExperiment(glbP.getChanged(**kwargs))

def tweakParams(bPermanent=False, frac=1, **kwargs):
    if bPermanent:
        glbP.tweak(kwargs,frac=frac)
        setupExperiment()
    else:
        _setupExperiment(glbP.getTweaked(frac=frac,**kwargs))
    print("tweaks: {}".format(kwargs))

def waitForStableTemp():
    from time import sleep
    stable_time = 0
    while 1:
        T_now = d.oven.T()
        print(T_now)
        if( abs(T_now - d.oven.pid.setpoint()) <1  ):
            stable_time += 1
        else:
            stable_time = 0
        if stable_time > 30:
            print("calling it stable")
            break
        sleep(1)


## STUFF BELOW HERE DOESN"T BELONG--------------------------------------------
#lasParamNames=[]#'prbI', 'prbT', 'pumpTime']
#fieldParamNames=['Bx', 'By', 'Bz']
#pulseParamNames1=['piX', 'piY', 'mPiX', 'mPiY',]
#pulseParamNames2=['piXOff', 'mPiXOff', 'piYOff', 'mPiYOff',
#    'pi2XOff', 'pi2YOff']#,'pi2RelWidth']
#
#def locAcquire(Naves=1, Nds=1, ):
#    numAttempts=1
#    while 1:
#        numAttempts+=1
#        sleep(0.2)
#        _,out,dt=acq.acquireStreaming(Nmin=Naves, bNoStale=True)#Raw()
#        #out=MT.smoothalong(out,Nds, axis=-1)[...,::Nds]
#        out=util.downsample_npts(out,Nds, axis=-1)
#        #outL.append(out)
#        #out=np.mean(out, axis=0)
#    t=np.arange(out.shape[-1])*dt*Nds
#    return t, out
#
#def getZeroLevel(Naves=50, par=None):
#    """Just turn off the pump for a little while and measure the level
#    """
#    if par is None:
#        par=glbP.p
#    oldPumpTime=par.pulseTiming.pumpTime
#    par.pulseTiming.pumpTime=0
#    _setupExperiment(par)#WithChanged(tau=4.1e-3, pulseSeqDesc=testSeq)
#    t,dat=locAcquire(Naves)
#    v0=dat.mean()
#    par.pulseTiming.pumpTime=oldPumpTime
#    _setupExperiment(par)#WithChanged(tau=4.1e-3, pulseSeqDesc=testSeq)
#    return v0
#
#lasDevDict=dict(
#    #prbI=1,
#    #prbT=0.5,
#    #pumpTime=-5e-6
#    )
#
#import dill as pickle
#def saveDataSet(filename, t, dat, sensorsD=None, grad=None, params=None, notes=''):
#    if grad is None:
#        grad=glbP.loadCal()
#    if params is None:
#        params=glbP.p
#    basePath=datDir+filename
#    infoD={
#            'params':params,
#            'notes':notes,
#        }
#
#    baseDir=os.path.dirname(basePath)
#    if not os.path.exists(baseDir):
#        os.makedirs(baseDir)
#    pickle.dump(infoD, open(basePath+'_pars.pkl', 'wb'), protocol=2)
#    np.savez(basePath+'_t.npz', t)
#    addSaved(filename, dat, sensorsD=sensorsD)
#    #np.savez_compressed(basePath+'_dat000.npz', dat)
#    np.savez(basePath+'_grad.npz', **grad)
#    return
#
#def addSaved(filename, dat, sensorsD=None):
#    basePath=datDir+filename
#    if not os.path.exists(basePath+'_pars.pkl'):
#        raise ValueError("No base data set")
#    for k in range(100): 
#        fname=basePath+'_dat{:03d}.npz'.format(k)
#        if not os.path.exists(fname):
#            np.savez(fname, data=dat)
#            if sensorsD is not None:
#                fnameSensors=basePath+'_sensors{:03d}.npz'.format(k)
#                np.savez(fnameSensors, **sensorsD)
#            break
#    else:
#        raise ValueError("Can't find a free spot for the file, last tried was: {}".format(fname))
#    return
#    
#
#
#vSclFact=1.
#p0= Box(
#    pulses=Box(
#        tPumpStart= 10e-6,
#        tMagStart = 00e-6
#        tMagWidth = 10e-6
#        tPumpWidth = 10e-6,
#        tTot=1700e-6,
#    ),
#    fields=Box(
#        Bx=1.00,
#        By=-0.0,
#        Bz=-0.00,
#    ),
#    modulations = Box(
#        Bx = 0.02,
#        BxFreq = 7,
#        #...
#        #
#        #PTheta = (0.001, 0.1),
#        #PPhi = (0.001, 0.15)
#    )
#    totalTime=1./62, # Period of mains AC
#)

if __name__=="__main__":
    from importlib import reload
    mgc=get_ipython().magic
    mgc(u"%load_ext autoreload")
    mgc(u"%autoreload 2")
    mgc(u"%matplotlib qt")
    #glbP.p=p0
    init_comms()
    setupExperiment()
    changeParams({'pulses.tPumpStart':10e-6})
    tweakParams({'fields.Bz': 0.01, 'fields.By': 0.1})
