""" Top level hardware control. 
Functions to set them all up given experiment parameters, as well as stop() and clear() to stop and clear the experiments.

"""
import pdb
from box import Box
from box import Box as B
import util
import numpy as np
import shared_parameters
from time import sleep
import os
from copy import deepcopy
from pulse_patterns import generateTriggerWaveforms, makePulseTrain
from functools import lru_cache
import fgs_mod as fgs
import teensy_wiggler

glbP = shared_parameters.SharedParams("NECOM")
#import fgAgilent
#from coils import Coils
#from oven import Oven

GLB_sampRate = int(1e6)



#Coils = util.DummyObj("Coils")

d = Box( #devices
    coils=None,
    oven = None,
    fgs =None,
    wiggler = teensy_wiggler,
    )

def init_comms():
    global d
    print("set up DC fields")
    d.coils = tldevice("COMX")
    #d.coils.setModFreqs(3,4,5)
    #print("trigTask inited")

    import tldevice
    d.oven = tldevice("COMX")
    #acq.init(bRemote=True)
    #acq.subscribe(b'raw')
    setupExperiment()
    print("Experiment setup")

    fgs.init()
    d.fgs = fgs


def retrievePresentOutputs():
    #outputs=Box()
    #outputs.fields=Box(
    #                vx=d.coils.x.field(), 
    #                vy=d.coils.y.field(),
    #                vz=d.coils.z.field(),
    #                )
    #outputs.fg=Box(t=d.fg.t,
    #                vx=d.fgCont.VX, 
    #                vy=d.fgCont.VY,
    #                vz=d.fgCont.VZ,
    #                )
    #pass


# lru_cache is so that if the function is called identically twice in a row, 
# it only operates once

# Set precision magnetic fields
@lru_cache(1)
def setupDCFields(Bx, By, Bz):
    d.coils.setFields(Bx, By, Bz)

@lru_cache(1)
def setupOven(set_temp, pid_params=None):
    d.oven.pid.setpoint(set_temp)

# precision coil current drivers
@lru_cache(1) 
def setupPrecisionModulations( Bx, By, Bz ): 
    """
    - Each is a tuple: (amp, frequency)
    """
    mod_paramsL = [(ax,tup) for ax, tup in zip('xyz', [Bx,By,Bz]) if tup is not None ]
    for ax, [amp, freq] in mod_paramsL):
        d.coils[ax].field.modulation.amp(amp)
        d.coils[ax].field.modulation.freq(freq)
    
# Synced to rep rate- i.e. Teensy driven stuff
@lru_cache(1)
def setupSyncedModulations(Bx=None, Bz=None, pump_Phi=None, pump_Theta=None, bAllOff=False ): 
    """
    - mods is a list of tuples: (ax, amp, period)
    """
    modL = []
    if pump_Phi:
        modL.append((0, pump_Phi.amp, pump_Phi.period_cycles))
    if pump_Theta:
        modL.append((1, pump_Theta.amp, pump_Theta.period_cycles))

    if Bx:
        modL.append(["DAC0", *Bx])
    if Bz:
        modL.append(["DAC1", *Bz])
    

    for mod_params in modL:
        d.wiggler.setMod(*mod_params)
    if bAllOff:
        d.wiggler.modOff()
    else:
        d.wiggler.modOn()
    

def setupMagPulses(tTotal, BxParams, ByParams, BzParams, period_cycles=2):

@lru_cache(1)
def setupPumpPulses(tTotal, pulseParams, Nreps = 2):

    t0 = pulseParams.t0
    wvfm = makePulseTrain( t0 + tTotal*np.arange(Nreps),  
        widths= Nreps*[width], heights=Nreps*[2.0], sampleRate=GLB_sampRate, Nsamples = 2*sampRate *(tTotal+5e-6) )
    d.fg_chs.pump.uploadWaveform( wvfm )
    d.fg_chs.pump.setTriggerMode("int")


@lru_cache(1)
def setupBigByPulses(tTotal, pulseParams, Nreps = 1):
    t0, amp, width = pulseParams.t0, pulseParams.amp, pulseParams.width
    
    wvfm = makePulseTrain( startTs =t0 + tTotal*np.arange(2),  
        pulseTimes = Nreps*[width, width], heights=[amp, -amp], 
        sampleRate=GLB_sampRate, tTotal = tTotal*2 )
    d.fgs.bigBy.uploadWaveform( wvfm )
    d.fgs.bigBy.setTriggerMode("ext")

def setupByPulses(tTotal, pulseParams, Nreps = 1):

@lru_cache(1)
def setupPumpAndNucPulses(tTotal, tPumpStart, tPumpWidth, tMagStart, tMagWidth):

    pump_wvfm = makePulseTrain([tPumpStart, tTotal+tPumpStart],  pulseTimes = 2*[tPumpWidth], heights=2*[2.0], sampleRate=sampRate, Nsamples = 2*sampRate *(tTotal+5e-6) )
    d.fg_chs.pump.uploadWaveform(pump_wvfm, chanNum = 0)
    d.fg_chs.pump.setTriggerMode("int", chanNum=0)

    Bz_wvfm = makePulseTrain([tMagStart, tTotal+tMagStart],  pulseTimes = 2*[tMagWidth], heights=[1,-1], sampleRate=sampRate, Nsamples = 2*sampRate *(tTotal+5e-6) )
    d.pumpAndBzFG.uploadWaveform(Bz_wvfm, chanNum = 1)
    d.pumpAndBzFG.setTriggerMode("int", chanNum=1)
    #Set sync_out on



def setupExperiment():
    _setupExperiment(glbP.p)

def _setupExperiment(params):
    params=deepcopy(params)
    setupOven(**params.oven)
    setupPumpAndNucPulses(**params.pump, **params.Bz_pulse)
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

def setupPulsing(tTot, pump= None, bigBy=None, Bx = None, By=None, Bz = None):
    if pump is not None:
        setupPumpPulses(tTot, pump)
    if bigBy is not None:
        setupBigByPulses(tTot, bigBy)
    
    setupMagPulses(Bx, By, Bz)

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
p0= B(
    pulses=B(
        pump = B( 
           t0= 10e-6,
           tWidth = 100e-6,
           amp = 10.0,
        ),
        bigBy = B(
            t0 = 0e-6,
            tWidth = 50e-6,
            amp = 1.0,
        ),
        By = B(
            t0 = 5e-6,
            tWidth = 5e-6,
            amp = 2.0,
        ),
        tTot=1700e-6,
    ),
    biasFields=Box(
        Bx=1.00,
        By=-0.0,
        Bz=-0.00,
    ),
    modsSynced = B(
        Bx = B( amp= 100, period_cycles = 54),
        By = B( amp= 100, period_cycles = 32),
        #...
        pump_Theta = B(amp = 100, period_cycles = 23),
        pump_Phi = B(amp = 100, period_cycles = 37),
    ),
    modsPrec = B(
        Bx = B(amp= 0.02, freq= 20.872e-3),
        By = B(amp = 0.02, freq= 15.1e-3),
        Bz = B(amp = 0.02, freq= 23.1e-3),
    ),
)

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
