""" Top level hardware control. 
Functions to set them all up given experiment parameters, as well as stop() and clear() to stop and clear the experiments.

"""
import time
import pdb
from tkinter import W
from box import Box
from box import Box as B
import numpy as np
import shared_parameters
from time import sleep
import os
from copy import deepcopy
from pulse_patterns import generateTriggerWaveforms, makePulseTrain
from functools import lru_cache
import fgs_mod as fgs
import teensy_wiggler
import tldevice
from async_components import ZMQSubReadChunked

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

def init_comms(bForce = False):
    if hasattr(init_comms, "alreadyRun") and not bForce:
        return
    global d
    print("set up DC fields")
    #d.coils.setModFreqs(3,4,5)
    #print("trigTask inited")
    d.oven = tldevice.Device("COM4")
    d.coils = tldevice.Device("COM6")
    #acq.init(bRemote=True)
    #acq.subscribe(b'raw')
    fgs.init()
    d.fgs = fgs
    init_comms.alreadyRun = True

#def retrievePresentOutputs():
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
    d.coils.coil.x.current(Bx)
    d.coils.coil.y.current(By)
    d.coils.coil.z.current(Bz)

@lru_cache(1)
def setupOven(setpoint, pid_params=None):
    d.oven.therm.pid.setpoint(setpoint)

# precision coil current drivers
@lru_cache(1) 
def setupPrecisionModulations( Bx, By, Bz ): 
    """
    - Each is a tuple: (amp, frequency)
    """
    mod_paramsL = [(ax,tup) for ax, tup in zip('xyz', [Bx,By,Bz]) if tup is not None ]
    for ax, par in mod_paramsL:
        coil = d.coils.coil.__dict__[ax]
        coil.modulation.amplitude(par.amp)
        coil.modulation.frequency(par.freq)
    
# Synced to rep rate- i.e. Teensy driven stuff
@lru_cache(1)
def setupSyncedModulations(Bx=None, Bz=None, Bx_1 = None, Bz_1 = None, pump_Phi=None, pump_Theta=None, bAllOff=False ): 
    """
    - mods is a list of tuples: (ax, amp, period)
    """
    modL = []
    if pump_Phi:
        modL.append((0, pump_Phi.amp, pump_Phi.period_cycles))
    if pump_Theta:
        modL.append((1, pump_Theta.amp, pump_Theta.period_cycles))

    if Bz:
        modL.append(["dac0", Bz.amp, Bz.period_cycles])
    if Bz_1:
        modL.append(["dac0_1", Bz_1.amp, Bz_1.period_cycles])
    if Bx:
        modL.append(["dac1", Bx.amp, Bx.period_cycles])
    if Bx_1:
        modL.append(["dac1_1", Bx_1.amp, Bx_1.period_cycles])
    
    print(modL)
    for mod_params in modL:
        d.wiggler.setMod(*mod_params)
    if bAllOff:
        d.wiggler.modOff()
    else:
        d.wiggler.modOn()
    

def setupFGs(patternD, tTotal):
    d.fgs.setPulsePatterns(patternD, tTotal)


def setupExperiment():
    _setupExperiment(glbP.P)

def _setupExperiment(params):
    params=deepcopy(params)
    params=Box(params, frozen_box=True)
    #setupOven(**params.oven)
    setupDCFields(**params.biasFields)
    setupFGs(**params.pulses)
    setupSyncedModulations(**params.modsSynced)
    setupPrecisionModulations(**params.modsPrec)
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
        if( abs(T_now - d.oven.pid.setpoint()) <.5  ):
            stable_time += 1
        else:
            stable_time = 0
        if stable_time > 20:
            print("calling it stable")
            break
        sleep(1)

def doPumpAlign():
    setupFGs(**pumpAlign)

def recordRaw(name, Nsamples, metaD = {}, baseDir = "recorded", tStart = -1):
    if tStart == 0:
        tStart = time.time()
    
    if not name.endswith(".npy"):
        name = name + '.npy'
    fpath = os.path.join(baseDir, name)
    with open(fpath[:-4] + '_params.yaml', 'w') as file:
        file.write( glbP.P.to_yaml() )
    #Save signature
    np.savez(fpath[:-4] + "_signatures.npz", **glbP.loadArray("signatures"))

    if metaD is not None:
        with open(fpath[:-4] + '_meta.yaml', 'w') as file:
            file.write( Box(metaD).to_yaml() )
    IN_PORT = 5560
    reader = ZMQSubReadChunked(port= IN_PORT, topic= 'raw')
    Nwritten = 0
    from npy_append_array import NpyAppendArray
    with NpyAppendArray(fpath) as dat_arr, NpyAppendArray(fpath[:-4]+'_t.npy') as t_arr:
        while 1: # wait until tStart
            retrieved = reader.retrieve()
            if retrieved:
                tL = retrieved['data']['tL']
                if tL[-1] >=tStart:
                    ind = np.searchsorted(tL, tStart)
                    datL = np.array(retrieved['data']['datL'][ind:])
                    tL = np.array(tL[ind:])
                    break
        while 1:
            dat_arr.append(datL)
            t_arr.append(tL)
            Nwritten += len(tL)
            print(f'Written: {Nwritten} | ', end='')
            if Nwritten > Nsamples:
                break
            while 1: 
                retrieved= reader.retrieve(Nmin=1)
                if retrieved:
                    break
            data = retrieved['data']
            tL = np.array(data['tL'])
            datL = np.array(data['datL'])
    return t_arr, dat_arr
pumpAlign = Box({"patternD": {'pump': {'startTs': [00e-6, 2000e-6], 
                        'widths':400e-6,
                        'heights': [7,4],
                        },
            "bigBy" :{"startTs": [000e-6,450e-6, 2000e-6, 2450e-6],
                        "widths": [200e-6, 4e-6, 200e-6, 4e-6],
                        "heights": [.5, -.4, -.5, .4]},
            "Bz" :{"startTs": [1200e-6, 3200e-6],
                        "widths": 500e-6,
                        "heights": [1., -1.]},
            }, 
            "tTotal" : 4000e-6,
            })
if __name__=="__main__":
    from importlib import reload
    from IPython.core.interactiveshell import InteractiveShell
    sh = InteractiveShell()
    mgc= sh.run_line_magic#get_ipython().magic
    mgc("load_ext", "autoreload")
    mgc("autoreload", "2")
    init_comms()
    #setupExperiment()
    #recordRaw("test7", Nsamples = 200, metaD = {"width": 27}, tStart=0)
    if 0:
        for length in np.arange(10e-6, 100e-6, 10e-6):
            setupTweaked({"pulses.patternD.bigBy.widths": length})
            recordRaw("by_width",Nsamples = 200, metaD = {"width": length})

    if 0:
        P.pulses.patternD.bigBy | B(width = 5, heights=7)
        changeParam(P | {} )
    #changeParams({'pulses.tPumpStart':10e-6})
    #tweakParams({'fields.Bz': 0.01, 'fields.By': 0.1})
