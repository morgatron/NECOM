"""
# pmANU relevant parameters: 

"""
import util
import numpy as np


##### TRIGGER PATTERNS:
from functools import partial
def makePulseTrain(startTimes, pulseWidths, pulseHeights, sampleRate, Nsamples, pulseFunc=partial(util.tophat, bDigitizeFix=True), smthFact=4):
    """Takes a list of pulse times, widths and heights and returns a digitized waveform with those pulses represented.

    """
    Nsamples=smthFacte*int(Nsamples)
    Npulses=len(startTimes)
    if not hasattr(pulseWidths, "__iter__"):
        pulseWidths=[pulseWidths]*Npulses
    if not hasattr(pulseHeights, "__iter__"):
        pulseHeights=[pulseHeights]*Npulses
    if not (len(startTimes)==len(pulseWidths)==len(pulseHeights) ):
        raise ValueError("All sequences should be the same length, OR scalars")
    #t=np.linspace(0,tSeqTotal,tSeqTotal*sampleRate)*1.0;
    t=np.arange(Nsamples, dtype='f8')/sampleRate
    y=np.zeros(Nsamples, dtype='f8')
    for startT, tWidth, height in zip(startTimes, pulseWidths, pulseHeights):
        #y+=np.where( (t>startT) & (t<startT+tWidth), height, 0.)
        plsShape=pulseFunc(t, tWidth, startT+tWidth/2., bDigitizeFix=True)*height
        #print("area of pls: {:.5f}".format(np.sum(plsShape)))
        y+=plsShape
    y= utils.smooth(y, window_len=smthFact)[int(smthFact/2)::smthFact]
    return t, y

def generateTriggerWaveforms(pulseTimingParams, pulseSeqDesc, Npts, sampleRate):
    """ Essentially a specialisation of makePulseTrain- just 
    """
    N=len(pulseSeqDesc)
    pt=pulseTimingParams
    pmpPulseWidth=pt.pumpTime

    tDeltas=[pt.tShort if seq[1].startswith('s') else pt.tLong for seq in pulseSeqDesc ] #+ [pt.depolLength]
    allPumpTimes=np.cumsum([0.0]+tDeltas[:-1])
    allPumpTimes+=1.0/sampleRate
    t, pumpTrigWvfm = makePulseTrain(allPumpTimes, pmpPulseWidth, 1., Nsamples=Npts, sampleRate=sampleRate)
    initPulseWidth=pmpPulseWidth
    initTrigWvfm=util.tophat(t, initPulseWidth, initPulseWidth/2+ 1./sampleRate )
    return t, pumpTrigWvfm, initTrigWvfm

def make_Bz_measurement_pattern(sampleRate, Ntau, tau, to_skip, pulse_width=5e-6, pulse_height=3.0 ):
    """ Pulse pattern to measure Bz field seen by alkali, by giving them a little side-ways kick once in a while.
    """
    startTimes = tau*np.arange(0,Ntau)
    pulseWidths = Ntau*[pulse_width]
    pulseHeights = [2*(0.5-k%2)*pulse_height  for k in range(Ntau)]
    tx,y1=makePulseTrain(startTimes=startTimes, 
                            pulseWidths=pulseWidths, 
                            pulseHeights= pulseHeights,
                            sampleRate=sampleRate,
                            Nsamples=,
                            )
  
    return tx, y1

def make_pump_alignment_optimsation_pattern(sampleRate, Nsamples, tau, to_skip, pulse_wdith, pulse_height ):
    """ Pattern to quickly optimise pumping of alkali.

    Strategy is to pin alkali off axis briefly to see how big the signal is.
    """
    N_pulses =
    startTimes = f(tau, to_skip)
    pulseWidths = Npulses*[pulse_width]
    pulseHeights = []
    tx,y1=makePulseTrain(startTimes=startTimes, 
                            pulseWidths=pulseWidths, 
                            pulseHeights= pulseHeights[:,0],
                            sampleRate=,
                            Nsamples=,
                            )
  
    return tx, y1, y2,y3