

from instrument_control.pulse_patterns import makePulseTrain
#from . import FG
from box import Box
from .rigol.rigolfg_1000 import RigolFG1000
from .rigol.rigolfg_900 import RigolFG900
import numpy as np
from . import util
import inspect
import pdb

SAMPLE_RATE = 1e6
GLB_pulsePatternDefaults = { 'sampleRate': SAMPLE_RATE, "smthFact":4}
#fgD = {name: None for name in addrD}
#for name, addr in addrD.items():
#    try:
#        fgD[name] = rm.get_instrument(addr)
#    except ValueError as e:
#        print(f"Couldn't open {name} FG, because of {e}")
#

# fgs.pump.setWaveform(x)
# fgs.pump.setPattern(x)
# fgs.setPatterns()
# fgs.setWaveforms()
# fgs.updateWaveforms()
# fgs.updatePatterns()
#



def setChanNumWrapper(inst, fn, chanNum ):
    #print("calling setChanNumWrapper on {}".format(fn))
    #print(f"type: {type(fn)}, sig: {inspect.signature(fn)}")
    def wrapped( *args, **kwargs ):
        #print(f"args :{args},kwargs:{kwargs} ")
        kwargs['chanNum'] = chanNum
        #return fn( inst, *args, **kwargs )
        return fn( *args, **kwargs )
    return wrapped

class Channel():
    """Represent a single FG Channel
    
    All it does is pass on all calls to the held FG object with the channel number set.
    """
    
    def __init__(self, fg, chanNum):
        self.fg = fg
        self.chanNum = chanNum
        #def wrap_methods( cls, wrapper ):
        attr_names = filter(lambda name: not name.startswith('_'), dir(fg))
        fnD = {name:fg.__getattribute__(name) for name in attr_names if callable(fg.__getattribute__(name))   }
        fnD = {name:fn for name, fn in fnD.items() if 'chanNum' in inspect.signature(fn).parameters}
        for name, fn in fnD.items():
            self.__dict__[name]  = setChanNumWrapper(fg, fn, chanNum ) 

    #def __getattribute__(self,name):
    #    print("calling getattribute...")
    #    attr = self.fg.__getattribute__(name)
    #    if hasattr(attr, '__call__'):
    #        def newfunc(*args, **kwargs):
    #            print('before calling %s' %attr.__name__)
    #            result = attr(*args, chanNum=chanNum, **kwargs) #?
    #            print('done calling %s' %attr.__name__)
    #            return result
    #        return newfunc
    #    else:
    #        return attr
dg1000, dg900_A, dg900_B, chs = None, None, None, None

def init():
    globals dg1000, dg900_A, dg900_B, chs
    dg1000 = RigolFG1000("USB0::0x1AB1::0x0642::DG1ZA193403439::INSTR")# (DG1022)
    dg900_A = RigolFG900("USB0::0x1AB1::0x0643::DG9A210800150::INSTR") # (first DG952) By, Bz
    dg900_B = RigolFG900("USB0::0x1AB1::0x0643::DG9A210800149::INSTR")# (second DG952)
    chs = Box(
        pump = Channel(dg1000, chanNum=0),
        hardBz = Channel(dg1000, chanNum=1),
        Bx = Channel(dg900_B, chanNum=0),
        By = Channel(dg900_A, chanNum=0),
        Bz = Channel(dg900_A, chanNum=1),
    )

def setPulsePattern(chanName, seqDesc):
    #endCutPts = None
    #if chanName != "pump" and 0:
    #    endCutPts = 2
    params = GLB_pulsePatternDefaults.copy()
    params.update(seqDesc)
    t, y = makePulseTrain(**params)
    chs[chanName].setOutputWaveform(t, y)

def setPulsePatterns(patternD, othersOff=True, **kwargs):
    """
    
    patternD is a dict of channel name strings and pulse-patterns. 
    E.g. patternD = {'pump': {'startTimes': [1e-6, 100e-6], 
                            'pulseWidths':5e6,
                            'pulseHeights': [2,-2]}
                            'sapmleRate' : 1e6, 
                            'totalTime' : 4e-3
                            }
                    'hardBz':...
                    }
    """
    all_chan_names = chs.keys()
    used_chan_names = patternD.keys()
    unused_chan_names = set(all_chan_names).difference(used_chan_names)
    for chan_name, pulse_desc in patternD.items():
        setPulsePattern(chan_name, pulse_desc | kwargs)
    for chan_name in unused_chan_names:
        chs[chan_name].setOutputState(0)

def allOff():
    for chan in chs.keys():
        chan.setOutput(0)

#class PmFgController(object):
#    rateX=250000000/10
#    rateY=250000000/10
#    rateZ=125000000/100
#    pulseHeightParams=None 
#    pulseTimeParams=None
#    t=np.zeros(1)
#    VX=np.zeros(1)
#    VY=np.zeros(1)
#    VZ=np.zeros(1)
#    bXChanged=None;
#    bYChanged=None;
#    bZChanged=None;
#    bNoZ=False
#
#    def __init__(self, ps=None):
#        self.bXChanged=True;
#        self.bYChanged=True;
#        self.bZChanged=True;
#        self.bXAmpChanged=True
#        self.bYAmpChanged=True
#        self.bZAmpChanged=True
#        self.allOff()
#        self.fgs = Box(
#            dg1000 = RigolFG1000("USB0::0x1AB1::0x0642::DG1ZA193403439::INSTR"),# (DG1022)
#            dg900_A = RigolFG900("USB0::0x1AB1::0x0643::DG9A210800150::INSTR"), # (first DG952) By, Bz
#            dg900_B = RigolFG900("USB0::0x1AB1::0x0643::DG9A210800149::INSTR")# (second DG952)
#        )
#        self.chans = {
#            'pump': ['dg1000', 1], 
#            'big_Bz': ['dg1000', 2], 
#            'Bz': ['dg900_B', 1], 
#        }
#
#    
#    def setPulsePattern(self, chanName, times, widths, heights, total_t):
#        fg, chanNum = self.chans[chanName]
#        t, y = makePulseTrain(pulse_widths, pulse_heights, sample_rate, Nsamples, smthFact=8)
#        fg.setOutputWaveform(t, y, chanNum=chanNum)
#    
#    #OLD STUFF===============
#    def setWaveForms(self, t,VX, VY, VZ, bReorderOnly=False):
#        self.t=t
#        if 0:
#            if t.size!= self.t.size or ~np.all(t==self.t):
#                #print("t changed")            
#                self.bXChanged=True
#                self.bYChanged=True
#                self.bZChanged=True
#        if self.VX.size!= VX.size or ~np.all(self.VX==VX):
#            #print("Xchanged")
#            self.bXChanged=True
#            self.bXAmpChanged=True
#        if self.VY.size!= VY.size or ~np.all(self.VY==VY):
#            #print("Ychanged")
#            self.bYChanged=True
#            self.bYAmpChanged=True
#        if self.VZ.size!= VZ.size or ~np.all(self.VZ==VZ):
#            #print("Zchanged")
#            self.bZChanged=True
#            self.bZAmpChanged=True
#        
#
#        if VX.size>0 and bReorderOnly:
#            if self.bXChanged and self.VX.max()==VX.max() and self.VX.min()==VX.min():
#                self.bXAmpChanged=False
#            if self.bYChanged and self.VY.max()==VY.max() and self.VY.min()==VY.min():
#                self.bYAmpChanged=False
#            if self.bZChanged and self.VZ.max()==VZ.max() and self.VZ.min()==VZ.min():
#                self.bZAmpChanged=False
#
#
#        self.VX, self.VY, self.VZ = VX, VY, VZ
#        self.updateOutputs()
#        return
#
#    def updateOutputX(self):
#        fg=self.fg33500
#        fg.allOff()
#        fg.setRate(self.rateX, 0)
#        if all(self.VX==0):
#            fg.uploadWaveform(self.VX/max(abs(self.VX)), name='VOLX', chanNum=0);
#            fg.setLH( -0.01, 0.01, chanNum=0 )
#        else:
#            #fg.uploadWaveform(self.VY/max(abs(self.VY)));
#            fg.uploadWaveform(self.VX/max(abs(self.VX)),name='VOLX',chanNum=0);
#            if self.bXAmpChanged:
#                fg.setLH( self.VX.min(), self.VX.max(), chanNum=0 )
#        errStr=fg.getErr()
#        errVal=int(errStr.split(',')[0])
#        #if errVal!= 0 and errVal != -221:
#        if errVal:
#            raise ValueError(errStr.split(',')[1])
#        self.bXChanged=False
#        self.bXAmpChanged=False
#
#    def updateOutputY(self):
#        fg=self.fg33500
#        fg.allOff()
#        fg.setRate(self.rateY, 1)
#        if all(self.VY==0):
#            fg.uploadWaveform(self.VY,name='VOLY',chanNum=1);
#            fg.setLH( -0.01, 0.01, chanNum=1 )
#        else:
#            #fg.uploadWaveform(self.VY/max(abs(self.VY)));
#            fg.uploadWaveform(self.VY/max(abs(self.VY)),name='VOLY',chanNum=1);
#            if self.bYAmpChanged:
#                fg.setLH( self.VY.min(), self.VY.max(), chanNum=1 )
#        errStr=fg.getErr()
#        errVal=int(errStr.split(',')[0])
#        #if errVal!= 0 and errVal != -221:
#        if errVal:
#            raise ValueError(errStr.split(',')[1])
#        self.bYChanged=False
#        self.bYAmpChanged=False
#
#    def updateOutputZ(self): 
#        fg=self.fgK
#        if self.bNoZ:
#            return
#
#        fg.setOutputState(False)
#        #fg.setRate(self.rateZ)
#        #self.VZ=self.VZ[:32000]
#        fgZPeriod=1./self.rateZ*self.VZ.size
#        print("fgZPeriod: {}".format(fgZPeriod))
#        fg.setPeriod(fgZPeriod)
#        if 1:
#            if abs(max(self.VZ)-min(self.VZ))<0.05:
#                fg.uploadWaveform(self.VZ*0, scl=False);
#                fg.setOffset(0)
#                fg.setAmp( 0.05 )
#            else:
#                fg.uploadWaveform(self.VZ/max(abs(self.VZ)),scl=True);
#                if self.bZAmpChanged:
#                    fg.setLH( self.VZ.min(), self.VZ.max() )
#
#        errStr=fg.getErr()
#        print(errStr)
#        errVal=int(errStr.split(',')[0])
#        #if errVal!= 0 and errVal != -221:
#        if errVal!=0:
#            raise ValueError(errStr.split(',')[1])
#        self.bZChanged=False
#        self.bZAmpChanged=False
#
#    def updateOutputs(self):
#        if self.bXChanged:
#            self.updateOutputX()
#        if self.bYChanged:
#            self.updateOutputY()
#        if self.bZChanged:
#            self.updateOutputZ()
#
#    def allOn(self):
#        #self.fgP.allOn()
#        self.fgK.allOn()
#        self.fg33500.allOn()
#        #self.fgK.allOn()
#
#    def allOff(self):
#        #self.fgP.allOff()
#        self.fgK.allOff()
#        self.fg33500.allOff()
#
#
#    def setWaveformsResponseCorrected(t, Vx,Vy, Vz):
#        """Should take the desired output waveforms (Vx,Vy, Vz) and change the actual output according to the known response.
#        """
#        raise NotImplementedError
#

samplePatterns = Box({'pump': {'startTimes': [20e-6, 2021e-6], 
                        'pulseWidths':300e-6,
                        'pulseHeights': [10,10],
                        },
            "hardBz" :{"startTimes": [0, 2000e-6],
                        "pulseWidths": 200e-6,
                        "pulseHeights": [1, -1]},
            })

if __name__=="__main__":
    from numpy import pi, sin, cos 
    t=np.arange(0,15e-3, 1e-6)
    y=np.sin(2*pi*t*1000)
    init()
    setPulsePatterns(patternD, totalTime = 4000e-6)
    #pfg=PmFgController()
    #pfg.setRates(1e6,1e6,1e6)
    #pfg.setWaveForms(y,y,y)