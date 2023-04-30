

from tkinter import W
from instrument_control.pulse_patterns import makePulseTrain
#from . import FG
from box import Box
try:
    from .rigol.rigolfg_1000 import RigolFG1000
    from .rigol.rigolfg_900 import RigolFG900
    from . import util
except ImportError:
    from rigol.rigolfg_1000 import RigolFG1000
    from rigol.rigolfg_900 import RigolFG900
    import util

import numpy as np
import inspect
import pdb

B_PLOT = True
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
dpm = None

def setupTriggers():
    for ch in chs.values():
        ch.setTriggerMode("ext")
        ch.setBurstMode("TRIG")
    chs.pump.setBurstMode("TRIG")
    chs.pump.setTriggerMode("int")
def init(bMockHardware = False):
    global dg1000, dg900_A, dg900_B, chs, dpm
    if bMockHardware:
        from unittest.mock import Mock
        dg1000 = Mock(RigolFG1000, name = "dg1000")
        dg900_A = Mock(RigolFG900, name = "dg900_A")
        dg900_B = Mock(RigolFG900, name = "dg900_B")
        chs = Box(
            pump = Mock(name = "pump"),#")(dg1000, chanNum=0),
            bigBy = Mock(name = "bigBy"),#")(dg1000, chanNum=0),
            Bx = Mock(name = "Bx"),#")(dg1000, chanNum=0),
            By = Mock(name = "By"),#")(dg1000, chanNum=0),
            Bz= Mock(name = "Bz"),#")(dg1000, chanNum=0),
        )

    else:
        dg1000 = RigolFG1000("USB0::0x1AB1::0x0642::DG1ZA193403439::INSTR")# (DG1022)
        dg900_A = RigolFG900("USB0::0x1AB1::0x0643::DG9A210800150::INSTR") # (first DG952) By, Bz
        dg900_B = RigolFG900("USB0::0x1AB1::0x0643::DG9A210800149::INSTR")# (second DG952)
        chs = Box(
            pump = Channel(dg1000, chanNum=0),
            bigBy = Channel(dg1000, chanNum=1),
            Bx = Channel(dg900_B, chanNum=0),
            By = Channel(dg900_A, chanNum=0),
            Bz = Channel(dg900_A, chanNum=1),
        )
    setupTriggers()


        #sendTimer = RepeatTimer(0.1, lambda: sender.send(**generate_dict_data()) )
        #sendTimer.start()

    if B_PLOT:
        from DPM import DockPlotManager 
        dpm = DockPlotManager("fg_settings")

def setPulsePattern(chanName, seqDesc):
    #endCutPts = None
    #if chanName != "pump" and 0:
    #    endCutPts = 2

    params = GLB_pulsePatternDefaults.copy()
    params.update(seqDesc)
    #if chanName == "pump":
        #params['tTotal']+=1e-6
    t, y = makePulseTrain(**params)
    chs[chanName].setOutputWaveform(t, y)

    if B_PLOT:
        dpm.addData(chanName, {"x":t, "y": y})

def setPulsePatterns(patternD, tTotal, othersOff=True, **kwargs):
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
    for chan_name, pulse_desc in patternD.items():
        setPulsePattern(chan_name, Box(dict(pulse_desc) | kwargs | {"tTotal":tTotal-7e-6}, frozen_box=True))

    # Turn off all unused channels
    unused_chan_names = set(all_chan_names).difference(used_chan_names)
    for chan_name in unused_chan_names:
        chs[chan_name].setOutputState(0)
    chs.pump.setBurstPeriod(tTotal)
    setupTriggers()

def allOff():
    for chan in chs.keys():
        chan.setOutput(0)

#

pumpAlign = Box({"patternD": {'pump': {'startTs': [00e-6, 2000e-6], 
                        'widths':400e-6,
                        'heights': [6,4],
                        },
            "bigBy" :{"startTs": [200e-6, 2200e-6],
                        "widths": 400e-6,
                        "heights": [.5, -.5]},
            "Bz" :{"startTs": [1000e-6, 3000e-6],
                        "widths": 500e-6,
                        "heights": [1., -1.]},
            }, 
            "tTotal" : 4000e-6,
            })
samplePatterns = Box({'pump': {'startTs': [20e-6, 2021e-6], 
                        'widths':300e-6,
                        'heights': [10,10],
                        },
            "bigBy" :{"startTs": [0, 2000e-6],
                        "widths": 200e-6,
                        "heights": [1, -1]},
            "Bx" :{"startTs": [500e-6, 2500e-6],
                        "widths": 100e-6,
                        "heights": [1, -1]},
            })

base= Box({"patternD": {'pump': {'startTs': [1e-6, 2001e-6], 
                        'widths':250e-6,
                        'heights': [10,10],
                        },
            "bigBy" :{"startTs": [0, 2000e-6],
                        "widths": 50e-6,
                        "heights": [1, -1]},
            }, 
            "tTotal" : 4000e-6,
            })
if __name__=="__main__":
    from numpy import pi, sin, cos 
    t=np.arange(0,15e-3, 1e-6)
    y=np.sin(2*pi*t*1000)
    init()
    #setPulsePatterns(samplePatterns, tTotal = 4000e-6)
    #setPulsePatterns(**pumpAlign)
    setPulsePatterns(**pumpAlign)
    #pfg=PmFgController()
    #pfg.setRates(1e6,1e6,1e6)
    #pfg.setWaveForms(y,y,y)
