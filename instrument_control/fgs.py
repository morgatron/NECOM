
import visa

from instrument_control.pulse_patterns import makePulseTrain
from . import FG
from box import Box
from rigol.rigolfg_1000 import RigolFG1000
from rigol.rigolfg_900 import RigolFG900
import numpy as np
import util


#fgD = {name: None for name in addrD}
#for name, addr in addrD.items():
#    try:
#        fgD[name] = rm.get_instrument(addr)
#    except ValueError as e:
#        print(f"Couldn't open {name} FG, because of {e}")
#

class DummyFG(FG.FG):
    def connect(self):
        pass;
    def __init__(self):
        pass;
    def close(self):
        pass;
    def setTriggerDelay(self, delay):
        pass;
    def setTriggerMode(self, mode):
        pass;
    def setOutputWaveForm(self, t, x, chNum):
        pass;
    def setOutputState(self, bOn, chNum=0):
        pass;
    def configureHandle(self):
        pass;
    def uploadWaveform(self, wvfm, name="VOLATILE"):
        pass;

class PmFgController(object):
    rateX=250000000/10
    rateY=250000000/10
    rateZ=125000000/100
    pulseHeightParams=None 
    pulseTimeParams=None
    t=np.zeros(1)
    VX=np.zeros(1)
    VY=np.zeros(1)
    VZ=np.zeros(1)
    bXChanged=None;
    bYChanged=None;
    bZChanged=None;
    bNoZ=False

    def __init__(self, ps=None):
        self.bXChanged=True;
        self.bYChanged=True;
        self.bZChanged=True;
        self.bXAmpChanged=True
        self.bYAmpChanged=True
        self.bZAmpChanged=True
        self.allOff()
        self.fgs = Box(
            dg1000 = RigolFG1000("USB0::0x1AB1::0x0642::DG1ZA193403439::INSTR"),# (DG1022)
            dg900_A = RigolFG900("USB0::0x1AB1::0x0643::DG9A210800150::INSTR"), # (first DG952) By, Bz
            dg900_B = RigolFG900("USB0::0x1AB1::0x0643::DG9A210800149::INSTR")# (second DG952)
        )
        self.chans = {
            'pump': ['dg1000', 1], 
            'big_Bz': ['dg1000', 2], 
            'Bz': ['dg900_B', 1], 
        }

    
    def setPulsePattern(self, chanName, times, widths, heights, total_t):
        fg, chaNum = self.chans[chanName]
        t, y = makePulseTrain(pulse_widths, pulse_heights, sample_rate, Nsamples, smthFact=8)
        fg.uploadWaveform()
    
    #OLD STUFF===============
    def setWaveForms(self, t,VX, VY, VZ, bReorderOnly=False):
        self.t=t
        if 0:
            if t.size!= self.t.size or ~np.all(t==self.t):
                #print("t changed")            
                self.bXChanged=True
                self.bYChanged=True
                self.bZChanged=True
        if self.VX.size!= VX.size or ~np.all(self.VX==VX):
            #print("Xchanged")
            self.bXChanged=True
            self.bXAmpChanged=True
        if self.VY.size!= VY.size or ~np.all(self.VY==VY):
            #print("Ychanged")
            self.bYChanged=True
            self.bYAmpChanged=True
        if self.VZ.size!= VZ.size or ~np.all(self.VZ==VZ):
            #print("Zchanged")
            self.bZChanged=True
            self.bZAmpChanged=True
        

        if VX.size>0 and bReorderOnly:
            if self.bXChanged and self.VX.max()==VX.max() and self.VX.min()==VX.min():
                self.bXAmpChanged=False
            if self.bYChanged and self.VY.max()==VY.max() and self.VY.min()==VY.min():
                self.bYAmpChanged=False
            if self.bZChanged and self.VZ.max()==VZ.max() and self.VZ.min()==VZ.min():
                self.bZAmpChanged=False


        self.VX, self.VY, self.VZ = VX, VY, VZ
        self.updateOutputs()
        return

    def updateOutputX(self):
        fg=self.fg33500
        fg.allOff()
        fg.setRate(self.rateX, 0)
        if all(self.VX==0):
            fg.uploadWaveform(self.VX/max(abs(self.VX)), name='VOLX', chanNum=0);
            fg.setLH( -0.01, 0.01, chanNum=0 )
        else:
            #fg.uploadWaveform(self.VY/max(abs(self.VY)));
            fg.uploadWaveform(self.VX/max(abs(self.VX)),name='VOLX',chanNum=0);
            if self.bXAmpChanged:
                fg.setLH( self.VX.min(), self.VX.max(), chanNum=0 )
        errStr=fg.getErr()
        errVal=int(errStr.split(',')[0])
        #if errVal!= 0 and errVal != -221:
        if errVal:
            raise ValueError(errStr.split(',')[1])
        self.bXChanged=False
        self.bXAmpChanged=False

    def updateOutputY(self):
        fg=self.fg33500
        fg.allOff()
        fg.setRate(self.rateY, 1)
        if all(self.VY==0):
            fg.uploadWaveform(self.VY,name='VOLY',chanNum=1);
            fg.setLH( -0.01, 0.01, chanNum=1 )
        else:
            #fg.uploadWaveform(self.VY/max(abs(self.VY)));
            fg.uploadWaveform(self.VY/max(abs(self.VY)),name='VOLY',chanNum=1);
            if self.bYAmpChanged:
                fg.setLH( self.VY.min(), self.VY.max(), chanNum=1 )
        errStr=fg.getErr()
        errVal=int(errStr.split(',')[0])
        #if errVal!= 0 and errVal != -221:
        if errVal:
            raise ValueError(errStr.split(',')[1])
        self.bYChanged=False
        self.bYAmpChanged=False

    def updateOutputZ(self): 
        fg=self.fgK
        if self.bNoZ:
            return

        fg.setOutputState(False)
        #fg.setRate(self.rateZ)
        #self.VZ=self.VZ[:32000]
        fgZPeriod=1./self.rateZ*self.VZ.size
        print("fgZPeriod: {}".format(fgZPeriod))
        fg.setPeriod(fgZPeriod)
        if 1:
            if abs(max(self.VZ)-min(self.VZ))<0.05:
                fg.uploadWaveform(self.VZ*0, scl=False);
                fg.setOffset(0)
                fg.setAmp( 0.05 )
            else:
                fg.uploadWaveform(self.VZ/max(abs(self.VZ)),scl=True);
                if self.bZAmpChanged:
                    fg.setLH( self.VZ.min(), self.VZ.max() )

        errStr=fg.getErr()
        print(errStr)
        errVal=int(errStr.split(',')[0])
        #if errVal!= 0 and errVal != -221:
        if errVal!=0:
            raise ValueError(errStr.split(',')[1])
        self.bZChanged=False
        self.bZAmpChanged=False

    def updateOutputs(self):
        if self.bXChanged:
            self.updateOutputX()
        if self.bYChanged:
            self.updateOutputY()
        if self.bZChanged:
            self.updateOutputZ()

    def allOn(self):
        #self.fgP.allOn()
        self.fgK.allOn()
        self.fg33500.allOn()
        #self.fgK.allOn()

    def allOff(self):
        #self.fgP.allOff()
        self.fgK.allOff()
        self.fg33500.allOff()


    def setWaveformsResponseCorrected(t, Vx,Vy, Vz):
        """Should take the desired output waveforms (Vx,Vy, Vz) and change the actual output according to the known response.
        """
        raise NotImplementedError



if __name__=="__main__":
    from numpy import pi, sin, cos 
    t=np.arange(0,15e-3, 1e-6)
    y=np.sin(2*pi*t*1000)
    pfg=PmFgController()
    pfg.setRates(1e6,1e6,1e6)
    pfg.setWaveForms(y,y,y)