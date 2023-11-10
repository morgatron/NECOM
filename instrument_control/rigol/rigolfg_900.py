""" Control of the Rigol DG952s. Not very tested. 
Mostly just copied from the old DG1000Z code. Some bits won't work as expected (because, Rigol)
"""
from time import sleep
from pylab import *
import time
from numpy import pi
import numpy as np
import pdb

#from .FG import FG
try:
    from . import FG
except ImportError:
    import FG

def array_to_binary_block(data):
    data=np.array(data)
    if np.max(np.abs(data))>1:
        raise ValueError("data needs to be within -1,1 for uploading")
    data *= (2**15-1)
    data=np.rint(data).astype('i2')
    dataBytes=bytes(data)
    N=len(dataBytes)
    Nstr=str(N)
    return ( "#{0}{1}".format(len(Nstr), Nstr),  dataBytes )

class RigolFG900(FG.FG):
    #addr="USB0::0x1AB1::0x0588::DG1D124004333::INSTR"
    #addr="USB0::0x1AB1::0x0643::DG9A210800150::INSTR"
    numChans=2 

    def configureHandle(self):
        self.handle.query_delay=0.2 #Used to use 0.5 s here
        self.handle.timeout=5000
        normalWrite=self.handle.write
        def newWrite(*args, **kwargs):
            normalWrite(*args, **kwargs)
            sleep(.2) #used to use 1.0 seconds here for the crappy rigols
        self.handle.write=newWrite

    def setOutputState(self, bOn, chanNum=0):
        headStr= "OUTPUT{}".format(chanNum+1)
        stateStr= 'ON' if bOn else 'OFF'
        #sleep(0.5);
        out=self.handle.write("{} {}".format(headStr, stateStr));
        #sleep(0.5);
        return out

    #@staticmethod
    #def array_to_text_block(data, scl=True):
    #    data=np.array(data,dtype='f8')
    #    if scl:
    #        lowV=data.min()
    #        highV=data.max()
    #        data=(data-lowV)/np.abs(highV-lowV)*2.-1.0
    #        #data/=abs(data).max()
    #        #data*=0.5#8191
    #    dataInt=np.rint(data).astype('f8')
    #    #pdb.set_trace()
    #    datStr=','.join([str(num) for num in dataInt])
    #    #print(datStr[:100])
    #    return datStr
    #@staticmethod
    #def array_to_binary_block(data, scl=True):
    #    data=np.array(data, dtype='f8')
    #    if scl and not np.all(data==0):
    #        data/=abs(data).max()
    #        data*=(2**15-1)
    #        
    #    data=np.rint(data).astype('i2')
    #    dataBytes=bytes(data)
    #    N=len(dataBytes)
    #    Nstr=str(N)
    #    return ( "#{0}{1}".format(len(Nstr), Nstr),  dataBytes )
    #    #return "#{0}{1}{2}".format(len(Nstr), Nstr), data.tobytes())

    def uploadWaveform(self, y, sclTo=None, chanNum=0):
        if len(y)> 2**14:
            print("Uploading waveforms of lenght > 2^14 usually doesn't have expected results.")
        #datStr=self.__super__.array_to_text_block(y, 0, 16383, scl=scl)
        #datStr=self.array_to_text_block(y, scl=scl)
        maxBlkSize = 9000
        Npts = len(y)
        flag = "CON"
        if sclTo is not None:
            y = FG.scaleTo(y, sclTo)

        pointsSent = 0
        while pointsSent < Npts:
            print(f"pts sent: {pointsSent}")
            datBlkHead,datBlk= array_to_binary_block(y[pointsSent:pointsSent+maxBlkSize])
            pointsSent += maxBlkSize
            if pointsSent >= Npts:
                flag = "END"
            head = bytes(f":SOURce{chanNum+1}:TRACe:DATA:DAC16 VOLATILE,{flag},{datBlkHead}", 'utf-8')
            print(head)
            self.handle.write_raw(head + datBlk)
        sleep(1); #Hopefully this isn't needed?
        self.curWaveform[chanNum] = y

    def setTriggerMode(self, mode="ext", chanNum=0):
        #allowed_modes=['int', 'ext', 'man']
        modeTransDict={'int':'INT',
                'ext':'EXT',
                'man':'BUS'}
        if not mode in modeTransDict:
            raise ValueError('Trigger mode needs to be one of {}'.format(modeTransDict.keys()) )
        label=modeTransDict[mode]
        return self.handle.write(f":TRIG{chanNum+1}:SOUR {label}") 
        
    def setTriggerDelay(self,delay):
        return self.handle.write(f'TRIG{chanNum+1}:DEL {delay}')

    def setBurstPeriod(self, period, chanNum= 0):
        return self.handle.write(f":SOUR{chanNum+1}:BURS:INT:PER {period}")

    def setBurstMode(self, mode, chanNum= 0):
        possibleModes = ['TRIG', 'INF', 'GAT']
        if mode not in possibleModes:
            raise ValueError(f"Mode needs to be one of {possibleModes}")
        self.setBurstOn(True, chanNum)
        return self.handle.write(f":SOUR{chanNum+1}:BURS:MODE {mode}")
    def setBurstOn(self, bOn=True, chanNum=0):
        onOff = "ON" if bOn else "OFF"
        self.handle.write(f":SOUR{chanNum+1}:BURS {onOff}")
        self.handle.write(f":SOUR{chanNum+1}:BURS:IDLE FPT")
    def trigger(self, chanNum=0):
        return self.handle.write(f":TRIG{chanNum+1}:IMM")

    def setLowHigh(self, low, high, chanNum=0):
        #self.handle.write('VOLT:LOW {:.3f}'.format(low) )
        self.handle.write(f':SOUR{chanNum+1}:VOLT:LOW {low:.5f}' )
        #sleep(1)
        #self.handle.write('VOLT:HIGH {:.3f}'.format(high) )
        self.handle.write(f':SOUR{chanNum+1}:VOLT:HIGH {high:.5f}' )

    def setOffset(self, offset, chanNum=0):
        self.handle.write(f':SOUR{chanNum+1}:VOLT:OFFS {offset:.5f}' )
    
    def setRate(self, sampRate, chanNum=-1):
        if chanNum == -1:
            self.setRate(sampRate, 0)
            self.setRate(sampRate, 1)
        else:
            self.handle.write("SOUR{}:FUNC:SEQ:SRAT {}".format(chanNum+1,sampRate))

    def getSampleRate(self, chanNum=0):
            return float(self.handle.query(f"SOUR{chanNum+1}:FUNC:SEQ:SRAT?"))

    def setPeriod(self, period, chanNum=-1):
        self.setRate( len(self.curWaveform[0])/period, chanNum=chanNum )

    def setLoad(self, load=50, chanNum=0):
        """If load is -ve, infinite is assumed
        """
        #chStr="" if chanNum==0 else ":CH2"
        if load >=1:
            loadStr=str(load)
        else:
            loadStr="INF"
        self.handle.write(f'OUTP{chanNum+1}:LOAD {loadStr}');

    def setInverted(self, bInvert=True, chanNum=0):
        invStr = "INV" if bInvert else "NORM"
        self.handle.write(f":OUTP{chanNum+1}:POL {invStr}")

if __name__=="__main__":
    from numpy import *
    import fg_addr
    t= linspace(0,1e-3, 4000)
    y = sin(2*pi*t*3000) 
    fg900=RigolFG900(fg_addr.field2);
    def sendAlternatingPulses(fg, amp=1, tDelay=10e-6, tWidth=20e-6, tTotal=3000e-6, chanNum=1):

        t=np.linspace(0,tTotal*2, 20000 )*1.0;
        y1 = np.where( (t>tDelay) & (t<tDelay+tWidth), amp, 0.)
        y2 = np.where( (t>tDelay+tTotal) & (t<tDelay+tWidth+tTotal), -amp, 0.)
        y = y1 + y2

        fg.setLoad(50,0)
        from pylab import plot,show
        plot(t,y)
        fg.setOutputWaveform(t, y,chanNum=chanNum)
        #fg.uploadWaveform(y, chanNum=0);
        #fg.setPeriod(tTotal*1e-6);
        #fg.setLowHigh(0,8)
        #print (rfg.setOutputWaveForm(t, y2, 1))
        #fg.allOn()
        #print("stuff")
        show()
    def sendPulse(fg, amp=1, tDelay=1, tWidth=200, tTotal=4096, chanNum=0):

        t=np.linspace(0,tTotal,50000)*1.0e-6;
        y=np.where( (t>tDelay) & (t<tDelay+tWidth), amp, 0.)

        fg.setLoad(50,0)
        from pylab import plot,show
        plot(t,y)
        fg.uploadAndSetWaveform(t, y,chanNum=chanNum)
        #fg.uploadWaveform(y, chanNum=0);
        #fg.setPeriod(tTotal*1e-6);
        #fg.setLowHigh(0,8)
        print(fg.handle.query("VOLT:OFFS?"))
        #print (rfg.setOutputWaveForm(t, y2, 1))
        #fg.allOn()
        #print("stuff")
        show()

    #sendPulse(0.,10.,100)
    #fg.setTriggerDelay(200*1e-6)
