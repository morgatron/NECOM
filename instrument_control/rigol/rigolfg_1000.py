""" Control of rigol 100x
"""
from time import sleep
from pylab import *
import time
from numpy import pi
import numpy as np
import pdb

#from . import FG
import FG
def array_to_binary_block(data):
    dataMax = 2**13 -1
    data=np.array(data)
    if np.max(np.abs(data))>1:
        raise ValueError("data needs to be within -1,1 for uploading")
    data += 1
    data *= dataMax
    data=np.rint(data).astype('u2')
    dataBytes=bytes(data)
    N=len(dataBytes)
    Nstr=str(N)
    return ( "#{0}{1}".format(len(Nstr), Nstr),  dataBytes )
class RigolFG1000(FG.FG):
    #addr="USB0::0x1AB1::0x0588::DG1D124004333::INSTR"
    numChans=2 
    
    def chanStr(self, chanNum):
        if chanNum==0:
            return ":CH1"
        elif chanNum == 1:
            return ":CH2"
        else:
            raise ValueError(f"Don't have a channel {chanNum}")

    def configureHandle(self):
        self.handle.query_delay=0.2
        self.handle.timeout=5000
        normalWrite=self.handle.write
        def newWrite(*args, **kwargs):
            normalWrite(*args, **kwargs)
            sleep(.2)
        self.handle.write=newWrite

    def setOutputState(self, bOn, chanNum=0):
        stateStr= 'ON' if bOn else 'OFF'
        #sleep(0.5);
        out=self.handle.write(f":OUTP{chanNum+1} {stateStr}");
        #sleep(0.5);
        return out

   # @staticmethod
   # def array_to_text_block(data, scl=True):
   #     data=np.array(data,dtype='f8')
   #     if scl:
   #         lowV=data.min()
   #         highV=data.max()
   #         data=(data-lowV)/np.abs(highV-lowV)*2.-1.0
   #         #data/=abs(data).max()
   #         #data*=0.5#8191
   #     dataInt=np.rint(data).astype('f8')
   #     #pdb.set_trace()
   #     datStr=','.join([str(num) for num in dataInt])
   #     #print(datStr[:100])
   #     return datStr
        #return "#{0}{1}{2}".format(len(Nstr), Nstr), data.tobytes())
    # def uploadWaveform(self,y, sclTo=1, chanNum=0, name="VOLATILE"):
    #     #datStr=self.__super__.array_to_text_block(y, 0, 16383, scl=scl)
    #     datStr=self.array_to_text_block(y, sclTo=sclTo)
    #     print(datStr)
    #     #self.handle.write(f":SOUR{chanNum+1}:DATA {name},{datStr}")
    #     self.handle.write(f":SOUR{chanNum+1}:TRACE:DATA:DAC16 {name},{datStr}")
    #     sleep(1); #Hopefully this isn't needed?
    def uploadWaveformOld(self,y, sclTo=1., chanNum=0):
        if len(y)> 2**14:
            print("Uploading waveforms of lenght > 2^14 usually doesn't have expected results.")
        #datStr=self.__super__.array_to_text_block(y, 0, 16383, scl=scl)
        #datStr=self.array_to_text_block(y, scl=scl)
        datBlkHead,datBlk=self.array_to_binary_block(y, sclTo=sclTo)
        #print(datStr)
        head = bytes(f":SOURce{chanNum+1}:TRACe:DATA:DAC16 VOLATILE,END,{datBlkHead}", 'utf-8')
        print(head)
        self.handle.write_raw(head + datBlk)
        #self.handle.write("DATA {},{}".format(name,datStr))
        sleep(1); #Hopefully this isn't needed?
        self.curWaveform[chanNum] = y
    def uploadWaveform(self, y, sclTo=None, chanNum=0):
        if len(y)> 2**14:
            print("Uploading waveforms of lenght > 2^14 usually doesn't have expected results.")
        #datStr=self.__super__.array_to_text_block(y, 0, 16383, scl=scl)
        #datStr=self.array_to_text_block(y, scl=scl)
        maxBlkSize = 7000
        Npts = len(y)
        flag = "CON"
        if sclTo is not None:
            y = FG.scaleTo(y, sclTo)
        print("max, min:")
        print(y.max(), y.min())
        pointsSent = 0
        while pointsSent < Npts:
            print(f"pts left: {pointsSent}")
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

    def trigger(self, chanNum=0):
        return self.handle.write(f":TRIG{chanNum+1}:IMM")

    def setBurstOn(self, bOn=True, chanNum=0):
        onOff = "ON" if bOn else "OFF"
        self.handle.write(f":SOUR{chanNum+1}:BURS {onOff}")
    #def setPeriod(self, T):
    #   self.handle.write('FREQ {:.5f}'.format(1./T))
    def setPeriod(self, period, chanNum=-1):
        self.setRate( len(self.curWaveform[0])/period, chanNum = chanNum )

    def setRate(self, sampRate, chanNum=-1):
        if chanNum == -1:
            self.setRate(sampRate, 0)
            self.setRate(sampRate, 1)
        else:
            self.handle.write("SOUR{}:FUNC:ARB:SRATE {}".format(chanNum+1,sampRate))        

    def setLowHigh(self, low, high, chanNum=0):
        #self.handle.write('VOLT:LOW {:.3f}'.format(low) )
        self.handle.write(f':SOUR{chanNum+1}:VOLT:LOW {low:.5f}' )
        #sleep(1)
        #self.handle.write('VOLT:HIGH {:.3f}'.format(high) )
        self.handle.write(f':SOUR{chanNum+1}:VOLT:HIGH {high:.5f}' )

    def setOffset(self, offset, chanNum=0):
        self.handle.write(f':SOUR{chanNum+1}:VOLT:OFFS {offset:.5f}' )
    
    def setLoad(self, load=50, chNum=0):
        """If load is -ve, infinite is assumed
        """
        chStr="" if chNum==0 else ":CH2"
        if load >0:
            loadStr="50"
        else:
            loadStr="INF"
        self.handle.write('OUTP:LOAD:{} {}'.format(chStr,loadStr) );

    def setInverted(self, bInvert=True, chanNum=0):
        invStr = "INV" if bInvert else "NORM"
        self.handle.write(f":OUTP{chanNum+1}:POL {invStr}")

if __name__=="__main__":
    from numpy import *
    import fg_addr
    t= linspace(0,1e-3, 12000)
    y = sin(2*pi*t*3000) 
    fg=RigolFG1000(fg_addr.pump);
    def sendPulse(tDelay=1, tWidth=200, tTotal=4096):

        t=np.linspace(0,tTotal,tTotal*20)*1.0;
        y=np.where( (t>tDelay) & (t<tDelay+tWidth), 5.0, 0.)

        fg.setLoad(50,0)
        from pylab import plot,show
        plot(t,y)
        fg.uploadAndSetWaveform(t, y,chNum=0)
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
    #fg.setOffset(-0.01);
