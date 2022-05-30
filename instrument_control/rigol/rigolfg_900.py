""" Control of the Rigol DG952s. Not very tested. 
Mostly just copied from the old DG1000Z code. Some bits won't work as expected (because, Rigol)
"""
from time import sleep
from pylab import *
import time
from numpy import pi
import numpy as np
import pdb

from . import FG

class RigolFG(FG.FG):
    #addr="USB0::0x1AB1::0x0588::DG1D124004333::INSTR"
    #addr="USB0::0x1AB1::0x0643::DG9A210800150::INSTR"

    numChans=2 
    


    def configureHandle(self):
        self.handle.query_delay=0.2 #Used to use 0.5 s here
        self.handle.timeout=5000
        normalWrite=self.handle.write
        def newWrite(*args, **kwargs):
            normalWrite(*args, **kwargs)
            sleep(.1) #used to use 1.0 seconds here for the crappy rigols
        self.handle.write=newWrite

    def setOutputState(self, bOn, chanNum=0):
        headStr= "OUTPUT{}".format(chanNum+1)
        stateStr= 'ON' if bOn else 'OFF'
        #sleep(0.5);
        out=self.handle.write("{} {}".format(headStr, stateStr));
        #sleep(0.5);
        return out

    @staticmethod
    def array_to_text_block(data, scl=True):
        data=np.array(data,dtype='f8')
        if scl:
            lowV=data.min()
            highV=data.max()
            data=(data-lowV)/np.abs(highV-lowV)*2.-1.0
            #data/=abs(data).max()
            #data*=0.5#8191
        dataInt=np.rint(data).astype('f8')
        #pdb.set_trace()
        datStr=','.join([str(num) for num in dataInt])
        #print(datStr[:100])
        return datStr
    @staticmethod
    def array_to_binary_block(data, scl=True):
        data=np.array(data, dtype='f8')
        if scl and not np.all(data==0):
            data/=abs(data).max()
            data*=(2**15-1)
            
        data=np.rint(data).astype('i2')
        dataBytes=bytes(data)
        N=len(dataBytes)
        Nstr=str(N)
        return ( "#{0}{1}".format(len(Nstr), Nstr),  dataBytes )
        #return "#{0}{1}{2}".format(len(Nstr), Nstr), data.tobytes())
    def uploadWaveform(self,y, scl=True, chanNum=0):
        if len(y)> 2**14:
            print("Uploading waveforms of lenght > 2^14 usually doesn't have expected results.")
        #datStr=self.__super__.array_to_text_block(y, 0, 16383, scl=scl)
        #datStr=self.array_to_text_block(y, scl=scl)
        datBlkHead,datBlk=self.array_to_binary_block(y, scl=scl)
        #print(datStr)
        head = bytes(":SOURce{}:TRACe:DATA:DAC16 VOLATILE,END,{}".format(chanNum+1,datBlkHead), 'utf-8')
        print(head)
        self.handle.write_raw(head + datBlk)
        #self.handle.write("DATA {},{}".format(name,datStr))
        sleep(1); #Hopefully this isn't needed?
        self.curWaveform[chanNum] = y

    def setOutputWaveform(self, y, sampleRate = 1e6,chanNum=0, bUseFullScale=False):
        print("don't use setOutputWaveform until we can work out what is wrong with setPeriod")
        self.setOutputState(False, chanNum=chanNum)
        if bUseFullScale:
            self.setLH(y.min(), y.max())
            y = (y - (y.max() + y.min)/2  )* 2/(y.max()-y.min()) 

        self.uploadWaveform(y, chanNum=chanNum)
        self.setRate(sampleRate, chanNum=chanNum)
        errStr=self.getErr()
        errVal=int(errStr.split(',')[0])
        #if errVal!= 0 and errVal != -221:
        if errVal:
            raise ValueError(errStr.split(',')[1])
        
        self.setOutputState(True, chanNum=chanNum)

    def uploadAndSetWaveform(self, t,x,chanNum=0):
        """Simple upload a waveform and set it active.

        NEEDS UPDATING
        """
        raise NotImplementedError("Needs updating!")
        figure()
        chanStr="" if chanNum==0 else ":CH2"
        self.handle.write('FUNC:USER{} VOLATILE'.format(chanStr))
        
        x=np.array(x)
        T=max(t)-min(t)
        F=int(1./T)
        self.handle.write('FREQ{} {}'.format(chanStr, F))
        #sleep(0.4)
        #self.handle.write('VOLT:UNIT{} VPP'.format(chanStr))
        #sleep(0.4)
        lowV=x.min()
        highV=x.max()
        x=(x-lowV)/np.abs(highV-lowV)*1.9999-1.0

        if lowV <-5. or highV>5.:
            raise ValueError("Waveform must be between -5 and 5 V")
        self.handle.write('VOLT:LOW{} {:.3f}'.format(chanStr, lowV))
        #sleep(0.4)
        self.handle.write('VOLT:HIGH{} {:.3f}'.format(chanStr, highV))
        #sleep(0.4);
        
        if chanNum==0:
            name= "CH1DEF" 
            #headStr= "FUNC:USER"
        elif chanNum==1:
            name="CH2DEF"
            self.handle.write("FUNC:CH2 USER")
            sleep(0.3);
            #headStr= "FUNC:USER:CH2"
        else:
            raise ValueError("Channel can be 0 or 1")
        
        catL=self.handle.query("DATA:CAT?").split(',')
        print(catL)
        while '"{}"'.format(name) in catL:
            self.handle.write("DATA:DEL {}".format(name) )
            sleep(0.2);
            catL=self.handle.query("DATA:CAT?").split(',')
            print("deleted 1");
        # This should be modified slightly, probably
        outForm=x*16383
        #datStr=','.join(['%i' % num for num in outForm])
        datStr=','.join(['{:.3f}'.format(num) for num in x])
        print(datStr[:100])
        #pdb.set_trace()
        plot(x)
        print("write: {}".format( self.handle.write("DATA VOLATILE,{}".format(datStr))) )
        sleep(8.0);
        print("copy, {}".format(self.handle.write('DATA:COPY {}'.format(name))) )
        sleep(8.0);
        tStart=time.time()
        while not '"{}"'.format(name) in catL:
            sleep(0.3)
            catL=self.handle.query("DATA:CAT?").split(',')
            print(catL)
            if time.time()-tStart > 5: 
                raise TimeoutError("Timed out waiting for waveform to be uploaded")
        sleep(0.5)
        out=self.handle.write('FUNC:USER{} {}'.format(chanStr, name))
        sleep(0.4)
        print("func:user {}".format(out) );
        return out
        
    def setTriggerMode(self, mode="ext", chanNum=0):
        """Modes are 'int', 'ext', or 'man'"""
        #allowed_modes=['int', 'ext', 'man']
        modeTransDict={'int':'IMM',
                'ext':'EXT',
                'man':'BUS'}
        if mode not in modeTransDict:
            raise ValueError('Trigger mode needs to be one of {}'.format(modeTransDict.keys()) )
        label=modeTransDict[mode]
        return self.handle.write("TRIG{0}:SOUR {1}".format(chanNum+1, label)) 
        
    def setTriggerDelay(self,delay, chanNum=0):
        return self.handle.write('TRIG{}:DEL {}'.format(chanNum+1, delay))

    def trigger(self):
        raise NotImplementedError('No software trigger available for the rigol1000')

    def setPeriod(self, T, chanNum=0):
        #self.handle.write('FREQ {:.5f}'.format(1./T))
        print("Called setPeriod- but setting the period doesn't work on the DG952. Use sample rate instead")
        self.handle.write('SOUR{}:PERIOD {:.5f}'.format(chanNum+1,T))


    def setLowHigh(self, low, high, chanNum=0):
        self.handle.write('SOUR{}:VOLT:LOW {:.3f}'.format(chanNum+1, low) )
        #sleep(1)
        self.handle.write('SOUR{}:VOLT:HIGH {:.3f}'.format(chanNum+1, high) )

    def setOffset(self, offset, chanNum=0):
        self.handle.write('SOUR{}:VOLT:OFFS {:.3f}'.format(chanNum+1,offset) )
    
    def setRate(self, sampRate, chanNum=-1):
        if chanNum == -1:
            self.setRate(sampRate, 0)
            self.setRate(sampRate, 1)
        else:
            self.handle.write("SOUR{}:FUNC:SEQ:SRAT {}".format(chanNum+1,sampRate))

    def getSampleRate(self, chanNum=0):
            return float(self.handle.query(f"SOUR{chanNum+1}:FUNC:SEQ:SRAT?"))

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
        super().setInverted(bInvert,)
        #sleep(1)
if __name__=="__main__":
    import numpy as np
    addr='USB0::0x1AB1::0x0643::DG9A210800149::INSTR'
    fg=RigolFG(addr);
    def sendPulse(tDelay=1, tWidth=3000, tTotal=4096):

        t=np.linspace(0,tTotal,16000)*1.0;
        y=np.where( (t>tDelay) & (t<tDelay+tWidth), 5.0, 0.)
        y2=np.where( (t>tDelay+tTotal/2) & (t<tDelay+tWidth+tTotal/2), 5.0, 0.)

        #fg.setLoad(50,0)
        from pylab import plot,show
        plot(t,y)
        plot(t,y2)

        fg.setOutputWaveform(y, sampleRate=t.size/tTotal, chanNum=0);
        fg.setOutputWaveform(y2, sampleRate=t.size/tTotal, chanNum=1)
        #fg.setPeriod(tTotal*1e-6);
        #fg.setLowHigh(0,4.5)
        print(fg.handle.query("VOLT:OFFS?"))

        #print (rfg.setOutputWaveForm(t, y2, 1))
        #fg.allOn()
        #print("stuff")
        show()

    sendPulse(0.,200e-6,12e-3)
    #fg.setTriggerDelay(200*1e-6)
    #fg.setOffset(-0.01);
