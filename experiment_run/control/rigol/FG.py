import pyvisa as visa
from pyvisa import VisaIOError
from pylab import fromstring, linspace, floor
import numpy as np
import abc
import pdb
from time import sleep
from functools import partial

def list_instruments():
    rm=visa.ResourceManager();
    resL=rm.list_resources()
    return resL

def scaleTo(data, sclTo):
    try:
        valRange= sclTo[1] - sclTo[0]
        lowV = data.min()
        highV = data.max()
        data = (data-lowV)/np.abs(highV-lowV) #Now between 0 and 1
        return data*valRange + sclTo[0]

    except TypeError:
        return data/abs(data).max()*sclTo


class FG(object):
    __metaclass__ = abc.ABCMeta
    #ip_address="136.159.248.161"
    addr = None
    handle=None;
    rm=None
    numChans=1.
    
 

    @staticmethod
    def array_to_text_block(data, sclTo = [-1, 1]):
        """We'll assume it'll be uploaded as integers"""
        data=np.array(data,dtype='f8')
        if sclTo is not None:
            data = scaleTo(data, sclTo)
        #data *= (2**15 -1)
        #dataInt=np.rint(data).astype(outputDtype)
        datStr=','.join(['{:.3f}'.format(num) for num in data])
        return datStr

    @staticmethod
    def array_to_binary_block(data, sclTo=[-1,1], dataMax = 2**13-1):
        data=np.array(data)
        if sclTo is not None:
            data = scaleTo(data, sclTo)
            #data/=abs(data).max()
        data *= dataMax
        data=np.rint(data + dataMax).astype('u2')
        dataBytes=bytes(data)
        N=len(dataBytes)
        Nstr=str(N)
        return ( "#{0}{1}".format(len(Nstr), Nstr),  dataBytes )

    #@abc.abstractmethod
    def connect(self, address=None):
        """ Connect to the instrument. 
        After this, self.handle will be an actual handle
        """
        if address is not None:
            self.addr=address
        #self.handle=visa.instrument("TCPIP::{0}::INSTR".format(self.ip_address));
        if self.rm is None:
            self.rm=visa.ResourceManager();
        self.handle=self.rm.open_resource("{0}".format(self.addr));
        self.configureHandle()
        self.check_connection();

    @abc.abstractmethod
    def configureHandle(self):
        """ Make sure handle is configured correctly for IO, e.g. set the query delay etc.
        This can be different for different instruments
        """
        self.handle.query_delay=0.5
        self.handle.timeout=5000
        normalWrite=self.handle.write
        def newWrite(*args, **kwargs):
            normalWrite(*args, **kwargs)
            sleep(.2)
        self.handle.write=newWrite

    def __init__(self, addr=None):
        if addr is not None:
            self.addr=addr
        self.connect();
        self.curWaveform = [None]*self.numChans
        self.last_upload_call = {}
        


    # REQUIRED TO IMPLEMENT=========================================
    @abc.abstractmethod
    def setOutputState(self, bOn, chNum=0):
        """ Turn that channel on/off
        """

    @abc.abstractmethod
    def setTriggerMode(self, mode="SINGLE"):
        """ Set the trigger mode: e.g. continuous or whatever
        """
        pass;
    @abc.abstractmethod
    def setTriggerDelay(self, delay):
        """ Amount of time to wait after a trigger
        """
    
    @abc.abstractmethod
    def softwareTrigger(self):
        """Trigger it once
        """
    
    ## Pre-defined methods
    def close(self):
        self.handle.close()
        
    def check_connection(self):
        try:
            ret_val=self.handle.query("*IDN?");
            print(ret_val)
        except VisaIOError:
            print("not connected! But lets try to fix it:")
            del self.handle
            self.handle=None;
            self.connect();
    
    def allOn(self):
        """Turn all outputs on"""
        for chNum in range(self.numChans):
            self.setOutputState(True, chNum);

    def allOff(self):
        """Turn all outputs off"""
        for chNum in range(self.numChans):
            self.setOutputState(False, chNum);

    
    def setOutputWaveform(self, t, x, chanNum=0, bUseFullScale=True):
        """Upload a waveform (t, x) and set it as active on channel chNum
        """
        last_p = self.last_upload_call[chanNum] if chanNum in self.last_upload_call else None
        if last_p and np.all(t == last_p[0]) and np.all(x == last_p[1]) and bUseFullScale == last_p[2]:
            print("skipping repeat upload")
            return
        else:
            print("running setOutputWaveform")
        self.setOutputState(0, chanNum)
        self.checkErr()
        if bUseFullScale:
            mn, mx = x.min(), x.max()
            absMax = max([abs(mn), abs(mx)])
            sclTo = [mn/absMax, mx/absMax]
            self.uploadWaveform(x, sclTo=sclTo, chanNum=chanNum)
            self.checkErr()
            self.setLowHigh( -absMax, absMax, chanNum=chanNum )
        else:
            self.uploadWaveform(x, sclTo=None, chanNum=chanNum)
            self.checkErr()
        self.setRate(1/(t[1]-t[0]), chanNum=chanNum)#Period(t[-1]-t[0])
        self.checkErr()
        self.setOutputState(1, chanNum=chanNum)
        self.last_upload_call[chanNum] = [t, x, bUseFullScale]

    def setLH(self, low, high, chanNum=0):
        #self.setAmp(0.01)
        offs=(low+high)/2.
        print("Offs:{}, low:{}, high:{}".format(offs,low,high))
        if high < low:
            newLow=high#-(2*low-high)
            newHigh=low
            print("newlow, newhigh: {} {}".format(newLow, newHigh))
            #print("newLow: {}, newHigh: {}".format(newLow, newHigh))
            self.setInverted(False, chanNum=chanNum);
            self.setOffset(offs, chanNum=chanNum)
            self.setLowHigh(newLow, newHigh,chanNum=chanNum)
            self.setInverted(True, chanNum=chanNum);
        else:
            self.setInverted(False, chanNum=chanNum);
            self.setOffset(offs, chanNum=chanNum)
            sleep(0.1)
            self.setLowHigh(low,high, chanNum=chanNum)
        #self.allOn()

    #def setLow(self,val):
        
    #    def setHigh(self,val):

        
    def wait(self):
        self.handle.query("*OPC?");
    
    def getErr(self):
        return self.handle.query("SYST:ERR?")

    def checkErr(self):    
        errStr=self.getErr()
        errVal=int(errStr.split(',')[0])
        if errVal:
            raise ValueError(f"Error val{errVal}: {errStr.split(',')[1]}")

    # OPTIONAL METHODS -----------------
    def setInverted(self, bInvert=True, chanNum=0):
        raise NotImplementedError()

    def setLowHigh(self, low, high, chanNum=0):
        """Set the low and high values for the waveform"""
        raise NotImplementedError()

    def setOffset(self, offset, chanNum=0):
        """Set the offset for the waveform"""
        raise NotImplementedError()

    def setPeriod(self, period, chanNum=0):
        """Set the period for the waveform"""
        raise NotImplementedError()

    def setAmp(self, amp, chanNum=0):
        """Set the amp for the waveform"""
        raise NotImplementedError()

    def uploadWaveform(self, wvfm, name="VOLATILE"):
        """Set the amp for the waveform"""
        raise NotImplementedError()
