import numpy as np

class QueueView(object):
    """Simple wrapper around an numpy array acting as a ring buffer that keeps track of a moving view of it
    
    """
    def __init__(self, arr, startInd=0, startSize=0):
        self._data=arr
        self.startInd=0;
        self.endInd=startInd+startSize
        self.Nmax=arr.shape[-1]

    def advanceEnd(self, Npts):
        self.endInd+=Npts
        if self.endInd>self.Nmax:
            self.endInd-=self.Nmax


    def view(self):
        if self.endInd<self.startInd:
            dat=np.hstack([self._data[...,self.startInd:], self._data[..., :self.endInd]])
        else:
            dat=self._data[...,self.startInd:self.endInd]
        return dat

    def dumpStart(self, ind):
        self.startInd+=ind
        if self.startInd>self.Nmax:
            self.startInd-=self.Nmax




if __name__=="__main__":
    arr=np.arange(2000).reshape(2,1000)
    qv=QueueView(arr, startInd=0, startSize=0)
