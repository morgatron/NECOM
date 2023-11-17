
import numpy as np
import pdb

class RingBuffer(object):
    """Simple wrapper around an numpy array acting as a ring buffer that keeps track of a moving view of it
    
    """

    def __init__(self, initialData, size = int(1e6), dtype=None):

        dtype = dtype if dtype else initialData.dtype
        dataShape = list(initialData.shape)
        self.Nmax = size if size > dataShape[-1] else dataShape[-1]
        dataShape[-1] = self.Nmax
        self._data = np.zeros(dataShape, dtype = dtype)
        self.startInd = 0;
        self.endInd = initialData.shape[-1]
        self.extendRight(initialData)
        #self._data = initialData

    def advanceEnd(self, Npts):
        self.endInd+=Npts
        if self.endInd>self.Nmax:
            self.endInd-=self.Nmax

    def dumpStart(self, ind):
        self.startInd+=ind
        if self.startInd>self.Nmax:
            self.startInd-=self.Nmax

    def view(self):
        if self.endInd<self.startInd:
            dat=np.hstack([self._data[...,self.startInd:], self._data[..., :self.endInd]])
        else:
            dat=self._data[...,self.startInd:self.endInd]
        return dat

    def extendRight(self, arr):
        """add new data to the right
        Needs testing!
        """
        N = arr.shape[-1]
        newEndInd = self.endInd + N 
        overflow = newEndInd - self.Nmax
        if overflow >0:
            self._data[self.endInd:] = arr[:N-overflow]
            self._data[:overflow] = arr[-overflow:]
            self.endInd = overflow

        else:
            self._data[..., self.endInd:newEndInd] = arr
            self.endInd = newEndInd
        # advanceEnd(arr.shape[-1])
        # self.view()







if __name__=="__main__":
    from matplotlib import pyplot as plt
    rb=RingBuffer(initialData = np.zeros((2,0)), size=int(1e6) )

    t0 = np.linspace(0,1000, int(1e6))
    y0 = np.vstack([np.sin(t0*2*np.pi), np.cos(2*t0*2*np.pi)])
    N = 10
    for k in range(2):
        t = t0[k*N:(k+1)*N]
        y = y0[:,k*N:(k+1)*N]
        rb.extendRight(y)

    dat = rb.view()
    plt.plot(dat[0],'.')
    plt.plot(dat[1],'.')


