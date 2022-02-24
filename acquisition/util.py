from __future__ import print_function, division
import pylab as pl
from numpy.fft import fft, ifft
import numpy as np
from scipy.signal import wiener, filtfilt, butter, gaussian
from numpy import iterable, array, pi, nan
from bisect import insort, bisect_left





def pruneRagged(arrL, axis=-1, maxPts=None):
    lengths=[ar.shape[axis] for ar in arrL]
    newN=np.min(lengths)
    if maxPts is not None and newN<np.max(lengths)-maxPts:
        raise ValueError("Array is just too ragged");
    pruned=np.array([ar[:newN] for ar in arrL])
    return pruned



def downsample_npts(vector, npts, axis=0, bRetErr=False):
    """
    downsample(vector, factor):
    Downsample (by averaging) a vector by an integer factor.
    """
    vector = np.rollaxis(vector , axis)


    #pdb.set_trace()
    if (vector.shape[0] % npts):
        cropN=vector.shape[0]%npts
        vectorNew=vector[:-cropN]
        #print( "Length of 'vector' is not divisible by 'factor'=%d!" %npts)
        #asd
        #return 0
    else:
        vectorNew=vector
    #vectorNew.shape = (len(vector)/npts, npts)
    reshaped=vectorNew.reshape(int(vector.shape[0]/npts), npts, *vectorNew.shape[1:])
    mns=np.mean(reshaped, axis=1)
    if bRetErr:
        sterr=np.std(reshaped, axis=1)
    if axis == -1:
        axis = mns.ndim - 1
    mns = np.rollaxis(mns , 0, axis + 1)
    if bRetErr:
        return mns, sterr
    return mns

def downsample(vector, factor):
    """
    downsample(vector, factor):
    Downsample (by averaging) a vector by an integer factor.
    """
    if (len(vector) % factor):
        print("Length of 'vector' is not divisible by 'factor'=%d!" % factor)
        return 0
    vector.shape = (len(vector)/factor, factor)
    return np.mean(vector, axis=1)


def smoothalong(x,window_len=10,window='flat',axis=0):
    #~ np.apply_along_axis(
    smoothed=np.apply_along_axis(smooth,axis,x,window_len,window)
    return smoothed

def smooth(x,window_len=10,window='flat', boundaryMode="void"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    
    see also: 

    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string   

    """
#    if x.ndim != 1:
#        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if window in ['hanning', 'hamming', 'bartlett', 'blackman']:
        w=eval('np.'+window+'(window_len)')
    elif window == 'flat': #moving average
        w=pl.ones(window_len,'d')
    elif hasattr(window, "__len__"):
        window_len=len(window)
        w=array(window, dtype='f8')
    
    #elif not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    else:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    #s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    #pl.figure()
    #pl.plot(s,'.')
    #pl.xlim([0,2*window_len]);
    #y=np.convolve(w/w.sum(),s,mode='same')
    w=w[::-1]/w.sum()
    Wcum=w.cumsum()
    WLon2=int(window_len/2)
    y=np.convolve(w, x, mode='same')
    #y[:WLon2]/= Wcum[WLon2:-1]
    y[:WLon2]/=Wcum[-WLon2-1:-1]
    #y[-WLon2:]/=1-Wcum[:WLon2]
    y[-(WLon2):]/=(1-Wcum[:WLon2])
    return y
    #return y[window_len-1:-window_len+1]


def gauss(t,p):
    '''
    Gaussian function, t is axis, 
    p[0] is peak value.
    p[1] is position of peak.
    p[2] is fwhm.
    
    p[0]*pl.exp(   -pow(t-p[1],2)/( (p[2])**2/(4*pl.log(2)) )    ) 
    
    '''
    return( p[0]*pl.exp(   -pow(t-p[1],2)/( p[2]**2/(4*pl.log(2)) )    ) )

class RingBuffer(object):
    """A simple FIFO ring-buffer class.

    Examples:
    >>> buf=RingBuffer(5)
    >>> buf
    array([], dtype=float64)
    >>> for k in range(7): buf.append(k) #overfill it and lose the oldest data
    >>> buf
    array([ 2.,  3.,  4.,  5.,  6.])
    """
    def __init__(self, size_max, default_value=0.0, dtype=float):
        """Initialize memory with default value"""
        self.size_max = size_max

        self._data = np.empty(size_max, dtype=dtype)
        self._data.fill(default_value)

        self.size = 0

    def append(self, value):
        """append an element"""
        self._data = np.roll(self._data, 1)
        self._data[0] = value 

        self.size += 1

        if self.size == self.size_max:
            self.__class__  = RingBufferFull

    def get_all(self):
        """return a list of elements from the oldest to the newest"""
        return(self._data)

    def get_partial(self):
        return(self.get_all()[0:self.size])

    def __getitem__(self, key):
        """get element"""
        return(self._data[key])

    def __repr__(self):
        """return string representation"""
        #s = self._data.__repr__()
        #s = s + '\t' + str(self.size)
        #s = s + '\t' + self.get_all()[::-1].__repr__()
        s = self.get_partial()[::-1].__repr__()
        return(s)

class RingBufferFull(RingBuffer):
    def append(self, value):
        """append an element when buffer is full"""
        self._data = np.roll(self._data, 1)
        self._data[0] = value

def naiveWeightedSmooth(y, weights, window, **kw):
    smthW=smooth(weights, window, **kw)
    smthY=smooth(y, window, **kw)

    return smthY/smthW
def weighted_smooth_win(y, w, window, mode='same'):
    smthW=np.convolve(w, window, mode=mode)
    smthY=np.convolve(y*w, window, mode=mode)/smthW
    if mode == 'valid':
        #Inds=r_[N/2-1:y.size-N/2]
        #msk=ones(w.size, dtype=bool)
        #msk[:N/2-1]=msk[-N/2:]=0
        slc=slice(N/2-1, y.size-N/2)
        return smthY, smthW, slc
    return smthY, smthW
def weighted_smooth(y, w, N=10, mode='same'):
    smthW=np.convolve(w, np.ones(N, dtype='f8'), mode=mode)
    smthY=np.convolve(y*w, np.ones(N, dtype='f8'), mode=mode)/smthW
    if mode == 'valid':
        #Inds=r_[N/2-1:y.size-N/2]
        #msk=ones(w.size, dtype=bool)
        #msk[:N/2-1]=msk[-N/2:]=0
        slc=slice(N/2-1, y.size-N/2)
        return smthY, smthW, slc
    return smthY, smthW


def funcSmooth(x, y, windowFunc, Npts):
    """ Smooth the array y using some varying window, windowFunc(x) that varies with x.

    >>> import numpy as np
    >>> f=lambda x: np.ones(x.size, dtype='f8')/x.size
    >>> t= np.arange(10);
    >>> y= np.ones(10);
    >>> y[::2]=0; # array of [0,1,0,1]
    >>> funcSmooth(t, y, f, 2)
    array([ 0. ,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5])
    >>> f2=lambda x: x
    >>> funcSmooth(t, y, f2, 2)
    array([ 0.,  1.,  1.,  3.,  3.,  5.,  5.,  7.,  7.,  9.])
    """
    out=np.empty(x.size)
    #inds0=np.r_[-Npts/2:Npts/2]
    indRange0 =array([-Npts/2, np.ceil(Npts/2)],dtype='i8')
    NptsOn2=Npts/2
    for k in xrange(x.size):
        #inds= inds0.astype('i8')+k
        indRange=indRange0+k
        if indRange[0]<0:
            indRange[0]=0
        if indRange[1] > x.size:
            indRange[1]=x.size
        #print(indRange)
        slc=slice(*indRange)
        env= windowFunc(x[slc])
        #env/= env.sum()
        out[k]= ( y[slc]*env ).sum()

    return out

def sliceSorted(t, t_0, t_end=None, delta_t=None):
        """ Cut up the times @t into sections that start at t_0 and and at t_end OR t_0 + delta_t if they're all the same.

        Returns a list of index arrays with length t_0.size
        """
        startIs=t.searchsorted(t_0.ravel()); #ravel here because x0 can be multi-d
        if t_end is not None:
            endIs = t.searchsorted(t_end.ravel())
            if delta_t is not None:
                raise Exception("should have t_end OR delta_t, not both!")
        elif delta_t:
            endIs = t.searchsorted(t_0.ravel()+delta_t)

        startIs=startIs[endIs<t.size] #Throw out cuts where the end index is out of range
        endIs=endIs[:startIs.size]

        if t_0.ndim>1:
            #If we lose some part so we have an incomplete sequence, we'll dump them.
            Norphans=startIs.size%t_0.shape[-1] #this is only general for 2d- need rethinking for more dimensions
            if Norphans>0:
                startIs=startIs[:-Norphans]
                endIs=endIs[:-Norphans]

            newShape=[int(startIs.size/t_0.shape[-1]), t_0.shape[-1]]
        else: 
            newShape=[endIs.size]

        indxL=empty(newShape, dtype='O')
        indxL.ravel()[:]=[np.r_[sI:eI] for sI,eI in zip(startIs, endIs)]

        return indxL

class DummyObj(object):
    class_name =""
    bInited = False
    def __init__(self, class_name="unnamed"):
        self.class_name=class_name
    def __getattr__(self, name):
        if not self.bInited:
            raise ValueError("DummyObj {} has not been iniied!".format(self.class_name))
        def method(*args, **kwargs):
            print(self.class_name + " called " + name)
            if args:
                print("it had arguments: " + str(args))
            if kwargs:
                print("it had kwargs: " + str(kwargs))
        return method
    def __call__(self, *args, **kwargs):
        if self.bInited:
            raise ValueError("Dummy Obj {} has been inited twice!".format(self.class_name))
        self.bInited=True
        print("initialising dummy obj {}".format(self.class_name))
        return self

    
    


if __name__=='__main__':
    import doctest
    doctest.testmod()
