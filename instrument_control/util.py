
#from fmtw import fft1n
from __future__ import print_function, division
import time
import pylab as pl
import matplotlib
from numpy.fft import fft, ifft
import numpy as np
import scipy as sp
from scipy import random
from scipy.interpolate import UnivariateSpline
from scipy.signal import wiener, filtfilt, butter, gaussian
from scipy.ndimage import filters
from functools import partial
import inspect
import pdb
import os
from numpy import iterable, array, pi, nan
import functools
from collections import deque
from itertools import islice
from bisect import insort, bisect_left
import copy
import warnings
from contextlib import contextmanager
from inspect import currentframe, getouterframes

bVerbose=False


from scipy import linalg as la
def Spre(op):
    op = np.array(op)
    opN=op.shape[0]
    #z=np.zeros(2*[opN**2])
    return np.matrix(la.block_diag( *(opN*[op]) ))
    #return np.block([[op,z], [z,op]])
def Spost(op):
    op=np.array(op)
    N=op.shape[0]
    blocky = (op.ravel()[:,np.newaxis, np.newaxis]*np.array(N**2*[np.eye(N)])).reshape(N,N,N,N)
    M = np.block([[*el] for el in [*blocky]]).T
    return np.matrix(M)

def herm_c(op):
    return np.conjugate(op.T)
def cmpPrePost(op):
    import qutip as q
    print(Spre(op.data.todense()) - q.superoperator.spre(op).data.todense())
    print(Spost(op.data.todense()) - q.superoperator.spost(op).data.todense())

def liouvillian(op, cOpL=[]):
    if op is not None:
        L = -1j*(Spre(op) - Spost((op)))

    else:
        N=cOpL[0].shape[0]
        L = np.matrix(np.zeros((N**2,N**2), dtype='object'))
    for cOp in cOpL:
        cdc = herm_c(cOp)*cOp
        L += Spre(cOp)*Spost(herm_c(cOp)) -\
            0.5*Spre(cdc) -0.5*Spost(cdc) 
    return L

def pruneRagged(arrL, axis=-1, maxPts=None):
    lengths=[ar.shape[axis] for ar in arrL]
    newN=np.min(lengths)
    if maxPts is not None and newN<np.max(lengths)-maxPts:
        raise ValueError("Array is just too ragged");
    pruned=np.array([ar[:newN] for ar in arrL])
    return pruned

@contextmanager
def let(**bindings):
    frame = getouterframes(currentframe(), 2)[-1][0] # 2 because first frame in `contextmanager` decorator  
    locals_ = frame.f_locals
    original = {var: locals_.get(var) for var in bindings.keys()}
    locals_.update(bindings)
    yield
    locals_.update(original)

def splitAtGaps(t, x=None, gapFact=2):
    """ Split the steadily changing array @t assuming that big jumps in value mean missing data. The array @x (same shape as @t), if provided, will be split divided at the same points as t

    Example:
    >>> arr=np.hstack([np.arange(5), np.arange(5)+9, np.arange(4)+20])
    >>> arr
    array([ 0,  1,  2,  3,  4,  9, 10, 11, 12, 13, 20, 21, 22, 23])
    >>> x= np.arange(arr.size)
    >>> tSplit, xSplit=splitAtGaps(arr, x)
    >>> tSplit
    [array([0, 1, 2, 3, 4]), array([ 9, 10, 11, 12, 13]), array([20, 21, 22, 23])]
    >>> xSplit
    [array([0, 1, 2, 3, 4]), array([5, 6, 7, 8, 9]), array([10, 11, 12, 13])]

    """
    tDiff=np.diff(t)
    medDt=np.median(tDiff)
    gapI=np.hstack([ [0], np.where(tDiff>gapFact*medDt)[0]+1, [t.size] ])
    #gapStartT=t[gapI]; 
    tSplit=[t[gapI[k]:gapI[k+1]] for k in range(gapI.size-1)]
    if x is not None:
        xSplit=[x[gapI[k]:gapI[k+1]] for k in range(gapI.size-1)]
        return tSplit, xSplit
    return tSplit

def vprint(*args, **kwargs):
    global bVerbose
    if bVerbose:
        print(*args)#, **kwargs)

def figDontReplace(labelIn=None, **kwargs):
    if labelIn is None:
        pl.figure(**kwargs)
    else:
        figlabels=pl.get_figlabels();
        label=labelIn
        k=0
        while label in figlabels:
            label= labelIn + ' -' + str(k)
            k+=1
        pl.figure(label, **kwargs)

def cardinalToDeg(st):
    D=dict(
           N=0,
           NE=45,
           E=90,
           SE=135,
           S=180,
           SW=225,
           W=270,
           NW=315,
           )
    if D.has_key(st):
        return D[st]
    else:
        raise ValueError("Don't know hwo to convert the string '{0}' to an angle".format(st))

def const_interp(x, xp, yp, offs=1):
    I=xp.searchsorted(x)-offs
    I=np.where(I>=xp.size, xp.size-1, I)
    I=np.where(I<0, 0, I)
    return yp[I]

def open_file_external(filepath):
    import subprocess, os, sys
    if sys.platform.startswith('darwin'):
        subprocess.call(('open', filepath))
    elif os.name == 'nt':
        os.startfile(filepath)
    elif os.name == 'posix':
        subprocess.call(('xdg-open', filepath))

def ssqe(sm, s, npts):
    return np.sqrt(np.sum(np.power(s-sm,2)))/npts

def testSpline(x, y, s, npts):
    sp = UnivariateSpline(x, y, s=240)
    plt.plot(x,sp(x))
    print("splerr", ssqe(sp(x), s, npts))
    return sp(x)
def gaussFilt(x, y, s, npts, std_ratio=3.):
    b = gaussian(npts, npts/std_ratio)
    ga = filters.convolve1d(y, b/b.sum())
    #plt.plot(x, ga)
    #print "gaerr", ssqe(ga, s, npts)
    return ga
def get_fwd(fobj):
    """Get file working directory
    """
    d=os.path.dirname(fobj)
    if d=='':
        d='./'
    else:
        d=d+'/'
    return d
#os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
def invertD(D):
    inv_map = {v: k for k, v in D.items()}
    return inv_map

def draw_fat_arrow(apex, width, height=None,head_space=None, ax=None,**kw):
    if ax==None:
        ax=pl.gca()
    if height==None:
        height=width*1

    head_height=height/4.
    if head_space==None:
        head_space=width*0.15
    head_width=width+head_space*2;
    
    head_points=pl.array([ 
        apex, 
        [apex[0]+head_width/2,apex[1]+head_height],
        [apex[0]-head_width/2,apex[1]+head_height], 
        apex,
        ])

    body_points=pl.array([
        [apex[0]+width/2,apex[1]+head_height],
        [apex[0]+width/2,apex[1]+head_height+height], 
        [apex[0]-width/2,apex[1]+head_height+height],
        [apex[0]-width/2,apex[1]+head_height], 
        ])

    all_points=pl.vstack([head_points[:2], body_points, head_points[2:]])
    
    arrow=pl.Polygon(all_points, **kw)
    return ax.add_patch(arrow)

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

def isApproxMultiple(val, factor, tol=1e-4):
    return ((val+tol)%factor)<2*tol

def rotate_quadrature_sample(angles_in, means_in, cov=None, bVarNotCovar=False, rotateTo=0, debug=False):
    """ Rotate samples if it makes sense to do so. Will return None instead of spitting out all nans

    Example:
    >>> from numpy import nan, pi
    >>> angles=[pi/2,5*pi]
    >>> means=[nan, 0.1]
    >>> cov=[[nan, nan], [nan, 1.0]]
    >>> rotate_quadrature_sample(angles,means,cov)
    (array([  0.,  nan]), array([-0.1,  nan]), array([[  1.,  nan],
           [ nan,  nan]]))
    #angles, means, cov
    """
    if cov is not None and np.all(np.isnan(cov)):
        return array([nan, nan]), array([nan, nan]), cov
    angles=array(angles_in).copy()
    means=array(means_in).copy()
    flag=False
    dth=angles[1]-angles[0]
    if dth<0:
        angles[1]+=pi
        means[1]*=-1
    if any(np.isnan(means)):
        if all(np.isnan(means)):
            return None
        mask=~np.isnan(means) 
        angle, mean= angles[mask][0], means[mask][0]
        angle, mean=wraptopi(angle, mean) #now in the 0->pi range
        finMeans=np.array([np.nan,np.nan])
        finAngles=np.array([np.nan,np.nan])
        if cov is not None:
            cov=array(cov)
            #if cov.ndim==1: #make a covariance matrix if there is none
                #cov=np.diag(cov)
                #cov[0,1]=np.nan
                #cov[1,0]=np.nan
            covVal=cov[~np.isnan(cov)][0]
            finCov=np.array(np.empty([2,2])*np.nan)
        if nearby_angles(angle, pi/2, 0.05) and 0:
            I=1
        else:
            I=0
        finMeans[I]=mean
        finAngles[I]=angle
        if mean<-1 and debug and 0:
            pdb.set_trace()
            
        if cov is not None:
            finCov[I,I]=covVal
        

    else:
        def rotM(theta):
            return np.array([[cos(theta), sin(theta)],
             [-sin(theta), cos(theta)]
             ])
        #If covarainces...
        R=rotM(-angles[0]+rotateTo)
        finMeans=R.dot(array(means))
        finAngles=array(angles)- angles[0] +rotateTo
        if cov is not None:
            if not bVarNotCovar:
                finCov=R.dot(cov).dot(R.T)
            else:
                finCov=R.dot(cov)
            finAngles, finMeans=wraptopi(finAngles,finMeans)

        dth=finAngles[1]-finAngles[0]
        if dth<0:
            finAngles[1]+=pi
            finMeans[1]*=-1

    if flag:
        print("Rotated {}:{} to {}:{}".format(angles_in%(2*pi),means_in, finAngles,finMeans))
    if cov is not None:
        return finAngles, finMeans, finCov
    else:
        return finAngles, finMeans
        

def weightedStats(values, weights,axis=-1, bReturnChi2=False,ddof=1):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    weights[np.isinf(weights) | np.isnan(weights) | np.isinf(values) | np.isnan(values)]=0
    values[np.isinf(values) | np.isnan(values)]=0
    N=len(values)
    average = np.average(values, weights=weights,axis=axis)
    variance = np.average((values-np.expand_dims(average, axis))**2, weights=weights,axis=axis)*N/(N-ddof)  # Fast and numerically precise
    unc = variance#/(N-1)
    #errBarUnc=1./sum(1/errs**2)**0.5
    errBarUnc=1./sum(weights)**0.5
    #chi2= np.average((values-np.expand_dims(average, axis))**2, weights=weights,axis=axis)*weights.sum()/(N-1)

    if bReturnChi2:
        chi2=sum(((values-np.expand_dims(average, axis))**2)*weights)/(N-1)
        #return (average, np.sqrt(variance), np.sqrt(unc), chi2) 
        return (average, errBarUnc, np.sqrt(unc), chi2) 
    #return (average, np.sqrt(variance), np.sqrt(unc)) 
    return (average, errBarUnc, np.sqrt(unc)) 

def cooksDistance(values, weights=None, fitFunc=None, Np=1):
    N=len(values)
    mn0, var0= fitFunc(values, weights)
    msk=np.ones(N, dtype='bool')
    cooksDL=[]
    resL=[]
    errL=[]
    for k in xrange(len(values)):
        msk[k]=0
        vals=values[msk]
        Ws=weights[msk]
        mn_k, var_k=fitFunc(vals, Ws) 
        cooksDL.append(sum((values-mn_k)**2)/var0/Np)
        resL.append(mn_k)
        errL.append(var_k)
        msk[k]=1
    return array(cooksDL), array(resL), array(errL)
    


def combine_angle_pairs(angles, means, covs):
    """ Combine samples at angles 'angleL' with means 'meanL' and covariance 'covL[k]' to give an estimate of a mean, a covariance, and ideally a chi-sq paramter
    Algorithm(?): rotate covariance matrices to the same angle (i.e. 0, 90) and average them.
    For single samples (no covariance), we can (try) just averaging the terms we know about? Or just ignore them for the moment.

    Examples:
    >>> angle1, val1, cov1= [pi/2,pi], [sqrt(2), -sqrt(2)], [[0.5, 0.1], [0.1,1.5]]
    >>> angle2, val2, cov2= [0, pi/2], [sqrt(2), sqrt(2)], [[0.3, 0.1], [0.1,1.2]]
    >>> combine_angle_pairs([angle1, angle2], [val1, val2], [cov1, cov2])
    (array([[ 1.57079633,  3.14159265],
           [ 0.        ,  1.57079633]]), array([ 0.77643098, -0.73945807]), array([[ 0.24411765,  0.0127451 ],
           [ 0.0127451 ,  0.34738562]]))
    """
    angles=array(angles)
    means=array(means)
    covs=array(covs)
    finMean=0;
    rotated=zip(*[rotate_quadrature_sample(*p) for p in zip(angles, means, covs)])
    Ws=pl.inv(rotated[2])
    Sigma_x=pl.inv(sum(Ws))
    #See Wikipedia pag on weighted mean with vector estimates
    finX=Sigma_x.dot( sum([W.dot(x) for W, x in zip(Ws, rotated[1]) ] )  )
    #mn, er= weightedStats(ys, errs**2, axis=0)

    return angles, finX, Sigma_x
    

    #if single samples:
def wraptopi(angles, vals, dev=0.1, debug=False):
    """Wrap an angle-value pair (or many angle-value pairs) to a multiple of pi

    Single value:
    >>> np.allclose(wraptopi(7*pi, 1), (0.0,-1))
    True

    Multiple values:
    >>> wraptopi([pi/4, 5*pi/5, 6.2*pi], [1,1,1])
    (array([ 0.78539816,  0.        ,  0.62831853]), array([ 1, -1,  1]))
    """
    if debug:
        pdb.set_trace()
    vals=array(vals)
    angles=array(angles)
    angles2pi=angles%(2*pi)
    wrapped_angles=(angles2pi+dev)%pi - dev
    ch=abs(wrapped_angles-angles2pi)
    wrapped_vals=np.where((ch>2*pi-dev) | (ch<dev/2), vals, -vals)
    if not wrapped_vals.shape: #if vals was a single value, the shape will be 0d (i.e.. will be an empty tuple).
        wrapped_vals=wrapped_vals.item()
    return wrapped_angles, wrapped_vals

def fill_around(x,y,dy, **kwargs):
    pl.fill_between(x,y-dy,y+dy, **kwargs)


def plot_with_err(th,x,err,**kwargs):
    new_kwargs=dict(alpha=0.5)
    new_kwargs.update(kwargs)
    plot(th,x)
    fill_between(th,x+err,x-err,alpha=0.3, **kwargs)


class dataset(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)
#The below is to be used to pickle these things as dictionaries, rather than classes. Cos classes pickle all dodgy.
class cont(object):
    """ A random container object
    """
def cont2dict(cnt):
    d= cnt.__dict__
    d['is_cont']=True;
    return d

def dict2cont(d):
    cnt=cont()
    cnt.__dict__.update(d)
    return cnt
    
def rec_conts2dict(D):
    for key in D.keys():
        if type(D[key])==dict:
            rec_conts2dict(D[key])
        elif type(D[key]).__name__=='cont':
            D[key]=cont2dict(D[key])

    return D

def rec_dict2conts(D):
    for key in D.keys():
        if type(D[key])==dict:
            if D[key].has_key('is_cont'):
                D[key]=dict2cont(D[key])
            else:
                rec_dict2conts(D[key])

    return D

def conv_along_axis(x1,x2,axis=1):
    """Supposedly an fft based convolve that is fast along two dimensions- but seems to be quicker just to loop through axes and use regular convolve??

    Dodgy hack- axis is actually ignored, assumed to be convolving along the second axis (ie axis=1)
    """
    if 0:
        N=x2.shape[axis] 
        topad=N-x1.size
        padleft=topad/2
        padright=topad/2
        if topad/2*2!=topad:
            padright+=1
        x1new=pl.hstack([zeros(padleft),x1,zeros(padright)])
        f2=fft(x2,axis=axis)
        f1=fft(x1new)
        F=f1*f2
        cnved=ifft(F)
    
    other_axis=0 #maybe
    out=pl.empty(x2.shape,x2.dtype)
    Ntimes=x2.shape[0]

    for k in xrange(Ntimes):
        out[k]=pl.convolve(x1,x2[k])

    return out

def get_err_func(base_err, base_args,*args, **kwargs):
    """Return a wrapper around a function so that it may be used as an error function for optimsation.

    
    Returns func, p0 where func is the wrapped function and p0 are intial values
    Currently only works for functions that have named parameters- these are named in kwarg to_vary

    Would be great if it could take some mix.
    """
    if kwargs.has_key('to_vary'):
        names_to_vary=kwargs.pop('to_vary')
        to_vary={}.fromkeys(names_to_vary)
        for name in names_to_vary:
            if name in kwargs.keys():
                to_vary[name]=kwargs.pop(name)
    to_fix=kwargs #the remainder should be fixed
    
    aspec=inspect.getargspec(base_err)  
    base_defs=aspec.defaults
    base_kwargs=dict(zip(aspec.args[-len(base_defs):], base_defs) )
    for name in to_vary:
        if to_vary[name]==None:
            to_vary[name]=base_kwargs[name]
        
    part_func=partial(base_err, *base_args, **to_fix)
    def func(p):
        d={}
        d.update(zip(names_to_vary, p))
        print("params: {0}".format(d))
        sys.stdout.flush()
        return part_func(**d)
    
    p0=[to_vary[name] for name in names_to_vary]
    test=func(p0)
    return func,p0


from numpy import convolve
def sumNormLorentzian(t,p):
    """Lorentzian normalised to the sum of the points in it, for use in MBEsim mainly
    p[0]=fwhm
    p[1]=centre


    """

    out= p[0]/(2.*np.pi)*  1./(  (t-p[1])**2   +   (p[0]/2.)**2.  )
    dt=t[1]-t[0]
    return out*dt#/(out.sum()*dt)

def simpleLor(t,p):
    '''
    p[0]=fwhm,
    p[1]=centre
    peak is at 1
    
    >>> t=np.linspace(-5,5,100)
    >>> y=simpleLor(t,[2.,1.] )
    >>> #confirmplot(t,y)
    '''
    
    return (p[0]/2)**2/ (    (t-p[1])**2   +   (p[0]/2)**2 )


def serrodyne(t, frequency, amplitude=1):
    y=(t*float(frequency))%1
    return (y-0.5)*2

def amp_modulate(t, amp, freq, base_band=0, bias=np.pi/2, chirp_fact=0):
    if len(base_band)>1:
        base = base_band
    else:
        base= exp(1j*2*np.pi*t*base_band)
    return sin(bias + 2*np.pi*t*freq)*base*exp(1j*2*pi*t*chirp_fact*amp*freq)
    

def serrodyne_chirp(t, start_freq, stop_freq, scaling=None):
    span=stop_freq-start_freq;
    y=serrodyne(t*np.linspace(start_freq, start_freq+span/2, t.size), 1.0)
    if scaling==None:
        return y
    else:
        return y*scaling

def chirp(t, start_freq, stop_freq, tStart=None, tStop=None, phase=0, bComplex=False):
    Istart= 0 if tStart is None else t.searchsorted(tStart)
    Iend= -1 if tStop is None else t.searchsorted(tStop)
    slc=slice(Istart,Iend)
    y= np.zeros(t.size)
    tRel = t[slc]
    span = stop_freq-start_freq
    chirpiness = span/(t[Iend] - t[Istart])
    yRel = cos(tRel*2*pi*(start_freq + chirpiness/2*tRel) + phase)
    y[slc] = yRel
    if bComplex:
        y= y+ 1j*chirp(t,start_freq,stop_freq,tStart,tStop, phase=phase-pi/2,
            bComplex=False)
    return y


def sawtooth(t, period):
    y= (t/float(period))%1
    return y

def triangle(t, period):
    dy=np.where( (t/float(period))%1 <0.5, 1., -1.)
    y=np.cumsum(dy)
    y/=(y.max()/2.)
    y-=1.
    return y

def square_wave(t, frequency, duty_cycle=0.50):
    return (t%(1./frequency)<duty_cycle*1./frequency)
    

def heaviside(x):
    out=pl.zeros(x.shape)
    out[pl.where(x>0)]=1;
    out[pl.where(x==0)]=0.5;
    
    return out
#from math import round
def tophat(x,width,centre=0, bDigitizeFix=False):
    """Top hat function
    @bDigitizeFix: always round off the number of high points to prevent inconsistent digitizing
    """
    if bDigitizeFix:
        #Assume consistent x spacing
        Istart=x.searchsorted(centre-width/2)
        Npts=int(round(float(width)/(x[Istart+1]-x[Istart])))
        out=zeros(x.shape, dtype='f8')
        out[Istart:Istart+Npts]=1.0
        return out

    if iterable(x):
        return heaviside(array(x)-centre+width/2)-heaviside(array(x)-centre-width/2)
    else:
        return 0 if abs(x-centre)*2 > width else 1

def smoothhat(x,width,lorwidth,centre=0):
    rind=pl.where( x > width/2+centre )[0]
    lind=pl.where( x < -width/2+centre  )[0]
    mind=pl.where( (x > -width/2+centre) &  (x < width/2+centre) )[0]
    
    out=pl.ones(x.size)
    
    out[rind]=simpleLor( x[rind], [lorwidth, width/2+centre] )
    out[lind]=simpleLor( x[lind], [lorwidth, -width/2+centre] )
    return out

def lophat(x,width, lwidth,rwidth,lbg=0,rbg=0.1,centre=0):
    '''
    Almost a top-hat convoluted with a lorentzian, but lop-sided.
    Really it's just a top hat centred at centre, with tacked-on half-lorentzians with lwidth at the left, and rwidth at the right.
    lbg and rbg are the zero levels on the left and right respectively.
    '''
    rind=pl.where( x > width/2+centre )[0]
    lind=pl.where( x < -width/2+centre  )[0]
    mind=pl.where( (x > -width/2+centre) &  (x < width/2+centre) )[0]
    
    out=pl.ones(x.size)
    
    out[rind]=(1-rbg)*simpleLor( x[rind], [rwidth, width/2+centre] )+rbg
    out[lind]=(1-lbg)*simpleLor( x[lind], [lwidth, -width/2+centre] )+lbg
    return out

from scipy.signal import fftconvolve
def get_k_real(fax,imag_part):
    '''
    function to get the real part of the response from the imaginary part of the response.
    ab should be -ve imaginary, and in lineary units of response to get the results you probably expect.
    '''
    df=fax[1]-fax[0]
    imp= -1j*imag_part

    kern=2./(1j*fax)
#   if imp.ndim==2:
#       kern=kern[:,np.newaxis] * np.ones(imp.shape)
    if imp.ndim==2:
        out=[]
        for y in imp.T:
            out.append(convolve(y,kern,'same')*df/2/np.pi)
        real_bit=array(out).T
    else:
        real_bit=convolve(imp,kern,'same')*df/2/np.pi#/sum(imag_part)
#   real_bit=apply_along_axis(convolve,0,imp,[kern,'same'])
#   real_bit=fftconvolve(imp,kern,'same')*df/2/np.pi#/sum(imag_part)
    return real_bit

from numpy import cos,sin,zeros 
def rotation(th_z_x_z):
    '''
    Make a rotation matrix, which, if  I remember correctly, is based on rotation about the z axis, followed by rotation about x, then about z again.
    '''
    cx,cy,cz = cos(th_x_y_z)
    sx,sy,sz = sin(th_x_y_z)
    R=zeros((3,3))
    R.flat = (cx*cz - sx*cy*sz, cx*sz + sx*cy*cz, sx*sy,
        -sx*cz - cx*cy*sz, -sx*sz + cx*cy*cz,
        cx*sy, sy*sz, -sy*cz, cy)
    return R

def rot3dS(th_a_b_g):
    '''
    Make a rotation matrix, which, if  I remember correctly, is based on rotation about the z axis, followed by rotation about x, then about z again.
    '''
    from sympy import cos, sin, Matrix
    ca,cb,cg = [cos(th) for th in th_a_b_g]
    sa,sb,sg = [sin(th) for th in th_a_b_g]

    R=zeros((3,3))
    R = Matrix([[cg*cb*ca - sg*ca, cg*cb*sa + sg*ca, -cg*sb,],
        [-sg*cb*ca-cg*sa,  -sg*cb*sa +cg*ca, sg*sb,],
        [sb*ca, sb*sa, cb]])
    #R.flat = (cx*cz - sx*cy*sz, cx*sz + sx*cy*cz, sx*sy,
    #    -sx*cz - cx*cy*sz, -sx*sz + cx*cy*cz,
    #    cx*sy, sy*sz, -sy*cz, cy)
    return Matrix(R)

def rot3dSAx(theta, u):
    from sympy import cos, sin, Matrix, Integer
    u=Matrix(u)
    u=u/(u.dot(u))
    cth=cos(theta)
    sth=sin(theta)
    ux,uy,uz=u
    O=Integer(1)
    T=Integer(2)
    R=Matrix([[cth + ux**2*(O-cth), ux*uy*(O-cth)- uz*sth, ux*uz*(O-cth) + uy*sth],
            [uy*ux*(O-cth)+uz*sth, cth+uy**2*(O-cth), uy*uz*(O-cth) - ux*sth],
            [uz*ux*(O-cth) - uy*sth, uz*uy*(O-cth)+ux*sth, cth+uz**2*(O-cth)],
            ])
    return R

from numpy import sqrt,array,arccos,empty
def cart2sph(v):
    '''
    Cartesian to spherical coordinates.
    v: (x,y,z)
    output: [r,theta,phi], where theta is angle from +z axis, phi is angle from +x axis (toward +y)
    '''
    r=sqrt(v[0]**2+v[1]**2+v[2]**2) 
    vn=array(v)/r
    theta=arccos(vn[2])
    
    if theta==0:
        phi=0
    else:
        phi=arccos(vn[0]/sin(theta))
        if v[1] <0:
            phi=-phi
    

    u=empty(v.shape);
    u[0]=r 
    u[1]=theta
    u[2]=phi
    return u

def rmatrix_from_vectors(v1,v2):
    '''
    Returns a transformation matrix to take vector v1 onto v2. They are each 3d (x,y,z) vectors. For u1, u2 =(r1:r2, theta1:theta2, phi1:phi2),
    We rotate v1 (lefthanded) about +z by phi-pi/2, bringing it to the +y,z plane. Then rotate about +x (lefthanded) by theta2-theta1, getting the final elevation correct.
    We finally rotate about the z axis again (lefthanded) by (-phi2-pi/2), rotating to the +x-axis then to the azimuthal angle of v2
    '''
    v1=array(v1,dtype='f8')
    v2=array(v2,dtype='f8')
    if len(v1) != len(v2) and len(v1)==3:
        raise "input vectors must be shape (3,)!"
    u1=cart2pol(v1)
    u2=cart2pol(v2)
    dTheta=u2[1]-u1[1]
    
    print [-u2[2]+pi/2,dTheta,u1[2]-pi/2]

    return u2[0]/u1[0]*rotation([-u2[2]+pi/2,dTheta,u1[2]-pi/2])

def rotMat2d(theta):
    return np.matrix([[cos(theta), -sin(theta)],
                   [sin(theta),  cos(theta)]])

def sph2cart(phi,theta,r):
    '''
    Spherical to cartesian coordinate:
    Seems a funny convention though-

    @Parameters:
    ------------
    phi: angle from -z axis?
    theta: angle from +x axis
    r: length
    '''
    x=r*np.cos(phi-pi/2)*np.cos(theta)
    y=r*np.cos(phi-pi/2)*np.sin(theta)
    z= -r*np.cos(phi)

    return x,y,z
def reflection_mat3d(vec):
    vec=np.array(vec)/np.linalg.norm(vec)
    a,b,c=vec
    A=np.matrix([  [1-2*a**2, -2*a*b, -2*a*c],
                [-2*a*b, 1-2*b**2, -2*b*c],
                [-2*a*c, -2*b*c, 1-2*c**2]
                ])
    return A
    #N= matrix(vec) #<- gives a row matrix (not column), so the transposes below are swapped
    #return matrix(identity(3)) - 2*N.T*N

def pairSort(x,y,kind='quicksort'):
    '''
    Function to sort x and y by x, returning x and y sorted by x.
    I don't know of a good way to do this in numpy except to combine x and y into a record array, then sort it by
    the first field (ie x). I then split it up again for the output

    ... or maybe I just use lexsort!!
    '''

    ind=np.lexsort((y,x))
    return x[ind],y[ind]
    
    #Old way:
    xy=np.empty(x.size,dtype=[('f1',x.dtype),('f2',y.dtype)])
    xy['f1']=x
    xy['f2']=y
    xy.sort(order='f1',kind=kind)
    #sd=np.sort(xy,order='f1',kind=kind)
    #return sd['f1'], sd['f2']
    return xy['f1'],xy['f2']

def genIntegrate(x,y):
    '''
    Function to take the normal scipy integrators (eg. trapz rule) and apply it to unsorted data.
    This function should just sort the data first, then apply the rule
    '''
    
    from scipy import integrate as ig
    xy=zeros(x.size,dtype=('float64,float64'))
    xy['f0']=x
    xy['f1']=y
    xy.sort(order='f0');
    igted=ig.trapz(xy['f1'],xy['f0'])
    return igted, locals()

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

def running_variance(y, N):
    mnY=smooth(y,N)
    runVar=smooth((y-mnY)**2, N)*N/(N-1)
    return runVar

from scipy.optimize import curve_fit
def nancurve_fit(f, xdata, ydata, p0=None, sigma=None, **kw):
    nansI=np.isnan(ydata)
    if sigma is not None:
        sigmaNoNan =sigma[~nansI]
    else: sigmaNoNan=None
    xnonan=xdata[~nansI]
    ynonan=ydata[~nansI]
    if ynonan.size==0:
        return np.array(p0)*0, np.ones((len(p0), len(p0)))*np.inf
    return curve_fit(f, xnonan, ynonan, p0=p0, sigma=sigmaNoNan, **kw)
    
def nancov(x):
    """ Assume x has shape (N,M) where N is number of samples and M is number of variables
    """
    nans=np.isnan(x)
    nanRows=np.any(nans, axis=0)
    return np.cov(x[:,~nanRows])

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> y=np.arange(6,dtype='f4')# linear interpolation of NaNs
        >>> y[2]=np.nan
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def nansmooth(x,window_len=10,window='flat'):
    x=naninterp(x, bCopy=True)
    return smooth(x, window_len, window)

def naninterp(x, bCopy=True, axis=0):
    if not np.any(~np.isnan(x)):
        return np.nan
    if bCopy:
        x=x.copy()
    nans, x1= nan_helper(x)
    x[nans]= np.interp(x1(nans), x1(~nans), x[~nans])
    return x

def findOutliers(x, windowLen=50, sigmas=3, bReturnMeanLine=False ):
    """Find outliers from the general trend of data @x.

    @x is first smoothed (and nans interpolated) with a moving average of length @windowLen, and any points which deviate from this line by more than 3 @sigma are flagged.
    """
    xSM=nansmooth(x, windowLen)
    sigma=np.nanmean(abs(x-xSM))
    outlierM=abs(x.ravel()-xSM) > sigmas*sigma
    if not bReturnMeanLine:
        return outlierM
    else:
        return outlierM, xSM

def normGauss(t,sig):
    return 1./np.sqrt(2*np.pi*sig**2)*pl.exp(   -t**2 / (2*sig**2) )

def voigtMil( wavenumberArray,centerLine,widthGauss, widthLorentz ):
    # voigt  Calculation of the VOIGT profile 
    #
    #   [y] = voigt( wavenumberArray,centerLine,widthGauss, widthLorentz )
    #   The function calculates the Voight profile using the algorithm 
    #   kindly provided by Dr. F. Schreier in FORTRAN and rewritten to MATLAB
    #   by Dr. N. Cherkasov
    # 
    #   For more details on algorithm see the publication:
    #   F. Schreier: Optimized Implementations of Rational Approximations for the Voigt ane Complex Error Function. 
    #   J. Quant. Spectrosc. & Radiat. Transfer, 112(6), 10101025, doi 10.1016/j.jqsrt.2010.12.010, 2011. 
    #
    #
    #   INPUT ARGUMENTS
    #       wavenumberArray - array 1*N of wavenumbers 
    #       centerLine - position of the band center
    #       widthGauss - parameter of the width of the Gaussian component (Half-width at half maximum)
    #       widthLorentz - parameter of the width of the Lorentzian component (Half-width at half maximum)
    #
    # 	OUTPUT
    #       y - array 1*N of intensities
    #
    # The function was used for the deconvolution of IR spectra
    # see the publication
    #
    # 27-December-2013 N. Cherkasov
    # Comments and questions to: n.b.cherkasov@gmail.com


    # converting to dimensionless coordinates
    x=sqrt(log(2))*(wavenumberArray-centerLine)/(widthGauss);
    y=sqrt(log(2))*(widthLorentz/widthGauss);

    w=complexErrorFunction(x,y);
    y=sqrt(log(2)/pi)/widthGauss*real(w);

    return y
from scipy.special import wofz
def voigt(x, p):
    """Voigt profile

    p[0] is amplitude
    p[1] is lorentzian linewidth, gamma
    p[2] is gaussian width, sigma
    """
    return p[0]*np.real(wofz( ((x-p[3])+ 1j*p[1])/(np.sqrt(2.0)*p[2]) ))/(np.sqrt(2*pi)*p[2])

def gauss2d_abc(x, y, x0, y0, a,b,c ):
    """pars: [a b
              c d]
    """
    return np.exp(- 
            (a*(x-x0)**2 
            - 2*b*(x-x0)*(y-y0) 
            + c*(y-y0)**2) 
            )
    
def gauss2d(x, y, x0, y0, sig_1,sig_2,theta ):
    """pars: [a b
              c d]
    """
    a=cos(theta)**2/2/sig_1**2 + sin(theta)**2/2/sig_2**2
    b=-sin(2*theta)/4/sig_1**2 + sin(2*theta)/4/sig_2**2
    c=sin(theta)**2/2/sig_1**2 +cos(theta)**2/2/sig_2**2
    return gauss2d_abc(x,y,x0,y0, a,b,c)

def gauss(t,p):
    '''
    Gaussian function, t is axis, 
    p[0] is peak value.
    p[1] is position of peak.
    p[2] is fwhm.
    
    p[0]*pl.exp(   -pow(t-p[1],2)/( (p[2])**2/(4*pl.log(2)) )    ) 
    
    '''
    return( p[0]*pl.exp(   -pow(t-p[1],2)/( p[2]**2/(4*pl.log(2)) )    ) )

def naivegauss(t,p):
    '''
    Gaussian function, t is axis, 
    p[0] is peak value.
    p[1] is position of peak.
    p[2] is fwhm.
    
    p[0]*pl.exp(   -pow(t-p[1],2)/( p[2]/(4*pl.log(2)) )    ) 
    
    '''
    return( p[0]*pl.exp(   -pow(t-p[1],2)/( 2*p[2])    ) )

def lorentzianWithOffset(t,param):
    '''
    Lorentzian with vertical offset.
    p[0] is peak value.
    p[1] is fwhm
    p[2] is peak position
    p[3] is vert offset
    
    (param[0])* (param[1]/2)**2 /( (t-param[2])**2 + (param[1]/2)**2 ) +param[3]
    '''
    return (param[0])* (param[1]/2)**2 /( (t-param[2])**2 + (param[1]/2)**2 ) +param[3]

def lorentzian(t,param):
    '''
    Lorentzian with vertical offset.
    p[0] is peak value.
    p[1] is fwhm
    p[2] is peak position

    (param[0])* (param[1]/2)**2 /( (t-param[2])**2 + (param[1]/2)**2 )
    '''
    return (param[0])* (param[1]/2)**2 /( (t-param[2])**2 + (param[1]/2)**2 )

def sinc(t,param):
    '''
    sinc with horizontal offset.
    p[0] is peak value.
    p[1] is fwhm
    p[2] is peak position

    (param[0])* sin(param[1]*t) / t
    '''
    return param[0] * sin( (t-param[2])/param[1] )  /  ((t-param[2])*param[1])

def normLorentzian(t, param):
    '''
    Lorentzian with vertical offset.
    p[0] is fwhm
    p[1] is peak position

    (param[0])* (param[1]/2)**2 /( (t-param[2])**2 + (param[1]/2)**2 )
    '''
    return 1./2./np.pi*param[0] /( (t-param[1])**2 + (param[0]/2)**2 )

from scipy import random
def lorentzDist(N=1,fwhm=0.5):
    """Generate random numbers according to a lorentz distribution

    """
    val= 2*fwhm*np.tan(np.pi*(random.uniform(size=N)-0.5))
    if N==1:
        return val[0]
    else:
        return val
        

#Stuff for dealing with file/path names
def strip_extension(path):
    return path.rsplit('.',1)[0]
def filenameNum(st):
    '''This function will simply replace 'm' by '-', and 'p' by '.' in the
    input string, then attempt to convert it to a float. This because when I
    put numbers in file names I do that substitution the other way'''

    st=st.replace('m','-')
    st=st.replace('p','.');

    return float(st);

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
def plot_errorbar_and_hist(x, y, yerr, ax1=None,ax2=None, subplotSpec=None, sharexs=[None, None], **kwargs):
    if isinstance(ax1, matplotlib.gridspec.SubplotSpec):
        subplotSpec=ax1
    if ax1 is None or subplotSpec is not None:
        if subplotSpec is not None:
            gs=GridSpecFromSubplotSpec(1,4,subplot_spec=subplotSpec)
        else:
            gs=GridSpec(1,4)
        ax1=pl.subplot(gs[0,:3], sharex=sharexs[0])
        ax2=pl.subplot(gs[0,3], sharex=sharexs[1])
    gdInd=~np.isnan(y)
    ax1.errorbar(x, y, yerr, fmt='.', **kwargs)
    pl.sca(ax2)
    nbins=30
    if sum(gdInd)>1:
        pl.hist(yerr[gdInd], bins=nbins)

    return ax1, ax2
    
def running_nanmedian(seq, window_size):
	"""Contributed by Peter Otten"""
	data = array(seq, dtype=float)
	result = []
	for i in xrange(1, window_size):
		window = data[:i]
		result.append(np.nanmedian(window))
	for i in xrange(len(data)-window_size+1):
		window = data[i:i+window_size]
		result.append(np.nanmedian(window))
	return result

def running_median_insort(seq, window_size):
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    N=window_size-1
    result= [result[N]]*N + result[N:]
    return window_size*[result[0]]+result[window_size:]

def plot_scatter_and_hist(x, y, ax1=None,ax2=None, subplotSpec=None, sharexs=[None,None], scatD={}, **kwargs):
    if isinstance(ax1, matplotlib.gridspec.SubplotSpec):
        subplotSpec=ax1
    if ax1 is None or subplotSpec is not None:
        if subplotSpec is not None:
            gs=GridSpecFromSubplotSpec(1,4,subplot_spec=subplotSpec)
        else:
            gs=GridSpec(1,4)
        ax1=pl.subplot(gs[0,:3],sharex=sharexs[0])
        ax2=pl.subplot(gs[0,3], sharex=sharexs[1])
    ax1.scatter(x, y, **scatD)
    pl.sca(ax2)
    nbins=30
    gdInd=~np.isnan(y)
    if sum(gdInd)>1:
        pl.hist(y[gdInd], bins=nbins)

    return ax1, ax2

def nonnan(arr, *kw):
    if type(arr)== list:
        arr=array(arr)
    bdMsk=pl.isnan(arr)
    for arg in kw:
        bdMsk=bdMsk | pl.isnan(arg)
    out= [arr[~bdMsk]]+[arg[~bdMsk] for arg in kw]
    if not kw:
        out=out[0]
    return out

#######Decorators###############
import inspect
def memoize_inst_meth(obj):
    @functools.wraps(obj)# I think this makes it report the proper function signature?
    def memoizer(*args, **kwargs):
        inst=args[0]
        if not hasattr(inst, 'cacheD'):
            inst.cacheD={}
        objstr=str(obj)
        if not inst.cacheD.has_key(objstr):
            inst.cacheD[objstr]={}
        cache=inst.cacheD[objstr]
        key_args=''.join([str(val.func_code)+str(id(val)) if hasattr(val, 'func_code')  else str(val) for val in args])
        key_kwargs=''.join([str(val.func_code)+str(id(val)) if hasattr(val, 'func_code') else str(key)+':'+str(val) for key,val in kwargs.iteritems()])

        #key = str(args) + str(kwargs)
        key=key_args+key_kwargs
        #print('key: {}'.format(key))
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    #memoizer.cache=cache
    memoizer.orig_args=inspect.getargspec(obj).args
    return memoizer

class cached_property(object):
    '''Decorator for read-only properties evaluated only once within TTL period.

    It can be used to created a cached property like this::

        import random

        # the class containing the property must be a new-style class
        class MyClass(object):
            # create property whose value is cached for ten minutes
            @cached_property(ttl=600)
            def randint(self):
                # will only be evaluated every 10 min. at maximum.
                return random.randint(0, 100)

    The value is cached  in the '_cache' attribute of the object instance that
    has the property getter method wrapped by this decorator. The '_cache'
    attribute value is a dictionary which has a key for every property of the
    object which is wrapped by this decorator. Each entry in the cache is
    created only when the property is accessed for the first time and is a
    two-element tuple with the last computed property value and the last time
    it was updated in seconds since the epoch.

    The default time-to-live (TTL) is 300 seconds (5 minutes). Set the TTL to
    zero for the cached value to never expire.

    To expire a cached property value manually just do::

        del instance._cache[<property name>]

    '''
    def __init__(self, ttl=300):
        self.ttl = ttl

    def __call__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__
        self.__module__ = fget.__module__
        return self

    def __get__(self, inst, owner):
        now = time.time()
        try:
            value, last_update = inst._cache[self.__name__]
            if self.ttl > 0 and now - last_update > self.ttl:
                raise AttributeError
        except (KeyError, AttributeError):
            value = self.fget(inst)
            try:
                cache = inst._cache
            except AttributeError:
                cache = inst._cache = {}
            cache[self.__name__] = (value, now)
        return value

def nearby_angles(anglesIn, ref, proximity):
   """Check if @anglesIn are within @proximity of @ref (all in radians)
   """
   anglesShft=anglesIn+np.pi-ref 
   anglesShft=anglesShft%(2*np.pi)
   return abs(anglesShft-np.pi) < proximity

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

def nearly_equal(a,b,sig_fig=5):
    return ( a==b or 
             int(a*10**sig_fig) == int(b*10**sig_fig)
           )

def calc_chi2(sig, err, cf=1.):
    """
    >>> import numpy as np
    >>> import pylab as pl
    >>> s_in= pl.rand(8)
    >>> e_in= np.arange(8)*1.0
    >>> mn, adjchi2, devSq= calc_chi2(s_in, e_in, cf=1)
    >>> nearly_equal(mn, 0, 1) 
    True
    >>> nearly_equal(adjchi2, 1, 1)
    True
    >>> devSq
    ...
    

    DOES NOT DO DEGREES OF FREEDOM PROPERLY!
     AND
    ONLY HANDLES 1D ARRAYS
    """
    inv_variance=1./err**2; # Variance from individual error bars

    inv_variance[np.isinf(inv_variance) | np.isnan(inv_variance) | np.isinf(sig) | np.isnan(sig)]=0
    inv_variance[np.isinf(values) | np.isnan(values)]=0 #HOPEFULY I'VE CHANGED THIS RIGHT
    average = np.average(values, inv_variance=inv_variance,axis=axis)
    #variance = np.average((values-np.expand_dims(average, axis))**2, inv_variance=inv_variance,axis=axis)  # Fast and numerically precise
    #unc = variance/(len(values)-1)

    #return average, np.sqrt(variance), np.sqrt(unc)
            
    inv_variance_sum=nansum(inv_variance)
    mn=nansum(sig*inv_variance)/inv_variance_sum


    N=sum(~isnan(sig))
    cf=cf*((N-1)/N)**2# 'Correlation factor': Used to be 2.54. This stuff is (ftm) worked out emprically
    degOfF=N/cf 
    devSq=(sig-mn)**2*inv_variance
    chi2=nansum(devSq)/degOfF; #: = var(st)/varfromerrorbars Needs checking too.
    return mn, chi2, devSq

def weighted_cross_val(y, weights, redF= np.average, Nresamps=1000, bRetSamps=False, bPlotHist=False):
    """Caclculate an expectation value and a deviation for a weighted data set using sub sampling
    
    @y: data values
    @weights: weights
    @redF: function to estimate. It is applied to each set of boot-strap samples
    
    Algorithm
    =========
    Non-weighted: Repeat the following Nresamps times: 
        * shuffle the y values
        * take each half and calculate the function
        * record the difference in estimates between the halves
    Weighted:
        * shuffle an array of indexes [0,1,2,... N]
        * shuffle the weights according to these indices, then calculate the cumulative sum
        * Calculate the (fractional) index corresponding to half the sum of all the weights, I
        * samples [0:I] belong to one data set, and [I+1:] belong to the other
        * The sample [I] will count (I-int(I)) to the left data set and (I+1-int(I)) to the right
            * Both the value and the weights will be scaled
    
    """
    #pdb.set_trace()
    N=y.size
    A=np.arange(N)
    weights=weights.copy()/(0.5*weights.sum()) # Sum of weights will be 2
    #C=cumsum(weights)
    k=0
    valL=[]
    while k < Nresamps:
        np.random.shuffle(A)
        Yb=y[A]
        Wb=weights[A]
        Cb=Wb.cumsum()
        If=Cb.searchsorted(1.)
        fr=(Cb[If]-1.)/Wb[If] # Fraction of the If-th sample that belongs to the left set
        
        leftW=np.where(Cb < 1, Wb, 0)
        rightW=np.where(Cb > 1, Wb, 0)
        leftW[If]=Wb[If]*(1.-fr)
        rightW[If]=Wb[If]*(fr)
        vals=[redF(Yb, weights=leftW), redF(Yb, weights=rightW)]
        #pdb.set_trace()
        valL.append(vals)
        k+=1
    sArr=array(valL)
    if bPlotHist:
        pl.hist(sArr, 50)
    if bRetSamps:
        return sArr
    else:
        #pdb.set_trace()
        return np.mean(sArr), np.sqrt(np.mean(np.diff(sArr,axis=1)**2))/2.

def bootstrap(y, redF= np.mean, Nresamps=1000, bRetSamps=False, bPlotHist=False, conf_int='std'):
    """Caclculate an expectation value and a deviation for a weighted data set using bootstrapping
    
    @y: data values
    @weights: weights
    @redF: function to estimate. It is applied to each set of boot-strap samples
    """
    N=y.shape[0]
    k=0
    valL=[]
    while k < Nresamps:
        Is=random.uniform(0, N, N).astype('i4')
        y_boot=y[Is]
        valL.append(redF(y_boot))
        k+=1
    sArr=np.array(valL)
    if bPlotHist:
        pl.hist(sArr, 50)


    if bRetSamps:
        return sArr
    else:
        if conf_int=='std':
            SE=np.std(sArr)
        else:
            sArr.sort()
            #pdb.set_trace()
            SE=sArr[[Nresamps*(1-conf_int)/2, Nresamps*(1+conf_int)/2]]
            
    return np.mean(sArr), SE
from scipy import interpolate
def weighted_double_bootstrap(y, weights=None, redF= np.mean, N1=300, N2=200, bRetSamps=False, bPlotHist=False, conf_int=0.68, theta0=None, retStdEr=False):
    """Caclculate an expectation value and a deviation for a weighted data set using bootstrapping
    
    @y: data values
    @weights: weights
    @redF: function to estimate. It is applied to each set of boot-strap samples
    """
    N=y.shape[0]
    if weights is None:
        weights=np.ones(N, dtype='bool')
    C=np.cumsum(weights)
    k=0
    valL=[]
    ciL=[]
    mnsL=[]
    gamma_iL=[conf_int-.5*(1-conf_int), conf_int, conf_int+.5*(1-conf_int)]
    while k < N1:
        Is=C.searchsorted(random.uniform(0, C[-1], N1))
        y_boot=y[Is]
        cis=[]
        mns=[]
        #pdb.set_trace()
        for gamma_i in gamma_iL:
            mn, CI=bootstrap(y_boot.copy(), redF=redF, Nresamps=N2, conf_int=gamma_i) 
            cis.append(CI)
            mns.append(mn)

        ciL.append(cis)
        valL.append(redF(y_boot))
        mnsL.append(mns)
        k+=1
    ciA=array(ciL)
    sArr=np.array(valL)
    theta_star=redF(sArr)
    gamma_i_starL=[sum(np.where( (ciA[:,k,0]< theta_star) & (ciA[:,k,1]>theta_star),1,0))/float(N1) for k in range(len(gamma_iL))]

    intpObj=interpolate.interp1d(gamma_i_starL, gamma_iL, 'quadratic')
    newConfInt=intpObj(conf_int)



    if bPlotHist:
        pl.hist(sArr, 50)


    if bRetSamps:
        return sArr
    else:
        if conf_int=='std':
            SE=np.std(sArr)
        else:
            sArr.sort()
            SE=sArr[[N1*(1-newConfInt)/2, N1*(1+newConfInt)/2]]
            #SE=sArr[[Nresamps*(1-conf_int)/2, Nresamps*(1+conf_int)/2]]

    if theta0 is not None:
        thetaCorr=3*theta0 - 3*np.mean(sArr) + np.mean(mnsL)
    else:
        thetaCorr=2*np.mean(sArr)-np.mean(mnsL)
            
    if retStdEr:
        return thetaCorr, SE, np.std(sArr)#, array(ciL)
    return thetaCorr, SE#, array(ciL)
def weighted_bootstrap(y, weights=None, redF= np.mean, Nresamps=1000, bRetSamps=False, bPlotHist=False, conf_int=0.68, theta0=None, retStdEr=False):
    """Caclculate an expectation value and a deviation for a weighted data set using bootstrapping
    
    @y: data values
    @weights: weights
    @redF: function to estimate. It is applied to each set of boot-strap samples
    """
    N=y.shape[0]
    if weights is None:
        weights=np.ones(N, dtype='bool')
    C=np.cumsum(weights)
    k=0
    valL=[]
    while k < Nresamps:
        Is=C.searchsorted(random.uniform(0, C[-1], N))
        y_boot=y[Is]
        valL.append(redF(y_boot))
        k+=1
    sArr=np.array(valL)
    if bPlotHist:
        pl.hist(sArr, 50)


    if bRetSamps:
        return sArr
    else:
        if conf_int=='std':
            SE=np.std(sArr)
        else:
            sArr.sort()
            #SE=sArr[[Nresamps*(conf_int/2), Nresamps*(1-conf_int/2)]]
            SE=sArr[[Nresamps*(1-conf_int)/2, Nresamps*(1+conf_int)/2]]
    if theta0 is None:
        theta0=np.mean(sArr)
            
    if retStdEr:
        return 2*theta0-np.mean(sArr), SE, np.std(sArr)
    return 2*theta0-np.mean(sArr), SE

def fancy_column_stats(sig, err, cf=1, chi2Max=np.inf, Nmin=3):
    """Function to estimate the covariance for set of orthogonal measurements, with possible nans and a filter for points that deviate alot
    @dat: Nx2 array of samples

    e.g. 
    >>> import numpy as np
    >>> dat= np.array([[0.1, 0.05], [0.05, 0.07], [np.nan, 0.06], [0.07, 0.04], [0.11, 0.05], [0.1, 0.02]])
    >>> err= 1.*np.ones((6,2))
    >>> dat
    [[0.1, 0.05], [0.05, 0.07], [nan, 0.06], [0.1, 0.02]]
    >>> valRes, errOutRes, chi2Res=fancy_column_stats(dat, err, cf=1., chi2Max=10, Nmin=5)

    Process:
        1. Check if all of one column is NaN, in which case just do a nanvar on the remaining one and return the result
        2. Mask out all rows with at least one nan entry
        3. Calculate chi2=(dat-datMean)*Ndof/inv_pt_var
        4. If chi2 is greater than @chi2Max, mask out the largest deviating point and go to 3
        4. Remove the point with the biggest deviation and recalculate val=norm(cov). If val reduces by more than 1/N (or something else??), consider it good and keep removing. Else, take the original.
        5. Check that we still have at least @Nmin points, return nans if we don't


    TODO:
    Add the possibility to calculate error (and maybe mean bias) using the bootstrap, in which case we could set chi2 to 1.
    """
    sig=array(sig)
    err=array(err)
    if sig.ndim<2:
        sig=sig.reshape(sig.size,1)
        err=err.reshape(sig.size,1)
    if sig.shape[0]<cf-1: #This won't work, we'll enforce a nan result
        sig*=nan
    NcolInit=sig.shape[1]
    nanColsI=np.all(np.isnan(sig), axis=0)
    gdColsI=~nanColsI
    if np.any(nanColsI):
        vprint("One column is all nans, we'll just do variance not covariance")
        sig=sig[:,gdColsI]
        err=err[:,gdColsI]
        #ampFinal= np.nanmean(sig, axis=0)

    #Eliminate any allnan rows
    nanRowsI=np.isnan(np.sum(sig, axis=1))
    sig=sig[~nanRowsI]
    err=err[~nanRowsI]

    # CHI2 ELIMINATION
    # SHOULD BE NO NANS LEFT AT THIS POINT
    ## While the set of string points doesn't fit a chi2 model, we eliminate the largest deviating points
    while 1:
        N=sig.shape[0]
        if any(np.sum(~np.isnan(sig[:]), axis=0)<Nmin): #if we have too few points left
            #print("Not enough string points (prob by chi2 limit) "),
            errFinal=nan
            ampFinal=nan*np.ones(gdColsI.sum())
            chi2=nan*np.ones(gdColsI.sum())
            break;

        inv_variance=1./err**2; # Variance from individual error bars
        inv_variance_sum=np.nansum(inv_variance, axis=0)
        inv_variance_norm=inv_variance/inv_variance_sum
        ampFinal=np.nansum(sig*inv_variance, axis=0)/inv_variance_sum
        #N=min(sig.sum(~np.isnan(sig), axis=0)) #Should really leave this as 2d
        #cf=3.*((N-1)/N)**2# 'Correlation factor': Used to be 2.54. This stuff is (ftm) worked out emprically
        degOfF=(N-1)/cf #underestimate I think?
        devSq=(sig-ampFinal)**2
        chi2=np.nansum(devSq*inv_variance,axis=0)/degOfF; #: = var(st)/varfromerrorbars Needs checking too.
        chi2comb=np.mean(chi2)
        if chi2comb>chi2Max and chi2Max: # If chi2 is too high, let's throw out another point
            I=np.nanargmax(devSq.sum(axis=1))# sample set with biggest mean deviation
            #I=nanargmax(dev)
            sig=inplace_without_index(sig, I)
            err=inplace_without_index(err, I)
            #sig[I]=nan
            #err[I]=nan
        else: #Done
            #ampFinal=np.mean(ampFinalArr)
            #errFinal=sqrt(np.nansum((sig-ampFinal)**2*inv_variance/inv_variance_sum, axis=0)/(degOfF-2))#/sqrt(1-stats.chi2.cdf(chi2, degOfF));
            errFinal=weighted_cov(sig, rowvar=0, weights=inv_variance.sum(axis=1))/degOfF
            break

    # Assemble final data
    if not hasattr(errFinal, '__iter__'):
        gdCol=np.where(gdColsI)[0][0]
        temp=np.nan*np.ones((2,2))#[[np.nan, np.nan], [np.nan, np.nan]]
        temp[gdCol, gdCol]=errFinal
        errFinal=temp
        temp=array([nan, nan])
        temp[gdCol]=ampFinal[0]
        ampFinal=temp.copy()
        temp[gdCol]=chi2[0]
        chi2=temp.copy()

    if gdColsI.sum()==0:
        #a=empty(NcolInit)*nan

        #temp=a.copy(); temp[gdColsI]=ampFinal; ampFinal=temp
        #temp=a.copy(); temp[gdColsI]=errFinal; errFinal=temp
        #temp=a.copy(); temp[gdColsI]=chi2; chi2=temp
        ampFinal=empty(2)*nan
        errFinal=empty((2,2))*nan
        chi2=empty(2)*nan
        #return a, a.copy(), a.copy()
    return ampFinal, errFinal, chi2

def inplace_without_index(arr, idx):
    arr[idx:-1] = arr[idx+1:]; arr = arr[:-1]
    return arr

def weighted_cov(m, y=None, rowvar=1, bias=0, ddof=None, weights=None,
        repeat_weights=False):
    """
    Estimate a covariance matrix, given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.
    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        form as that of `m`.
    rowvar : int, optional
        If `rowvar` is non-zero (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : int, optional
        Default normalization is by ``(N - 1)``, where ``N`` is the number of
        observations given (unbiased estimate). If the data is weighted then
        ``N`` corresponds to the effective number of samples given by
        :math:`\frac{\left(\sum_i{w_i}\right)^2}{\sum_i{w_i^2}}` If `bias` is
        1, then normalization is by ``N``. These values can be overridden by
        using the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        .. versionadded:: 1.5
        If not ``None`` then normalization is by ``(N - ddof)``, where ``N`` is
        as described in the description of ``bias``; this overrides the value
        implied by ``bias``. The default value is ``None``.
    weights : array_like, optional
        An array of weights associated with the values in `m` and `y`. Each
        value in `m` and `y` contributes to the covariance according to its
        associated weight. The weights array is 1-D and its length must be
        the size of `m` and `y` along the specified axis. If `weights=None`,
        then all data in `m` and `y` are assumed to have a weight equal to one.
    repeat_weights : bool, optional
        If ``True`` then weights are treated as the number of occurences of the
        values in `m` and `y`. If ``False`` (the default) then weights
        represent the relative importance of each value.
    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.
    See Also
    --------
    corrcoef : Normalized covariance matrix
    Examples
    --------
    Consider two variables, :math:`x_0` and :math:`x_1`, which
    correlate perfectly, but in opposite directions:
    >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    array([[0, 1, 2],
           [2, 1, 0]])
    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
    matrix shows this clearly:
    >>> np.cov(x)
    array([[ 1., -1.],
           [-1.,  1.]])
    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.
    Further, note how `x` and `y` are combined:
    >>> x = [-2.1, -1,  4.3]
    >>> y = [3,  1.1,  0.12]
    >>> X = np.vstack((x,y))
    >>> print np.cov(X)
    [[ 11.71        -4.286     ]
     [ -4.286        2.14413333]]
    >>> print np.cov(x, y)
    [[ 11.71        -4.286     ]
     [ -4.286        2.14413333]]
    >>> print np.cov(x)
    11.71
    """
    # Check inputs
    #if ddof is not None and ddof != int(ddof):
    #    raise ValueError(
    #        "ddof must be integer")

    # Handles complex arrays too
    m = np.asarray(m)
    if y is None:
        dtype = np.result_type(m, np.float64)
    else:
        y = np.asarray(y)
        dtype = np.result_type(m, y, np.float64)
    X = np.array(m, ndmin=2, dtype=dtype)

    if X.shape[0] == 1:
        rowvar = 1
    if rowvar:
        axis = 0
    else:
        axis = 1

    if X.shape[axis] == 0:
        return np.array([]).reshape(0, 0)

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0
    if weights is None:
        fact = float(X.shape[1 - axis] - ddof)
    else:
        weights = np.asarray(weights)
        if weights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional weights")
        if weights.shape[0] != X.shape[1 - axis]:
            raise RuntimeError("incompatible numbers of samples and weights")
        if any(weights < 0):
            raise RuntimeError("weights cannot be negative")
        weight_sum = float(sum(weights))
        if repeat_weights:
            fact = weight_sum - ddof
        else:
            weights /= weight_sum
            N_eff = 1.0/sum(weights*weights)
            fact = (N_eff - ddof)/N_eff
    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning)
        fact = 0.0

    if y is not None:
        y = np.array(y, copy=False, ndmin=2, dtype=dtype)
        X = np.concatenate((X, y), axis)

    avrs = np.average(X, axis=1-axis, weights=weights)
    if not rowvar:
        X -= avrs
        if weights is None:
            X_T = X.T
        else:
            X_T = X.T*weights
        return (np.dot(X_T, X.conj()) / fact).squeeze()
    else:
        X_T = (X.T - avrs)
        if weights is None:
            X = X_T.T
        else:
            X = X_T.T*weights
        return (np.dot(X, X_T.conj()) / fact).squeeze()

def category_plot(names, ydata, err=None, fmt='o',  ax=None, xoffs=0, **kwargs):
    n=len(names)
    if ax is None:
        ax=pl.gca()
    if err is None:
        lines=ax.plot(pl.arange(n)+xoffs, ydata,  fmt, **kwargs)
    else:
        lines=ax.errorbar(pl.arange(n)+xoffs, ydata, err, fmt=fmt, **kwargs)
    ax.set_xticks(pl.arange(n))
    ax.set_xticklabels(names)
    ax.set_xlim([-0.5, n-0.5])
    return lines


from scipy import stats
def Grubbs_outlier_test(y_i, alpha=0.05, bResiduals=False, cf=1):
    """
    Perform Grubbs' outlier test.
    
    ARGUMENTS
    y_i (list or numpy array) - dataset
    alpha (float) - significance cutoff for test

    RETURNS
    G_i (list) - Grubbs G statistic for each member of the dataset
    Gtest (float) - rejection cutoff; hypothesis that no outliers exist if G_i.max() > Gtest
    no_outliers (bool) - boolean indicating whether there are no outliers at specified significance level
    index (int) - integer index of outlier with maximum G_i    
    """
    if not bResiduals:
        centralValue=y_i.mean()
    else:
        centralValue=0
    s = y_i.std()
    G_i = np.abs(y_i - centralValue) / s
    N = y_i.size/cf
    t = stats.t.isf(1 - alpha/(2*N), N-2) 
    Gtest = (N-1)/np.sqrt(N) * np.sqrt(t**2 / (N-2+t**2))    
    G = G_i.max()
    index = G_i.argmax()
    no_outliers = (G < Gtest)
    return [G_i, Gtest, no_outliers, index]

def Grubbs_remove(y_in, weights=None, alpha=0.05, rdof=2, cf=1.):
    """ Recursively remove outliers from data and return the result

    @y_in: the data to be 'cleaned'
    @weights: proportional to the 'frequency', or the inverse variance, of each point
    @alpha: What confidence level to use
    @rdof: redundant degrees of freedom
    @cf: correlation factor
    """
    if weights is None:
        weights=1.0;
    bNoOutliers=False;
    gdMask=np.ones(y_in.shape, dtype=np.bool)
    gdMask[np.isnan(y_in)]=False
    num=0;
    while not bNoOutliers and num<15:
        yCentral=np.average(y_in[gdMask], weights=weights[gdMask])
        y=(y_in-yCentral)*weights
        G_I, Gtest, bNoOutliers, index=Grubbs_outlier_test(y[gdMask], alpha, bResiduals=True)
        if not bNoOutliers:
            print("Remove value {} at index {}".format(y[index], index))
            gdMask[index]=False
        num+=1;
    return ~gdMask, y[gdMask]

def Grubbs_outlier_cutoff_distribution(N):
    """
    Generate the Grubbs' outlier test statistic cutoff distribution for various values of significance.
    """
    npoints = 50
    alphas = np.logspace(0, -20, npoints)
    from scipy import stats
    Gtest = np.zeros([npoints], np.float64)
    for (i,alpha) in enumerate(alphas):
        t = stats.t.isf(1 - alpha/(2*N), N-2) 
        Gtest[i] = (N-1)/np.sqrt(N) * np.sqrt(t**2 / (N-2+t**2));  
    return [alphas, Gtest]
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
def outlier_identify(y, weights, Nsm=20, alpha=0.05, to_find=100, bPlot=False):
    gdI= ~(np.isnan(y) | np.isnan(weights))
    y=y[gdI]
    weights=weights[gdI]
    
    bNoOutliers = False
    gdMask=np.ones(y.shape, dtype=bool)
    k=0;
    #print("alpha: {}".format(alpha))
    while not bNoOutliers and k < to_find:
        if Nsm==-1:
            smthY, _, _= weightedStats(y, weights)   

        else:
            smthY, _= weighted_smooth(y[gdMask], weights[gdMask], Nsm, mode='same')
        #gdMask=gdMask & msk
        #plot(y-smthY)
        resids = (y[gdMask]-smthY)*np.sqrt(weights[gdMask])
        if bPlot:
            pl.plot(resids,'.')
        [G_i, Gtest, bNoOutliers, index] = Grubbs_outlier_test(resids, alpha, bResiduals=True)
        #plot(t[gdMask][index], resids[index], 'rx')
        if not bNoOutliers:
            I=np.cumsum(gdMask).searchsorted(index)
            gdMask[I+1]=False
        k+=1;
    #figure()
    #plot(t, y)
    #plot(t[~gdMask], y[~gdMask], 'rx')
    return ~gdMask

def noise_spectrum_amp(N, shapeF=None, sampleRate=1, complexNoise=False):
    """Generate random noise with a given spectrum
    
    @N: number of points
    @shapeF: spectral shape of the noise. Either a callable function of frequency, 
            or an array of size N which will multipy the fft'd y data
    @sampleRate: the sample rate of the noise which will be used if shapeF is a function
    
    """
    y=random.normal(size=N) #White noise
    if complexNoise:
        y+=1j*random.normal(size=N)
        y=np.exp(1j*200*pi*random.normal(size=N))
    Y=fft(y)
    
    if callable(shapeF):
        f=np.fft.fftfreq(N, 1/sampleRate)
        sclF=shapeF(f)
    else:
        sclF=shapeF
        
    #Interpolate out any infinities and nans
    sclF[np.isinf(sclF)]=0
    #sclF.real=naninterp(sclF.real)
    #sclF.imag=naninterp(sclF.imag)
    Yout=Y*sclF #sqrt because we're dealing with the amplitude, not the power spectrum
    yout=ifft(Yout)
    if not complexNoise:
        yout=yout.real

    return yout
def noise_spectrum(N, shapeF=None, sampleRate=1, complexNoise=False):
    """Generate random noise with a given spectrum
    
    @N: number of points
    @shapeF: spectral shape of the noise. Either a callable function of frequency, 
            or an array of size N which will multipy the fft'd y data
    @sampleRate: the sample rate of the noise which will be used if shapeF is a function
    
    """
    y=random.normal(size=N) #White noise
    if complexNoise:
        y+=1j*random.normal(size=N)
    Y=fft(y)
    
    if callable(shapeF):
        f=np.fft.fftfreq(N, 1/sampleRate)
        sclF=np.sqrt(np.abs(shapeF(f)))
    else:
        sclF=np.sqrt(np.abs(shapeF))
        
    #Interpolate out any infinities and nans
    sclF[np.isinf(sclF)]=np.nan
    sclF=naninterp(sclF)
    Yout=Y*sclF #sqrt because we're dealing with the amplitude, not the power spectrum
    yout=ifft(Yout)
    if not complexNoise:
        yout=yout.real

    return yout


def invisi_plot(spec):
    fig =pl.gcf()
    ax=fig.add_subplot(spec)
    #ax.set_title("harmonic: {}".format(n))
    #ax.set_title(axTitle)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.patch.set_alpha(0.)
    ax.set_frame_on(False)
    return ax

import sys, time
try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False
class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)

        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print('\r '.format(self), end="")
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

def combinedD(d1, *args):
    out=copy.copy(d1); #out.update(d2)
    for arg in args:
        out.update(arg)
    return out

class Bunch(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

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

def fill_regions_x(xStarts, xEnds, ax=None, yRange=None, **kwargs):
    fillArgs=dict( linewidth=0,
                    alpha=0.5,
            )
    fillArgs.update(kwargs)
    if ax is None:
        ax=pl.gca()
    if not yRange:
        yRange=ax.get_ylim()
   
    for xStart, xEnd in zip(xStarts, xEnds):
        ax.fill_betweenx(yRange, [xStart, xStart], [xEnd,xEnd], **fillArgs )
    #vstack([xStarts,xEnds]).T.ravel()

def isd2uhz(isd):
    secsPerSd=(23.+56./60)/24*24*3600
    return 1./secsPerSd*isd

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

    
    

#def zeroOut(arr, tol=1e-10):

if __name__=='__main__':
    import doctest
    doctest.testmod()
