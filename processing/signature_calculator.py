""" Monitor the stream of 'traces' ('frames'?) and calculate 'signatures' for various compoments.

Not yet decided the best way to do this. There are at least a few possibilities I can think of.

* Use some ind of initial guess at signatures, use those to fit out signals and update to better signatures, and iterate.

Pseudo-code:
signatures = np.array([square(), square(), square()])
for k in range(10):
    fit = sm.OLS(X, signatures).fit()
    signatures = np.sum(fit.params*X, axis=?)
    print(fit.residuals)

* Use Independant Component Analysis to obtain components automagically. Perhaps with some kind of refining step,
like subtracting off the parts that fit best using the above techniques then applying ICA to the remainder.

"""
import numpy as np
from statsmodels import api as sm
from sklearn.decomposition import FastICA, PCA
import time
from scipy import signal, fft
from matplotlib import pyplot as plt


def makeTopHatBasis(Nsamples, Ndivs):
    width_pts = Nsamples//Ndivs
    basis = np.zeros( (Ndivs,Nsamples) ,dtype='f8')
    for k in range(Ndivs):
        basis[k,k*width_pts:(k+1)*width_pts] = 1
    return basis
def phase_shift(y, phase, fs, axis=0):
    fax = fft.fftfreq(y.shape[axis], 1/fs)
    Y = fft.fft(y, axis=axis)
    Y[:,fax>0] *= exp(1j*phase)
    yShifted = fft.ifft( Y, axis=axis)
    return yShifted


def find_dominant_freqs(y, fs, nps=512):
    fax, Y=signal.welch(y,fs=fs, nperseg=nps, axis=0)
    fInds = np.argmax(Y, axis=0)
    freqs = fax[fInds]
    peak_heights= [Y[ind, k] for k, ind in enumerate(fInds) ]
    return list(zip(freqs, peak_heights/ np.max(peak_heights) ))



def do_ica(X, fs, n_components=5):
    X = util.smoothalong(X.copy(), 10, axis=1)
    #Xfilt = util.butter_bandpass_filter(X.T, 32, 34, 190, order=9).T
    Xfilt = X
    print("ICA start")
    # Compute ICA
    tStart = time.time()
    ica = FastICA(n_components=n_components, algorithm='parallel')
    S_ = ica.fit_transform(Xfilt)  # Reconstruct signals
    print("Took %i s for doing ICA"%(time.time() - tStart) )
    A_ = ica.mixing_  # Get estimated mixing matrix

    dominant_freqs = find_dominant_freqs(S_, fs=fs)
    #print('ind')
    for k, pair in enumerate(dominant_freqs):
        print("{}: ({}, {:.2f})".format(k,*pair) )

    plt.figure()
    plt.plot(A_)
    plt.legend(np.arange(10))
    plt.title('signatures')
    plt.figure()
    plt.plot(S_)
    plt.legend(np.arange(10))
    plt.title('vs time')    
    return A_, S_, dominant_freqs



if __name__ == "__main__":
    import pylab as pl
    import util
    import shared_parameters

    glbP = shared_parameters.SharedParams("NECOM")

    if 1:
        glbP.dS_dX = 
    else:
        Nbasis = 4

        datD = np.load("test_data.npz") 
        tL = datD['tL']
        y = datD['y'][:4000]
        y = util.smoothalong(y,20, axis=0 )
        y = util.smoothalong(y,20, axis=1 )
        y -= y.mean(axis=0)[None,:]
        t = np.arange(y.shape[1])*datD['dt']
        def normalise(Y):
            return Y/np.linalg.norm(Y, axis=1)[:,None]
            #return Y/np.sqrt((Y*Y).sum(axis=1))[:,None]
        

        signatures = makeTopHatBasis(t.size, Nbasis)
        signatures += .1*np.random.normal(size=signatures.shape)
        signatures = sm.add_constant(signatures.T).T
        signatures = normalise(signatures)
        X = y.T.astype('f8')
        sigL = [signatures]
        fitpL = []
        for k in range(20):
            signatures += 0.00*normalise(np.random.normal(size=signatures.shape))
            fit = sm.OLS(X, signatures.T).fit()
            fit_AC = fit.params - 1*fit.params.mean(axis=1)[:,None]
            fit_AC = fit_AC[:, :-50]
            new_signatures = (fit_AC[:,None]*X[:,:-50]).mean(axis=-1)
            signatures = (1*signatures + normalise(new_signatures))/2
            print (fit.eigenvals)

            sigL.append(signatures)
            fitpL.append(fit.params)

        fitA = np.array(fitpL)
        sigA = np.array(sigL)
        for k in range(Nbasis+1):
            pl.figure();
            pl.plot(sigA[::3,k].T)