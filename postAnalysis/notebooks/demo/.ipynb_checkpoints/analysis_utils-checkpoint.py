import numpy as np
import xarray as xr
import box
from os import path
from scipy.optimize import lsq_linear

def load_dataset(file_path, dataDir=""):
    """ Simple wrapper around xarray that reads the parameters and other metadata nicely
    """
    opts = {#"chunks": {}, #??Trying to get it to lazy load, but this doesn't seem to do it.
            'cache':True,
           }
    ds =xr.open_dataset(path.join(dataDir, file_path), **opts )

    for key in ['meta', 'params']: # Interpret strings as YAML, convert to nested boxes
        nested_box = box.from_file.box_from_string(ds.attrs[key], 'yaml')
        ds.attrs[key] = nested_box
    signature_labels = [ key for key in ds.attrs.keys() if key.startswith("signature") ]
    ds.attrs['signatures'] = box.Box()
    for label in signature_labels:
        ds.signatures[label.removeprefix("signature_")] = ds.attrs.pop(label) 
    return ds
    
def split_pm(arr):
    Npts = arr.shape[-1]
    s1, s2 = arr[...,:Npts//2], arr[...,-(Npts//2):]
    return s1 + s2, s1-s2

# Fit to the halfSigs
def fit_response(yL, sigD, N = None):
    if N is None:
        N = len(yL)
    exog = np.vstack([trim(sig) for sig in sigD.values()]).T
    exog = np.vstack([exog.T, np.ones(exog.shape[0], dtype='f8')]).T
    #print(sigD.keys())
    def fit_pm(y):
        yp, ym = [trim(arr) for arr in split_pm(y)]
        return np.hstack([lsq_linear(exog[:,:2], ym).x, lsq_linear(exog, yp).x ] )
    return np.array([fit_pm(y) for y,_ in zip(yL,range(N))])


#def idx_of_first_rising_edge(arr):
#    #first digitize it: convert to range 0->1 and then threshold
#    mn, mx = arr.min(), arr.max()
#    digitized = arr > (mn + mx)/2 # where it's greater than the halfway point of the range
#    return np.argmax(digitized)

def convert_to_binary(x):
    """ Interpret an aray as binary data. Will take any value greater than halfway between min(x) 
    and max(x) and make it 1, otherwise 0.
    """
    mn, mx = np.min(x), np.max(x)
    binaried = x > (mn + mx)/2 # where it's greater than the halfway point of the range
    return binaried
    
def index_of_coinciding_first_rising_edges(arrs):
    """ assuming binary valued arrays in arr, find the index of the first time that all of the arrs have a rising edge simultaneously.
    Specificaly returns the index of the first point when all have transitioned high
    """
    arrs = [convert_to_binary(arr) for arr in arrs]
    # Assume binary already
    up_vals = np.diff(arrs, axis=1).sum(axis=0)
    idx = np.argmax(up_vals == len(arrs))
    print(".")
    return idx +1
    