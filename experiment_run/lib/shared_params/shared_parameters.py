""" Parameters shared and persistent between modules. Try to only modify them in one place!

"""
from box import Box, box_from_file
from copy import deepcopy
import os
from pathlib import Path
import time
from os import stat
import numpy as np
from collections.abc import Mapping

#def iterate_nested_dict(dict_obj):
#    ''' Iterate over all values of nested dictionaries. Returns 
#    '''
#    # Iterate over all key-value pairs of dict argument
#    for key, value in dict_obj.items():
#        # Check if value is of dict type
#        if isinstance(value, dict):
#            # If value is dict then iterate over all its values
#            for pair in  nested_dict_pairs_iterator(value):
#                yield (key, *pair)
#        else:
#            # If value is not dict type then yield the value
#            yield (key, value)

class SharedParams(object):
    curParFilename=None
    lastLoadedModTime = -1;
    def __init__(self,  set_name="default", overwrite_with=None):
        dir = Path.home().joinpath("shared_params_store")
        dir.mkdir(parents=True, exist_ok=True)
        self.curParFilename = dir.joinpath(set_name + ".yaml")
        if not self.curParFilename.exists():
            self.curParFilename.touch()
        if overwrite_with is not None:
            if isinstance(overwrite_with, str):
                self.save(self.load(overwrite_with))
            else:
                self.save(overwrite_with)

    def changedSinceLoad(self):
        return self.lastLoadedModTime < stat(self.curParFilename).st_mtime

    def load(self, filename=None):
        if filename is None:
            filename=self.curParFilename
        return box_from_file(filename, box_dots=True)
        #epInit.pulseAmps.bonusZHeight=0

    def save(self, params, filename=None):
        if filename is None:
            filename=self.curParFilename

        print(f"saving to {self.curParFilename}")
        with open(filename, 'w') as file:
            yml = params.to_yaml()
            if yml:
                res = file.write(yml)
                print(res)

        self.lastModifiedTime = time.time()
        #pickle.dump(params,open(filename, 'bw'))
        #epInit=pickle.load(open(pklFilename, 'br'))

    def getCopy(self): #This is the actual parameters
        p = self.load().copy()
        self.lastLoadedModTime = stat(self.curParFilename).st_mtime
        return p
    
    @property
    def P(self):
        return self.getCopy()

    @P.setter
    def P(self, value):
        raise AttributeError("Use modify or tweak to change parameters")
    
    def replace(self, newParams):
        self.save(newParams)

    def getChanged(self, **kwargs):
        pCopy = self.getCopy()
        pCopy._box_config['box_dots'] = True
        pCopy =  pCopy + Box(kwargs)
        pCopy._box_config['box_dots'] = False
        return pCopy

    def change(self,**kwargs):
        newP = self.getChanged(**kwargs)
        self.replace(newP)
        return newP

    #def loadArrayDict(self, name):
    #    dirName = os.path.dirname(self.curParFilename)
    #    if not name.endswith(".npz"):
    #        name = name +".npz"
    #    filePath = os.path.join(dirName, name)
    #    return dict(np.load(filePath), allow_pickle=True)

    #def saveArrayDict(self, ):
    #    dirName = os.path.dirname(self.curParFilename)
    #    filePath = os.path.join(dirName, name)
    #    np.savez(filePath, **dct )

    def saveArray(self, **arrs):
        dirName = os.path.dirname(self.curParFilename)
        for name, arr in arrs.items():
            filePath = os.path.join(dirName, name)
            if isinstance(arr, Mapping):
                np.savez(filePath, **arr)
            else:
                np.savez(filePath, only = arr )

    def loadArray(self, name):
        dirName = os.path.dirname(self.curParFilename)
        if not name.endswith(".npz"):
            name = name +".npz"
        filePath = os.path.join(dirName, name)
        loaded = dict(np.load(filePath))
        if "only" in loaded:
            return loaded['only']
        return loaded
        


    def getTweaked(self, tweakD, root = None, scale=1.0):
        p0 = self.getCopy()
        if root is not None:
            p = p0[root]
        else:
            p = p0
        changedD = Box({key:p[key] + scale*tweakD[key] for key in tweakD})
        p.merge_update(changedD)
        return p

    def tweak(self, tweakD, root = None, scale=1.0):
        """ Adjust the specified parameters by the given amounts. Takes a 'root' parameter to make
        specifying nested parameters easier.
        
        nested paramters (or the root) should be specified in dot format for nested dicts (e.g. "fields.x")
        """
        newP = self.getTweaked(tweakD, root, scale)
        self.replace(newP)
        return newP




if __name__=="__main__":
    p0= Box(
        pulses=Box(
            tPumpStart= 10e-6,
            tMagStart = 00e-6,
            tMagWidth = 10e-6,
            tPumpWidth = 10e-6,
            tTot=1700e-6,
        ),
        fields=Box(
            Bx=1.00,
            By=-0.0,
            Bz=-0.00,
        ),
        modulations = Box(
            Bx = 0.02,
            BxFreq = 7,
            #...
            #
            #PTheta = (0.001, 0.1),
            #PPhi = (0.001, 0.15)
        ),
        oven = Box(
            T = 120,
            #Pid params?
        ),
    )
    params = SharedParams(filename = "shared_params_testing.yaml", overwrite_with=p0)
    params.change(Bx = 0.5)
    print(params.P)
