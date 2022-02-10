""" Parameters shared and persistent between modules. Try to only modify them in one place!

"""
from box import Box, box_from_file
from copy import deepcopy
import os

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
    def __init__(self,  filename="shared_params.yaml", overwrite_with=None):
        self.curParFilename= filename
        if overwrite_with is not None:
            if isinstance(overwrite_with, str):
                self.save(self.load(overwrite_with))
            else:
                self.save(overwrite_with)

    def load(self, filename=None):
        if filename is None:
            filename=self.curParFilename
        return box_from_file(filename, box_dots=True)
        #epInit.pulseAmps.bonusZHeight=0

    def save(self, params, filename=None):
        if filename is None:
            filename=self.curParFilename
        open(filename, 'w').write(params.to_yaml())
        #pickle.dump(params,open(filename, 'bw'))
        #epInit=pickle.load(open(pklFilename, 'br'))

    def getCopy(self): #This is the actual parameters
        return self.load()
    
    @property
    def P(self):
        return self.getCopy()

    @P.setter
    def P(self, value):
        raise AttributeError("Use modify or tweak to change parameters")
    
    def replace(self, newParams):
        self.save(newParams)

    def modify(self,**kwargs):
        p = self.getCopy() + Box(kwargs)
        self.replace(p)

    def tweak(self, tweakD, root = None, scale=1.0, bPermanent=False ):
        """ Adjust the specified parameters by the given amounts. Takes a 'root' parameter to make
        specifying nested parameters easier.
        
        nested paramters (or the root) should be specified in dot format for nested dicts (e.g. "fields.x")
        """
        p0 = self.getCopy()
        if root is not None:
            p = p0[root]
        else:
            p = p0

        changedD = Box({key:p[key] + scale*tweakD[key] for key in tweakD})
        p.merge_update(changedD)
        if bPermanent:
            self.replace(p0)
        return p0




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
    params.modify(Bx = 0.5)
    print(params.P)
