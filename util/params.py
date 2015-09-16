import os

# These imports are here so that when we evaluate parameters we have access to the functions
from math import *
import numpy as n

class EvaluatedParams(dict):
    """
    Stores parameters that have been evaluated and ensures consistency of default parameters 
    if defaults are used in multiple places or are needed in visualization as well as in 
    computation.
    """
    def __init__(self,basedict = None):
        dict.__init__(self)
        
        self.defaults = {}
        if basedict is not None:
            self.update(basedict)

    def get(self, k, default):
        if k not in self.defaults:
            self.defaults[k] = default
        else:
            assert self.defaults[k] == default
        return dict.get(self,k,self.defaults[k])

    def __getitem__(self, k):
        if k not in self and k in self.defaults:
            return self.defaults[k]
        else:
            return dict.__getitem__(self,k)

class Params():
    """
    Class for parameters of an experiment.
    Loads parameters from multiple files, using dictionary syntax.
    The parameters are loaded in order of the file specified, so the first file can be defaults, 
    the second can be task specific, the third can be experiment specific, etc.

    Each entry in the globalparams dict is eval'ed when evaluate is called, and the arguments passed to evaluate 
    can be used in the evaluation.
    """

    def __init__(self, fnames = []):
        self.globalparams = {}
        self.fnames = fnames
        self.load()

    def load(self):
        "Load all fnames in order. Skip ones that don't exist."
        for fname in self.fnames:
            if os.path.isfile(fname):
                with open(fname) as f:
                    indict = eval(f.read()) # read entire file
            self.globalparams.update(indict)

    def partial_evaluate(self, fields__, **kwargs):
        locals().update(kwargs)
        cparams = EvaluatedParams(kwargs)
        for k in fields__:
            if k in self.globalparams:
                v = self.globalparams[k]
                if isinstance(v, str):
                    cparams[k] = eval(v)
                else:
                    cparams[k] = v
        return cparams

    def evaluate(self, skipfields = set(), **kwargs):
        locals().update(kwargs)
        cparams = EvaluatedParams(kwargs)
        for k,v in self.globalparams.iteritems():
            if isinstance(v, str) and k not in skipfields:
                cparams[k] = eval(v)
            else:
                cparams[k] = v
        return cparams

    def __setitem__(self, k, v):
        self.globalparams[k] = v

    def get(self, k, default=None):
        return self.globalparams.get(k,default)

    def __getitem__(self, k):
        return self.globalparams[k]

    def __delitem__(self, k):
        del self.globalparams[k]






