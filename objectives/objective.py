import warnings

class Objective():
    def __init__(self,fspace):
        self.fspace = fspace
        self.diagout = None
        self.statout = None
        self.ostream = None

    def get_preconditioner(self,precond_type):
        warnings.warn("{0} does not have an implemented preconditioner".format(type(self).__name__))
        return 0

    def setup(self,params,diagout,statout,ostream):
        self.params = params
        self.diagout = diagout
        self.statout = statout
        self.ostream = ostream

    def set_dataset(self,cryodata):
        self.cryodata = cryodata

    def set_params(self,cparams):
        assert False
        self.params = cparams

    def set_data(self,cparams,minibatch):
        self.params = cparams
        self.minibatch = minibatch

    def learn_params(self, params, cparams, M=None, fM=None):
        pass
            

