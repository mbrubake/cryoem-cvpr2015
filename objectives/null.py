from objective import Objective

import numpy as n

class NullPrior(Objective):
    def __init__(self,fspace):
        Objective.__init__(self,fspace)

    def set_params(self,cparams):
        self.params = cparams

    def get_preconditioner(self,precond_type):
        return 0
    
    def scalar_eval(self,vals,compute_gradient=False):
        if compute_gradient:
            return n.ones_like(vals),n.zeros_like(vals)
        else:
            return n.ones_like(vals) 

    def eval(self, M, compute_gradient=True, fM=None, **kwargs):
        outputs = {}

        if compute_gradient:
            if self.fspace:
                dM = density.zeros_like(fM)
            else:
                dM = density.zeros_like(M)
            return 0.0, dM, outputs
        else:
            return 0.0, outputs


