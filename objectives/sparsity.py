from objective import Objective

import numpy as n

class SparsityPrior(Objective):
    def __init__(self,assumepos=False):
        Objective.__init__(self,False)
        self.assumepos = assumepos

    def set_params(self,cparams):
        self.params = cparams

    def get_preconditioner(self,precond_type):
        return 0
    
    def scalar_eval(self,vals,compute_gradient=False):
        cparams = self.params

        assumepos = self.assumepos
        mscale = cparams['modelscale']
        cP1 = float(cparams['sparsity_lambda'])/mscale
        lcP1 = n.log(cP1) if cP1 > 0 else 0
        if assumepos:
            P1 = cP1 * vals - lcP1
            if compute_gradient:
                dP1dvals = cP1 * n.ones_like(vals)
        else:
            P1 = cP1 * n.abs(vals) - lcP1
            if compute_gradient:
                dP1dvals = cP1 * n.sign(vals)

        if compute_gradient:
            return P1,dP1dvals
        else:
            return P1

    def learn_params(self, params, cparams, M=None, fM=None):
        mscale = params['modelscale']

        scale = mscale/n.mean(M[M>0])
        print "sparsity_lambda = {0} (was {1})".format(scale,params['sparsity_lambda']) 
        params['sparsity_lambda'] = scale
        cparams['sparsity_lambda'] = scale

    def eval(self, M, compute_gradient=True, fM=None, **kwargs):
        N_D_Train = self.cryodata.N_D_Train

        outputs = {}

        if compute_gradient:
            nlogP,dnlogP = self.scalar_eval(M.reshape((-1,)),True)
            return n.sum(nlogP,dtype=n.float64)/N_D_Train, \
                   dnlogP.reshape(M.shape)/N_D_Train, \
                   outputs
        else:
            nlogP = self.scalar_eval(M.reshape((-1,)),False)
            return n.sum(nlogP,dtype=n.float64)/N_D_Train, \
                   outputs

