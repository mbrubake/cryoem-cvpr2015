from objective import Objective

import numpy as n

class GaussianPrior(Objective):
    def __init__(self,fspace=True):
        Objective.__init__(self,fspace)

    def set_params(self,cparams):
        self.params = cparams

    def get_preconditioner(self,precond_type):
        N_D_Train = self.cryodata.N_D_Train

        mscale = self.params['modelscale']
        stddev = self.params['gaussprior_stddev']*mscale

        return (1.0/(stddev**2))/N_D_Train

    def scalar_eval(self,vals,compute_gradient=False):
        cparams = self.params

        mscale = cparams['modelscale']
        stddev = cparams['gaussprior_stddev']*mscale
        cP1 = 1.0/(stddev**2)
        lcP1 = n.log(stddev) + 0.5*n.log(2.0*n.pi)

        P1 = (0.5*cP1) * vals**2 + lcP1
        
        if compute_gradient:
            dP1dvals = cP1*vals
        
        if compute_gradient:
            return P1,dP1dvals
        else:
            return P1

    def learn_params(self, params, M=None, fM=None):
        mscale = params['modelscale']

        if self.fspace:
            sigma2 = n.mean(fM.real**2) + n.sum(fM.mean**2)
        else:
            sigma2 = n.mean(M**2)
        sigma = n.sqrt(sigma2)/mscale
        print "gaussprior_stddev = {0} (was {1})".format(sigma,params['gaussprior_stddev']) 
        params['gaussprior_stddev'] = sigma
        

    def eval(self, compute_gradient=True, M=None, fM=None, **kwargs):
        cryodata = self.cryodata
        stats = cryodata.get_data_stats()
        N_D_Train = stats['N_D_Train']

        outputs = {}
        
        if not self.fspace:
            if compute_gradient:
                nlogP,dnlogP = self.scalar_eval(M.reshape((-1,)),True)
                return n.sum(nlogP,dtype=n.float64)/N_D_Train, \
                       dnlogP.reshape(M.shape)/N_D_Train, \
                       outputs
            else:
                nlogP = self.scalar_eval(M.reshape((-1,)),False)
                return n.sum(nlogP,dtype=n.float64)/N_D_Train, \
                       outputs
        else:
            cparams = self.params
            mscale = cparams['modelscale']
            stddev = cparams['gaussprior_stddev']*mscale
            cP1 = 1.0/(stddev**2)
            lcP1 = n.log(stddev) + 0.5*n.log(2.0*n.pi)
            
            nlogP = (0.5*cP1/N_D_Train)*(n.sum(fM.real**2) + n.sum(fM.imag**2)) + (float(fM.size)/N_D_Train)*lcP1
            if compute_gradient:
                dnlogP = (cP1/N_D_Train)*fM
                return nlogP, dnlogP, outputs
            else:
                return nlogP, outputs
            