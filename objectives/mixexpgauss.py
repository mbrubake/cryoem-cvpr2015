from objective import Objective

import numpy as n

class MixExpGaussPrior(Objective):
    def __init__(self):
        Objective.__init__(self,False)

    def set_params(self,cparams):
        self.params = cparams

    def scalar_eval(self,vals,compute_gradient=False):
        cparams = self.params

        mscale = cparams['modelscale']
        alpha = cparams['mixeg_prob_exp'] # probability of exponential component
        gamma = cparams['mixeg_gamma']*mscale # scale parameter of exponential component
        mu = cparams['mixeg_mu']*mscale # mean of normal component
        sigma = cparams['mixeg_sigma']*mscale # std dev of normal component
        sigma2 = sigma**2

#        print gamma, mu, sigma

        logP_exp = (n.log(alpha) - n.log(2.0*gamma)) - vals/gamma
        if compute_gradient:
            dlogP_exp = n.ones_like(vals)/(-gamma)

        logP_nrm = (n.log(1.0-alpha) - 0.5*n.log(2*n.pi*sigma2)) \
                          - (0.5/sigma2)*(vals - mu)**2
        if compute_gradient:
            dlogP_nrm = (-1.0/sigma2) * (vals - mu)

        logP = n.logaddexp(logP_exp,logP_nrm)
        if compute_gradient:
            dlogP = n.exp(logP_exp - logP) * dlogP_exp + \
                    n.exp(logP_nrm - logP) * dlogP_nrm


        if compute_gradient:
            return -logP,-dlogP
        else:
            return -logP
        
    def learn_params(self,params,cparams,M=None,fM=None):
        mscale = params['modelscale']
        alpha = params['mixeg_prob_exp'] # probability of exponential component
        gamma = params['mixeg_gamma']*mscale # scale parameter of exponential component
        mu = params['mixeg_mu']*mscale # mean of normal component
        sigma = params['mixeg_sigma']*mscale # std dev of normal component
        
        for i in range(1):
            # E-step to compute pexp
            sigma2 = sigma**2
            logpexp = -n.log(2.0*gamma) - M/gamma + n.log(alpha)
            logpgauss = -0.5*n.log(2*n.pi*sigma2) - (0.5/sigma2)*(M - mu)**2 + n.log(1.0-alpha)
            pexp = n.exp(logpexp - n.logaddexp(logpexp,logpgauss))

            # M-step to compute parameter values
            sumpexp = n.sum(pexp)
#             alpha = sumpexp/float(M.size)
#             gamma = n.sum(pexp*M)/sumpexp
#             mu = n.sum((1.0-pexp)*M)/(M.size - sumpexp)
            sigma = n.sqrt(n.sum((1.0-pexp)*(M - mu)**2)/(M.size - sumpexp))

        gamma /= mscale
        mu /= mscale
        sigma /= mscale

        print "p(exp) = {0} (was {1})".format(alpha,params['mixeg_prob_exp'])
        print "  gamma = {0} (was {1})".format(gamma,params['mixeg_gamma'])
        print "  mu = {0} (was {1})".format(mu,params['mixeg_mu'])
        print "  sigma = {0} (was {1})".format(sigma,params['mixeg_sigma'])
        
        params['mixeg_prob_exp'] = alpha
        cparams['mixeg_prob_exp'] = params['mixeg_prob_exp']
        
        params['mixeg_gamma'] = gamma
        cparams['mixeg_gamma'] = params['mixeg_gamma']
        
        params['mixeg_mu'] = mu
        cparams['mixeg_mu'] = params['mixeg_mu']
        
        params['mixeg_sigma'] = sigma
        cparams['mixeg_sigma'] = params['mixeg_sigma']
        
        

    def eval(self, M, compute_gradient=True, fM=None, **kwargs):
        cryodata = self.cryodata

        stats = cryodata.get_data_stats()
        N_D_Train = stats['N_D_Train']

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


