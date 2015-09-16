import numpy as n
from sagd import find_L
from base import *

class SGDMomentumStep(BaseStep):
    def __init__(self,predGrad,adaptL=False):
        BaseStep.__init__(self)

        assert predGrad == False, 'NAG is currently unsupported'

        self.v_t = None
        self.pred_grad = predGrad
        self.adaptL = adaptL
        if adaptL:
            self.L = None
        self.g_hat = None

    def get_pred_step(self,params):
        if self.v_t == None:
            return None
        else:
            mu = params['sgd_momentum']
            return mu*self.v_t

    def do_step(self, x, params, cryodata, evalobj, **kwargs):
        f, g, res_train = evalobj( x )
        g = g.reshape((-1,1))

        mu = params['sgd_momentum']
        alpha = params['sgd_avggrad_alpha']

        curr_g = params['sigma']**2*g
        if alpha > 0:
            if self.g_hat == None:
                self.g_hat = (1-alpha)*curr_g
            else:
                self.g_hat *= alpha
                self.g_hat += (1-alpha)*curr_g
            curr_g = self.g_hat

        if self.adaptL:
            incL = params.get('sgd_incL',None)
            if self.L != None and incL != None and incL != 1.0:
                self.L *= incL

            L0 = self.L
            if L0 == None:
                L0 = params.get('sgd_L0',1.0)
                doLS = True
                max_ls_its = None
            else:
                doLS = params.get('sgd_linesearch',True)
                max_ls_its = params.get('sgd_linesearch_maxits',3)

            if doLS:
                self.L = find_L( x, f, g, evalobj, L0, max_ls_its )

            eps = params['sgd_learnrate']/self.L
        else:
            eps = params['sgd_learnrate']

        if self.v_t == None:
            self.v_t = 0
        else:
            self.v_t *= mu
        self.v_t -= eps*curr_g

        return f, g, self.v_t, res_train, 0  # 0 extra operations on data


