from objective import Objective

import numpy as n

class NegativePenalty(Objective):
    def __init__(self):
        Objective.__init__(self,False)

    def set_params(self,cparams):
        self.params = cparams

    def eval(self, M, compute_gradient=True, fM=None, **kwargs):
        cparams = self.params
        cryodata = self.cryodata

        stats = cryodata.get_data_stats()
        N_D_Train = stats['N_D_Train']

        mscale = cparams['modelscale']
        neg_pen = float(cparams['negative_penalty'])
        neg_pen_alpha = float(cparams.get('negative_penalty_alpha', 3))
        cP2 = (-1.0)**(neg_pen_alpha) * (neg_pen/(mscale**3*N_D_Train)) 
        P2 = cP2 * n.sum( M[M < 0]**neg_pen_alpha )
        if compute_gradient:
            dP2dM = cP2 * (M < 0) * M**(neg_pen_alpha - 1.0)

        outputs = {}

        if compute_gradient:
            self.ostream("  Negative Penalty Grad Norm: %.3g" % n.linalg.norm(dP2dM))
            return P2,dP2dM,outputs
        else:
            return P2,outputs

