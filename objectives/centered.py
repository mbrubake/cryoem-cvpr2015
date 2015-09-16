from objective import Objective

import numpy as n

import geom

class CenteredPenalty(Objective):
    def __init__(self):
        Objective.__init__(self,False)

    def set_params(self,cparams):
        self.params = cparams

    def eval(self, M, compute_gradient=True, fM=None, **kwargs):
        cparams = self.params
        cryodata = self.cryodata

        resolution = cparams['resolution']
        posvar = float(cparams['centered_var'])/resolution**2

        coords = n.require(geom.gencoords(M.shape[0],3).reshape(M.size,3), \
                           dtype=M.dtype)

        stats = cryodata.get_data_stats()
        N_D_Train = stats['N_D_Train']

        outputs = {}

        Msum = n.sum(M, dtype=n.float64)

        com_num = n.dot(M.reshape((1,-1)),coords)
        com = com_num / Msum
        nlogP = n.sum(com**2)/(2*posvar*N_D_Train)

        if compute_gradient:
            dcom = (coords.reshape(M.size,3) / Msum) - (com_num.reshape(1,3) / Msum**2)
            dnlogP = n.dot(dcom,com.T)/(posvar*N_D_Train)
            return nlogP, dnlogP.reshape(M.shape), outputs
        else:
            return nlogP, outputs

