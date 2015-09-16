from objective import Objective

import numpy as n
import geom
import itertools as itools

def compute_CAR_matrix(N,C):
#    print N, C
    print 'Computing CAR matrix...', ; sys.stdout.flush()

    midC = ((C.shape[0]-1)/2, (C.shape[1]-1)/2, (C.shape[2]-1)/2)
    assert C[midC[0],midC[1],midC[2]] == 0

    coords = geom.gencoords(C.shape[0],3,).reshape(C.shape + (3,))
    nzC = C != 0.0

    ijs = []
    vs = []
    for (i,j,k) in itools.product(xrange(N),xrange(N),xrange(N)):
        out_ind = i*N**2 + j*N + k
        in_coords = (coords + n.array([i,j,k]).reshape((1,1,1,3))).reshape((-1,3))
        in_inds = in_coords[:,0]*N**2 + in_coords[:,1]*N + in_coords[:,2]
        valid_ins = nzC.reshape((-1,))
        for ell in range(3):
            valid_ins = n.logical_and(valid_ins,in_coords[:,ell] >= 0)
            valid_ins = n.logical_and(valid_ins,in_coords[:,ell] < C.shape[ell])
        nvalid = n.sum(valid_ins)
        ijs.append(n.vstack([ out_ind*n.ones(nvalid), in_inds[valid_ins] ]))
        vs.append(C[valid_ins.reshape(C.shape)])

    W = spsp.csr_matrix((n.concatenate(vs),n.hstack(ijs)),shape=(N**3,N**3),dtype=n.float32)

    del vs
    del ijs

    print 'done.'

    return W

class CARPrior(Objective):
    def __init__(self):
        Objective.__init__(self,False)
        self.car_type = None
        self.car_N = None
        self.car_C = None

    def set_params(self,cparams):
        self.params = cparams

    """ Get the CAR weight matrix """
    def get_W(self,N,car_type):
        if self.car_type != car_type or self.car_N != N:
            self.car_type = car_type
            self.car_N = N
            self.car_C = None

            if car_type.startswith('gauss'):
                sigma = float(car_type[5:])
                Csz = 2*round(3*sigma) + 1
                midpt = (Csz-1)/2
                self.car_C = n.exp((-0.5/sigma**2)*n.sum(geom.gencoords(Csz,3).reshape((Csz,Csz,Csz,3))**2,axis=3))
                self.car_C[midpt,midpt,midpt] = 0.0
                self.car_C /= self.car_C.sum()
            else:
                assert False, 'Unrecognized car_type'
            self.car_W = compute_CAR_matrix(N,self.car_C)

        return self.car_W

    def scalar_eval(self,vals,compute_gradient=False):
        cparams = self.params

        mscale = cparams['modelscale']
        tau = cparams['car_tau']*mscale # std dev of normal component
        tau2 = tau**2

        nlogP_nrm = 0.5*n.log(2*n.pi*tau2) + (0.5/tau2)*vals**2
        if compute_gradient:
            dnlogP_nrm = (1.0/tau2)*vals
            return nlogP_nrm,dnlogP_nrm
        else:
            return nlogP_nrm

    def eval(self, M, compute_gradient=True, fM=None, **kwargs):
        cparams = self.params
        cryodata = self.cryodata

        stats = cryodata.get_data_stats()
        N_D_Train = stats['N_D_Train']
        N = M.shape[0]

        car_type = cparams['car_type']
        mscale = cparams['modelscale']
        tau = mscale*cparams['car_tau'] # std dev of predictive error

        W = self.get_W(N,car_type)
        predM = W.dot(M.reshape((N**3,)))
        predErr = M.reshape((N**3,)) - predM
        print 'pred_tau = {0}'.format(n.sqrt(n.mean(predErr**2))/mscale)
        L = 0.5*n.dot(M.reshape((N**3,)),predErr)/tau**2

        if compute_gradient:
            return L/N_D_Train, predErr.reshape(M.shape)/(N_D_Train*tau**2), {}
        else:
            return L/N_D_Train, {}

