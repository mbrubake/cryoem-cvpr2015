from fixed import FixedImportanceSampler
import numpy as n
import time

import pyximport; pyximport.install(setup_args={"include_dirs":n.get_include()},reload_support=True)
import fisher_util

import scipy.spatial as ss
import scipy.sparse as sp

class FixedFisherImportanceSampler(FixedImportanceSampler):
    def __init__(self,suffix,symmetry=None):
        FixedImportanceSampler.__init__(self,suffix)

        self.symmetry = symmetry

        # Compute symmetry operator
        if self.symmetry is not None:
            self.symmetry_Rs = n.array(self.symmetry.get_rotations())
        else:
            self.symmetry_Rs = None

    def evaluate_kernel(self,inds,vals,odomain,params,logspace=False, timing=False):
        """
        Evaluate the kernel at all points in the current domain at the
        inds of odomain with value vals
        """ 
        if timing: 
            times = []
            tic = time.time();
        dirs1 = self.domain.get_dirs()
        pscale = params.get('is_fisher_pscale'+self.suffix,params.get('is_fisher_pscale',1.0))
        kappa = n.log((2**odomain.dim)*pscale)/(1-n.cos(odomain.resolution))
        chiral_flip = params.get('is_fisher_chirality_flip'+self.suffix,params.get('is_fisher_chirality_flip',True))
        Rs = self.symmetry_Rs
        if timing: times += [time.time()-tic]
        if params.get('is_use_sparse_fisher', False):
            assert Rs is None, "Sparse Fisher doesn't support Symmetry"
            dirs2 = odomain.get_dirs()
            if timing: times += [time.time()-tic]

            eps = params.get('is_sparse_fisher_eps', 1e-14)
            kernmat = getkernmat(dirs1, dirs2, kappa, eps, chiral_flip)
            if timing: times += [time.time()-tic]
            if logspace:
                vals = n.exp(vals)
            if inds is None:
                spvals = vals.ravel()
            else:
                spvals = n.zeros(kernmat.shape[1])
                spvals[inds] = vals
            if timing: times += [time.time()-tic]
            ret = kernmat.dot(spvals).ravel()
            if timing: times += [time.time()-tic]
            ret = n.maximum(ret, eps)
            ret /= ret.sum()
            if logspace:
                ret = n.log(ret)
            if timing: times += [time.time()-tic]

        else:
            dirs2 = odomain.get_dirs(inds)
            ret = fisher_util.compute_fisher_kernel(dirs1, dirs2, vals.reshape((-1,)),
                                                Rs, kappa, chiral_flip, True, 
                                                logspace, None)

        if timing: return ret, times
        else: return ret

precomputed_kernmats = {}
def getkernmat(dirsout, dirsin, kappa, eps, chiral_flip):
    "Assume dirsout and dirsin and unit norm vectors"

    arghash = hash ( (dirsout[::10].tostring(), dirsin[::10].tostring(), kappa, eps, chiral_flip) )
    if arghash in precomputed_kernmats:
        return precomputed_kernmats[arghash]
    else:
        kdout = ss.cKDTree(dirsout*(1.0+eps))  #this is to make sure the distance between identicals is nonzero (bad hack but it works)
        kdin = ss.cKDTree(dirsin)
        if chiral_flip: 
            kdin2 = ss.cKDTree(-dirsin)

        distthres = n.sqrt( -2.0/kappa * n.log(eps) )
        distmat = kdout.sparse_distance_matrix(kdin, distthres).tocsr()
        if chiral_flip:
            distmat += kdout.sparse_distance_matrix(kdin2, distthres).tocsr()
        distmat = distmat.tocoo()

        kerndata = n.exp(-kappa/2.0 * distmat.data**2)
        kernmat = sp.coo_matrix((kerndata, (distmat.row, distmat.col)), distmat.shape)

        precomputed_kernmats[arghash] = kernmat

        return kernmat
