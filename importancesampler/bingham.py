from fixed import FixedImportanceSampler
import numpy as n
import geom

import pyximport; pyximport.install(setup_args={"include_dirs":n.get_include()},reload_support=True)
import fisher_util

class FixedBinghamImportanceSampler(FixedImportanceSampler):
    def __init__(self,suffix,symmetry=None):
        FixedImportanceSampler.__init__(self,suffix)

        self.symmetry = symmetry

        # Compute symmetry operator
        if self.symmetry is not None:
            Rs = self.symmetry.get_rotations()
            self.symmetry_quats = n.array([geom.rotmat3D_to_quat(R) for R in Rs])
        else:
            self.symmetry_quats = None

    def evaluate_kernel(self,inds,vals,odomain,params,logspace=False):
        """
        Evaluate the kernel at all points in the current domain at the
        inds of odomain with value vals
        """ 
        dirs1 = self.domain.get_dirs()
        dirs2 = odomain.get_dirs(inds)

        pscale = params.get('is_bingham_pscale'+self.suffix,params.get('is_bingham_pscale',1.0))
        kappa = n.log((2**odomain.dim)*pscale)/(1-n.cos(odomain.resolution/2.0)**2)
        chiral_flip = params.get('is_bingham_chirality_flip'+self.suffix,params.get('is_bingham_chirality_flip',True))
        quats_sym = self.symmetry_quats

        ret = fisher_util.compute_bingham_kernel(dirs1, dirs2, vals.reshape((-1,)),
                                                 quats_sym, kappa, chiral_flip, True, 
                                                 logspace, None)

        return ret


