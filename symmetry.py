import numpy as n
from util import memoize
import density

import pyximport; pyximport.install(setup_args={"include_dirs":n.get_include()},reload_support=True)
import sincint

@memoize
def get_symmetryop(symstr):
    if symstr is None:
        return None
    else:
        return SymmetryOp(symstr)

""" A class to handle symmetrizing 3D densities """
class SymmetryOp:
    def __init__(self,symclass='',symorder=None):
        if symorder is None:
            # Parse symclass
            self.symclass = symclass[0].lower()
            self.symorder = int(symclass[1:])
        else:
            self.symclass = symclass.lower()
            self.symorder = symorder
        self.symstring = '{0}{1}'.format(self.symclass,self.symorder)

        assert self.symclass in set(['c','d',''])
        assert self.symorder >= 1

        self.thetas = n.linspace(0,360.0,self.symorder,endpoint=False)
        
        self.Rs = []
        for ct in self.thetas[1:]:
            theta = ct*(n.pi/180.0)
            R = n.array([ [ n.cos(theta),-n.sin(theta), 0],
                          [ n.sin(theta), n.cos(theta), 0],
                          [            0,            0, 1] ])

            self.Rs.append(R)

        if self.symclass == 'd':
            dR2 = n.array([ [ -1.0,    0,    0 ],
                            [    0,  1.0,    0 ],
                            [    0,    0, -1.0 ] ])
            self.Rs.append(dR2)
            for ct in self.thetas[1:]:
                theta = ct*(n.pi/180.0)
                R = n.array([ [ n.cos(theta),-n.sin(theta), 0],
                              [ n.sin(theta), n.cos(theta), 0],
                              [            0,            0, 1] ])
    
                self.Rs.append(n.dot(dR2,R))

    def get_order(self):
        if self.symclass == 'd':
            return 2*self.symorder
        else:
            return self.symorder

    def get_rotations(self,exclude_dsym=False):
        if exclude_dsym:
            return self.Rs[0:(self.symorder-1)]
        else:
            return self.Rs

    def in_asymunit(self,pts,eps=1e-8):
        """ Check if a point is in the asymmetric unit, pts is Nx3 """
        dtheta2 = self.thetas[1]*(n.pi/180.0)/2.0

        pts = pts / n.sqrt(n.sum(pts**2,axis=1)).reshape((-1,1))

        phi = n.arccos(pts[:,2]).reshape((-1,))
        theta = n.arctan2(pts[:,1], pts[:,0]).reshape((-1,))
        
        ret = n.logical_and(-eps-dtheta2 <= theta,
                                            theta <= dtheta2+eps)
        if self.symclass == 'd':
            ret = n.logical_and(ret, phi <= n.pi/2.0 + eps)
        
        return ret
    
    def identify_asymunit(self,pts,res):
        N = pts.shape[0]
        
        cres = n.cos(res)
        
#         cosd = pts.dot(pts.T)
        
        Rs = self.get_rotations()
        
        ret = n.zeros(N,dtype=n.int32)
        
        ret[0] = 1
        in_asym = [0]
        in_sym = []
        in_unk = range(1,N)
        for itNum in xrange(1,N):
            asym_pts = pts[in_asym,:]
            unk_pts = pts[in_unk,:]
            
            testI = unk_pts.dot(asym_pts.T).max(axis=1).argmax()
            testptI = in_unk[testI]
            cpt = unk_pts[testptI]
            found_sym = False
            for R in Rs:
                asym_cosds = asym_pts.dot(R.dot(cpt))
                nclose = n.sum(asym_cosds >= cres)
                if nclose:
                    found_sym = True
                    break

            if found_sym:
                ret[testptI] = -1
                in_sym.append(testptI)
            else:
                ret[testptI] = 1
                in_asym.append(testptI)
            in_unk = in_unk[0:testI] + in_unk[testI+1:] 


    def apply(self,M,normalize=True,kernel='sinc',kernelsize=3,rad=2):
        if self.symclass == '':
            return M

        if n.iscomplexobj(M):
            symRs = n.array(self.get_rotations(exclude_dsym=True), dtype=n.float32)

            N = M.shape[0]
            if rad*self.get_order()*N < 500: # VERY heuristic 
                symM = sincint.symmetrize_fspace_volume(M,rad,kernel,kernelsize,symRs=symRs)
                if self.symclass == 'd':
                    symR = n.array([ [ [ -1.0,   0,    0 ],
                                       [    0, 1.0,    0 ],
                                       [    0,   0, -1.0 ] ] ], dtype=n.float32)
                    symM = sincint.symmetrize_fspace_volume(symM,rad,kernel,kernelsize,symRs=symR)
            else:
                return density.real_to_fspace(self.apply(density.fspace_to_real(M),normalize=normalize))
        else:
            symRs = n.array(self.get_rotations(), dtype=n.float32)

            symM = sincint.symmetrize_volume(n.require(M,dtype=n.float32),symRs)
#             symRs = n.array(self.get_rotations(exclude_dsym=True), dtype=n.float32)
# 
#             symM = sincint.symmetrize_volume_z(M,symRs)
# 
#             if self.symclass == 'd':
#                 symR = n.array([ [ [ -1.0,     0,   0 ],
#                                    [    0,  -1.0,   0 ],
#                                    [    0,     0, 1.0 ] ] ], dtype=n.float32)
#                 symM = n.swapaxes(sincint.symmetrize_volume_z(n.swapaxes(symM,1,2),symR),1,2)
                
        if normalize:
            symM /= self.get_order()
             
        return symM

