import numpy as n

import cryoops as cops
import geom

class FixedPlanarDomain():
    def __init__(self,pts,res):
        self.pts = pts
        self.resolution = res
        self.sqdist_mat = self.get_sqdist_mat(self)
        self.dim = self.pts.shape[1]
        if pts.shape[0] == 1:
            self.pt_resolution = res
        else:  
            sqdist = n.copy(self.get_sqdist_mat())
            n.fill_diagonal(sqdist, n.inf)
            self.pt_resolution = n.sqrt(sqdist.min(axis=1))

    def __len__(self):
        return self.pts.shape[0]

    def __eq__(self,other):
        return type(self) == type(other) and len(self) == len(other) \
                  and self.dim == other.dim \
                  and self.resolution == other.resolution \
                  and n.all(self.pts == other.pts)

    def __ne__(self,other):
        return not self.__eq__(other)

    def get_pts(self,inds = None):
        if inds is None:
            return self.pts
        else:
            return self.pts[inds,:]

    def get_pt_resolution(self,inds = None):
        if len(self) == 1:
            return self.resolution
        else:
            if inds is None:
                return self.pt_resolution
            else:
                return self.pt_resolution[inds]

    def get_sqdist_mat(self,other = None,curr_inds = None, other_inds = None):
        if other is None:
            ret = self.sqdist_mat
            if curr_inds is not None:
                ret = ret[curr_inds,:]
            if other_inds is not None:
                ret = ret[:,other_inds]
            return ret
        else:
            self_pts = self.get_pts(curr_inds)
            other_pts = other.get_pts(other_inds)
            D = self_pts.shape[1]
            err = self_pts.reshape((-1,1,D)) - other_pts.reshape((1,-1,D))

            return n.sum(err**2,axis=2)

    def compute_operator(self,interp_params,inds=None):
        pts = self.get_pts(inds)
        return cops.compute_shift_phases(pts,interp_params['N'],interp_params['rad'])

class FixedDirectionalDomain():
    def __init__(self,dirs,res):
        self.dirs = dirs
        self.resolution = res
        self.dim = self.dirs.shape[1] - 1

    def __len__(self):
        return self.dirs.shape[0]

    def __eq__(self,other):
        return type(self) == type(other) and len(self) == len(other) \
                 and self.dim == other.dim \
                 and self.resolution == other.resolution \
                 and n.all(self.dirs == other.dirs)

    def __ne__(self,other):
        return not self.__eq__(other)

    def get_dirs(self,inds = None):
        if inds is None:
            return self.dirs
        else:
            return self.dirs[inds,:]

class FixedSphereDomain(FixedDirectionalDomain):
    def __init__(self,dirs,res,sym=None):
        FixedDirectionalDomain.__init__(self,dirs,res)
        self.sym = sym

    def compute_operator(self,interp_params,inds=None):
        if inds is None:
            dirs = self.dirs
        else:
            dirs = self.dirs[inds]

        return cops.compute_projection_matrix(dirs,sym=self.sym,**interp_params)
    
    def get_symmetry_order(self):
        if self.sym is None:
            return 1
        else:
            return self.sym.get_order()
        

class FixedCircleDomain(FixedDirectionalDomain):
    def __init__(self,theta,res):
        FixedDirectionalDomain.__init__(self,
                                        n.array([n.cos(theta),
                                                 n.sin(theta.ravel())]).T,
                                        res)
        self.theta = theta

    def compute_operator(self,interp_params,inds=None):
        if inds is None:
            theta = self.theta
        else:
            theta = self.theta[inds]

        N = interp_params['N']
        kern = interp_params['kern']
        kernsize = interp_params['kernsize']
        rad = interp_params['rad']
        zeropad = interp_params.get('zeropad',0)
        N_src = N if zeropad == 0 else N + 2*int(zeropad*(N/2))
        return cops.compute_inplanerot_matrix(theta,N,kern,kernsize,rad,N_src, onlyRs = interp_params.get('onlyRs', False))
    
class FixedSO3Domain():
    def __init__(self,dirs,thetas,res,sym=None):
        self.dirs = dirs
        self.thetas = thetas
        self.resolution = res
        self.sym = sym
    
    def __len__(self):
        return self.dirs.shape[0] * len(self.thetas) 

    def compute_operator(self,interp_params,inds=None):
        if inds is None:
            Rs = n.array([[geom.rotmat3D_dir(d,t)[:,0:2] for t in self.thetas] for d in self.dirs])
            Rs = Rs.reshape((-1,3,2))        
        else:
            N_I = len(self.thetas)
            Rs = n.array([geom.rotmat3D_dir(self.dirs[i/N_I],self.thetas[n.mod(i,N_I)])[:,0:2] for i in inds])

        return cops.compute_projection_matrix(Rs,sym=self.sym,projdirtype='rots',**interp_params)

    def get_symmetry_order(self):
        if self.sym is None:
            return 1
        else:
            return self.sym.get_order()

