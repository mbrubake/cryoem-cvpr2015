import geom
import numpy as n

import pyximport; pyximport.install(setup_args={"include_dirs":n.get_include()},reload_support=True)
import sincint

precomputed_Rs = {}
def compute_projection_matrix(projdirs,N,kern,kernsize,rad,projdirtype='dirs',sym=None, onlyRs=False, **kwargs):
    projdirs = n.asarray(projdirs,dtype=n.float32)
    if projdirtype == 'dirs':
        # Input is a set of projection directions
        dirhash = hash(projdirs.tostring())
        if onlyRs and dirhash in precomputed_Rs:
            Rs = precomputed_Rs[dirhash]
        else:
            Rs = n.vstack([geom.rotmat3D_dir(d)[:,0:2].reshape((1,3,2)) for d in projdirs])
            if onlyRs:
                precomputed_Rs[dirhash] = Rs
    elif projdirtype == 'rots':
        # Input is a set of rotation matrices mapping image space to protein space
        Rs = projdirs
    else:
        assert False, 'Unknown projdirtype, must be either dirs or rots'

    if sym is None:
        symRs = None
    else:
        symRs = n.vstack([ n.require(R,dtype=n.float32).reshape((1,3,3)) for R in sym.get_rotations()])

    if onlyRs:
        return Rs
    else:
        return sincint.compute_interpolation_matrix(Rs,N,N,rad,kern,kernsize,symRs)

precomputed_RIs = {}
def compute_inplanerot_matrix(thetas,N,kern,kernsize,rad,N_src=None,onlyRs = False):
    dirhash = hash(thetas.tostring())
    if N_src is None:
        N_src = N
        scale = 1
    else:
        scale = float(N_src)/N
    if onlyRs and dirhash in precomputed_RIs:
        Rs = precomputed_RIs[dirhash]
    else:
        Rs = n.vstack([scale*geom.rotmat2D(n.require(th,dtype=n.float32)).reshape((1,2,2)) for th in thetas])
        if onlyRs:
            precomputed_RIs[dirhash] = Rs
    if onlyRs:
        return Rs
    else:
        return sincint.compute_interpolation_matrix(Rs,N,N_src,rad,kern,kernsize,None)

def compute_shift_phases(pts,N,rad):
    xy = geom.gencoords(N,2,rad)
    N_T = xy.shape[0]
    N_S = pts.shape[0]

    S = n.empty((N_S,N_T),dtype=n.complex64)
    for (i,(sx,sy)) in enumerate(pts):
        S[i] = n.exp(2.0j*n.pi/N * (xy[:,0] * sx + xy[:,1] * sy))

    return S

def compute_premultiplier(N, kernel, kernsize, scale=512):
    krange = N/2
    koffset = (N/2)*scale

    x = n.arange(-scale*krange,scale*krange)/float(scale)

    if kernel == 'lanczos':
        a = kernsize/2
        k = n.sinc(x)*n.sinc(x/a)*(n.abs(x) <= a)
    elif kernel == 'sinc': 
        a = kernsize/2.0
        k = n.sinc(x)*(n.abs(x) <= a)
    elif kernel == 'linear':
        assert kernsize == 2
        k = n.maximum(0.0, 1 - n.abs(x))
    elif kernel == 'quad':
        assert kernsize == 3
        k = (n.abs(x) <= 0.5) * (1-2*x**2) + ((n.abs(x)<1)*(n.abs(x)>0.5)) * 2* (1-n.abs(x))**2
    else:
        assert False, 'Unknown kernel type'

    sk = n.fft.fftshift(n.fft.ifft(n.fft.ifftshift(k))).real
    premult = 1.0/(N*sk[(koffset-krange):(koffset+krange)])
    
    return premult

if __name__ == '__main__':
    
    kern = 'sinc'
    kernsize = 3
    
    N = 128
    
    pm1 = compute_premultiplier(N,kern,kernsize,512)
    pm2 = compute_premultiplier(N,kern,kernsize,8192)
    
    print n.max(n.abs(pm1-pm2))


