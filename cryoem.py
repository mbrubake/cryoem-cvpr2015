import numpy as n
from geom import gencoords
import scipy.ndimage.interpolation as spinterp
import scipy.ndimage.filters as spfilter

import pyximport; pyximport.install(setup_args={"include_dirs":n.get_include()},reload_support=True)
import sparsemul

def compute_density_moments(M,mu=None):
    N = M.shape[0]
    absM = (M**2).reshape((N**3,1))
    absM /= n.sum(absM) 
    coords = gencoords(N,3).reshape((N**3,3))

    if mu == None:
        wcoords = coords.reshape((N**3,3)) * absM
        mu = n.sum(wcoords,axis=0).reshape((1,3))

    wccoords = n.sqrt(absM/N**3) * (coords - mu)
    covar = n.dot(wccoords.T,wccoords)

    return mu, covar

def rotate_density(M,R,t=None, upsamp=1.0):
    assert len(M.shape) == 3

    N = M.shape[0]

    Nup = int(n.round(N*upsamp))
#     print "Upsampling by", upsamp, "to", Nup, "^3"
    coords = gencoords(Nup,3).reshape((Nup**3,3)) / float(upsamp)
    if t is None:
        interp_coords = n.transpose(n.dot(coords, R.T)).reshape((3,Nup,Nup,Nup)) + N/2
    else:
        interp_coords = n.transpose(n.dot(coords, R.T) + t).reshape((3,Nup,Nup,Nup)) + N/2
    out = spinterp.map_coordinates(M,interp_coords,order=1)

    return out

def align_density(M, upsamp=1.0):
    assert len(M.shape) == 3

    (mu,covar) = compute_density_moments(M)

    (w,V) = n.linalg.eigh(covar)
    idx = w.argsort()
    w = w[idx]
    V = V[:,idx]

    if n.linalg.det(V) < 0:
        # ensure we have a valid rotation
        V[:,0] *= -1

    out = rotate_density(M,V,mu,upsamp)

#    (mu,covar) = compute_density_moments(out)

    return out, V

def rotational_average(M,maxRadius=None, doexpand=False, normalize=True, return_cnt=False):
    N = M.shape[0]
    D = len(M.shape)
    
    assert D >= 2, 'Cannot rotationally average a 1D array'

    pts = gencoords(N,D).reshape((N**D,D))
    r = n.sqrt(n.sum(pts**2,axis=1)).reshape(M.shape)
    ir = n.require(n.floor(r),dtype='uint32')
    f = r - ir

    if maxRadius is None:
        maxRadius = n.ceil(n.sqrt(D)*N/D)

    if maxRadius < n.max(ir)+2:
        valid_ir = ir+1 < maxRadius
        ir = ir[valid_ir]
        f = f[valid_ir]
        M = M[valid_ir]

    if n.iscomplexobj(M):
        raps = 1.0j*n.bincount(ir, weights=(1-f)*M.imag, minlength=maxRadius) + \
                    n.bincount(ir+1, weights=f*M.imag, minlength=maxRadius)
        raps += n.bincount(ir, weights=(1-f)*M.real, minlength=maxRadius) + \
                n.bincount(ir+1, weights=f*M.real, minlength=maxRadius)
    else:
        raps = n.bincount(ir, weights=(1-f)*M, minlength=maxRadius) + \
               n.bincount(ir+1, weights=f*M, minlength=maxRadius)
    raps = raps[0:maxRadius]

    if normalize or return_cnt:
        cnt = n.bincount(ir, weights=(1-f), minlength=maxRadius) + \
              n.bincount(ir+1, weights=f, minlength=maxRadius)
        cnt = cnt[0:maxRadius]

    if normalize:
        raps[cnt <= 0] = 0
        raps[cnt > 0] /= cnt[cnt > 0]

    if doexpand:
        raps = rotational_expand(raps,N,D)
    
    if return_cnt:
        return raps, cnt
    else:
        return raps

def rotational_expand(vals,N,D,interp_order=1):
    interp_coords = n.sqrt(n.sum(gencoords(N,D).reshape((N**D,D))**2,axis=1)).reshape((1,) + D*(N,))
    if n.iscomplexobj(vals):
        rotexp = 1.0j*spinterp.map_coordinates(vals.imag, interp_coords, 
                                               order=interp_order, mode='nearest')
        rotexp += spinterp.map_coordinates(vals.real, interp_coords, 
                                           order=interp_order, mode='nearest')
    else:
        rotexp = spinterp.map_coordinates(vals, interp_coords, 
                                          order=interp_order, mode='nearest')
    return rotexp

def resize_ndarray(D,nsz,axes):
    zfs = tuple([float(nsz[i])/float(D.shape[i]) if i in axes else 1 \
                 for i in range(len(nsz))])
    sigmas = tuple([0.66/zfs[i] if i in axes else 0 \
                    for i in range(len(nsz))])
#    print zfs, sigmas, D.shape
#     print "blurring...", ; sys.stdout.flush()
    blurD = spfilter.gaussian_filter(D,sigma=sigmas,order=0,mode='constant')
#     print "zooming...", ; sys.stdout.flush()
    return spinterp.zoom(blurD,zfs,order=0)

def compute_fsc(VF1,VF2,maxrad,width=1.0,thresholds = [0.143,0.5]):
    assert VF1.shape == VF2.shape
    N = VF1.shape[0]
    
    r = n.sqrt(n.sum(gencoords(N,3).reshape((N,N,N,3))**2,axis=3))
    
    prev_rad = -n.inf
    fsc = []
    rads = []
    resInd = len(thresholds)*[None]
    for i,rad in enumerate(n.arange(1.5,maxrad*N/2.0,width)):
        cxyz = n.logical_and(r >= prev_rad,r < rad)
        cF1 = VF1[cxyz] 
        cF2 = VF2[cxyz]
        
        if len(cF1) == 0:
            break
        
        cCorr = n.vdot(cF1,cF2) / n.sqrt(n.vdot(cF1,cF1)*n.vdot(cF2,cF2))
        
        for j,thr in enumerate(thresholds):
            if cCorr < thr and resInd[j] is None:
                resInd[j] = i
        fsc.append(cCorr.real)
        rads.append(rad/(N/2.0))
        prev_rad = rad

    fsc = n.array(fsc)
    rads = n.array(rads)

    resolutions = []
    for rI,thr in zip(resInd,thresholds):
        if rI is None:
            resolutions.append(rads[-1])
        elif rI == 0:
            resolutions.append(n.inf)
        else:
            x = (thr - fsc[rI])/(fsc[rI-1] - fsc[rI])
            resolutions.append(x*rads[rI-1] + (1-x)*rads[rI])
    
    
    return rads, fsc, thresholds, resolutions


# So the key is to make sure that the image is zero at the nyquist frequency (index n/2)
# The interpolation idea is to assume that the actual function f(x,y) is band-limited i.e.
# made up of exactly the frequency components in the FFT. Since we are interpolating in frequency space, 
# The assumption is that in frequency space the signal F(wx,wy) is band-limited. 
# This means that it's fourier transform should have components less than the nyquist frequency.
# But the fourier transform of F(wx,wy) is ~f(x,y) since FFT and iFFT are same. So f(x,y) must be nonzero at the nyquist frequency (and preferrably even less than that) which means in image space, the n/2 row and n/2 column (and n/2 page). 
# since the image will be zero at the edges once some windowing (circular or hamming etc) is applied,
# we can just fftshift the image since translations do not change the FFT except by phase. This makes the nyquist components
# zero and everything is fine and dandy. Even linear iterpolation works then, except it leaves ghosting.
  
def getslices (V, SLOP, res=None):
    vV = V.reshape((-1,))

    assert vV.shape[0] == SLOP.shape[1]

    if res is None:
        res = n.zeros(SLOP.shape[0],dtype=vV.dtype)
    else:
        assert res.shape[0] == SLOP.shape[0]
        assert len(res.shape) == 1 or res.shape[1] == 1
        assert res.dtype == vV.dtype
        res[:] = 0

    if n.iscomplexobj(vV):    
        sparsemul.spdot(SLOP, vV.real, res.real)
        sparsemul.spdot(SLOP, vV.imag, res.imag)
    else:
        sparsemul.spdot(SLOP, vV, res)
        
    return res
    
# 3D Densities
# ===============================================================================================    

def window (v, func='hanning', params=None):
    """ applies a windowing function to the 3D volume v (inplace, as reference) """
    
    N = v.shape[0]
    D = v.ndim
    if any( [ d != N for d in list(v.shape) ] ) or D != 3:
        raise Exception("Error: Volume is not Cube.")
    
    def apply_seperable_window (v, w):
        v *= n.reshape(w,(-1,1,1))
        v *= n.reshape(w,(1,-1,1))
        v *= n.reshape(w,(1,1,-1))
    
    if func=="hanning":
        w = n.hanning(N)
        apply_seperable_window(v,w)
    elif func=='hamming':
        w = n.hamming(N)
        apply_seperable_window(v,w)
    elif func=='gaussian':
        raise Exception('Unimplimented')
    elif func=='circle':
        c = gencoords(N,3)
        if params==None:
            r = N/2 -1
        else:
            r = params[0]*(N/2*1)
        v *= (n.sum(c**2,1)  < ( r ** 2 ) ).reshape((N,N,N))
    elif func=='box':
        v[:,0,0] = 0.0
        v[0,:,0] = 0.0
        v[0,0,:] = 0.0
    else:
        raise Exception("Error: Window Type Not Supported")

def generate_phantom_density(N,window,sigma,num_blobs,seed=None):
    if seed is not None:
        n.random.seed(seed)
    M = n.zeros((N,N,N),dtype=n.float32)

    coords = gencoords(N,3).reshape((N**3,3))
    inside_window = n.sum(coords**2,axis=1).reshape((N,N,N)) < window**2

    curr_c = n.array([0.0, 0.0 ,0.0])
    curr_n = 0
    while curr_n < num_blobs:
        csigma = sigma*n.exp(0.25*n.random.randn())
        radM = n.sum((coords - curr_c.reshape((1,3)))**2,axis=1).reshape((N,N,N))
        inside = n.logical_and(radM < (3*csigma)**2,inside_window)
#        M[inside] = 1
        M[inside] += n.exp(-0.5*(radM[inside]/csigma**2))
        curr_n += 1

        curr_dir = n.random.randn(3)
        curr_dir /= n.sum(curr_dir**2)
        curr_c += 2.0*csigma*curr_dir
        curr_w = n.sqrt(n.sum(curr_c**2))
        while curr_w > window:
            curr_n_dir = curr_c/curr_w
            curr_r_dir = (2*n.dot(curr_dir,curr_n_dir))*curr_n_dir - curr_dir
            curr_c = curr_n_dir + (curr_w - window)*curr_r_dir
            curr_w = n.sqrt(n.sum(curr_c**2))

    return M
