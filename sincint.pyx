#cython: boundscheck=False

# Fast code for interpolation

# the interpolant should be stored as a sparse matrix. Either as a csr_matrix or as a custom version
# need to support totally empty columns (for points totally outside the circle)
# need to support replication without recomputation (for only computing half of the fft)
# need to be very fast at slicing
# slice generation time is irrelevant for lebedev (unless need to use richardson extrap but not even then)
# memory should be manageable (100pts * 100 angles = 10000 slices, each 128*128*5*5*5 = 2M*4 bytes per slice...
# need to somehow only store x, y, z filters seperately since they are seperable
# could store indeces of 5x5x5 neighborhood for each point in a slice, and the x,y,z filter..
# extract the values (this is slow, since copying) then apply the filters sequentially then sum

# for now just generate one slice operator (csr_matrix) from an angle 
# and generate a slice from that

# NOTE: NUMPY WILL USE THREADED MKL, WHICH DOES NOT PLAY NICE WITH MULTIPROCESSING.
# FOR MULTIPROCESSING TO GIVE A (LARGE) BENIFIT, export MKL_NUM_THREADS=1
# BUT THIS SLOWS DOWN MAJOR NUMPY OPERATIONS

import numpy as n
cimport numpy as n
from geom import *
import scipy.sparse as sp

DTYPE = n.float32
CDTYPE = n.complex64
ITYPE = n.int32
UITYPE = n.uint32
ctypedef n.float32_t DTYPE_t
ctypedef n.complex64_t CDTYPE_t
ctypedef int ITYPE_t
ctypedef unsigned int UITYPE_t

from libc.math cimport round,floor,ceil,sqrt

#from cython.parallel import parallel, prange, threadid
#cimport openmp
from libc.stdlib cimport abort, malloc, free

## ---------- LOOK UP TABLE PARAMETERS --------------------------------------------
# size of largest kernel is 2*lut_range
cdef int lut_range = 10 
# each interval of size 1 has lut_scale points
cdef unsigned int lut_scale = 1000
# the origin is located at lut_offset
cdef unsigned int lut_offset = lut_scale*lut_range
x = n.linspace(-lut_range,lut_range,lut_scale*(2*lut_range)+1)[:-1]

cdef DTYPE_t[:] sinclut = n.require(n.sinc(x),dtype=DTYPE)


# lanczoslut = n.sinc(x)*n.sinc(x/a) for a kernel with 2a taps
cdef DTYPE_t[:] lanczos3lut  = n.require(n.sinc(x)*n.sinc(x/1.5)*(n.abs(x) <= 1.5),dtype=DTYPE) 
cdef DTYPE_t[:] lanczos4lut  = n.require(n.sinc(x)*n.sinc(x/2.0)*(n.abs(x) <= 2.0),dtype=DTYPE) 
cdef DTYPE_t[:] lanczos5lut  = n.require(n.sinc(x)*n.sinc(x/2.5)*(n.abs(x) <= 2.5),dtype=DTYPE) 
cdef DTYPE_t[:] lanczos6lut  = n.require(n.sinc(x)*n.sinc(x/3.0)*(n.abs(x) <= 3.0),dtype=DTYPE)
cdef DTYPE_t[:] lanczos7lut  = n.require(n.sinc(x)*n.sinc(x/3.5)*(n.abs(x) <= 3.5),dtype=DTYPE) 
cdef DTYPE_t[:] lanczos8lut  = n.require(n.sinc(x)*n.sinc(x/4.0)*(n.abs(x) <= 4.0),dtype=DTYPE)
cdef DTYPE_t[:] lanczos9lut  = n.require(n.sinc(x)*n.sinc(x/4.5)*(n.abs(x) <= 4.5),dtype=DTYPE) 
cdef DTYPE_t[:] lanczos10lut = n.require(n.sinc(x)*n.sinc(x/5.0)*(n.abs(x) <= 5.0),dtype=DTYPE)

cdef DTYPE_t[:] quadlut = ((n.abs(x) <= 0.5) * (1-2*x**2) + ((n.abs(x)<1)*(n.abs(x)>0.5)) * 2* (1-n.abs(x))**2 ).astype(DTYPE)


## ------------ FUNCTIONS ----------------------------------------------
cdef void kernel_lut (DTYPE_t[:] x1, ITYPE_t[:,:] x2, DTYPE_t[:] lut, DTYPE_t[:] res) nogil:
    cdef unsigned int r,c,R,C
    cdef unsigned int lutI

    R = x2.shape[0]
    C = x2.shape[1]

    for r in xrange(R):
        lutI = <unsigned int>((x2[r,0] - x1[0]) * lut_scale)+lut_offset
        res[r] = lut[lutI]
        for c in xrange(1,C):
            lutI = <unsigned int>((x2[r,c] - x1[c]) * lut_scale)+lut_offset
            res[r] *= lut[lutI]

cdef void kernel_lanczos3 (DTYPE_t[:] x1, ITYPE_t[:,:] x2, DTYPE_t[:] res) nogil:
    kernel_lut(x1,x2,lanczos3lut,res)

cdef void kernel_lanczos4 (DTYPE_t[:] x1, ITYPE_t[:,:] x2, DTYPE_t[:] res) nogil:
    kernel_lut(x1,x2,lanczos4lut,res)

cdef void kernel_lanczos5 (DTYPE_t[:] x1, ITYPE_t[:,:] x2, DTYPE_t[:] res) nogil:
    kernel_lut(x1,x2,lanczos5lut,res)

cdef void kernel_lanczos6 (DTYPE_t[:] x1, ITYPE_t[:,:] x2, DTYPE_t[:] res) nogil:
    kernel_lut(x1,x2,lanczos6lut,res)

cdef void kernel_lanczos7 (DTYPE_t[:] x1, ITYPE_t[:,:] x2, DTYPE_t[:] res) nogil:
    kernel_lut(x1,x2,lanczos7lut,res)

cdef void kernel_lanczos8 (DTYPE_t[:] x1, ITYPE_t[:,:] x2, DTYPE_t[:] res) nogil:
    kernel_lut(x1,x2,lanczos8lut,res)

cdef void kernel_lanczos9 (DTYPE_t[:] x1, ITYPE_t[:,:] x2, DTYPE_t[:] res) nogil:
    kernel_lut(x1,x2,lanczos9lut,res)

cdef void kernel_lanczos10 (DTYPE_t[:] x1, ITYPE_t[:,:] x2, DTYPE_t[:] res) nogil:
    kernel_lut(x1,x2,lanczos10lut,res)

cdef void kernel_sinc (DTYPE_t[:] x1, ITYPE_t[:,:] x2, DTYPE_t[:] res) nogil:
    kernel_lut(x1,x2,sinclut,res)

cdef void kernel_linear (DTYPE_t[:] x1, ITYPE_t[:,:] x2, DTYPE_t[:] res) nogil:
    cdef unsigned int r, c, R, C

    R = x2.shape[0]
    C = x2.shape[1]

    for r in xrange(R):
        res[r] = 1.0 - x2[r,0] + (2*x2[r,0] - 1)*x1[0]
        for c in xrange(1,C):
            res[r] *= 1.0 - x2[r,c] + (2*x2[r,c] - 1)*x1[c]

cdef void kernel_quad (DTYPE_t[:] x1, ITYPE_t[:,:] x2, DTYPE_t[:] res) nogil:
    kernel_lut(x1,x2,quadlut,res)

# An interpolation kernel function f(x1,x2,res) computes the weights for the tap
# points x2 when evaluating at interpolation point x1  
ctypedef void (*kernfptr)(DTYPE_t[:], ITYPE_t[:,:], DTYPE_t[:]) nogil

cdef kernfptr get_kernel_func(kernel,kernsize):
    cdef kernfptr kernfunc = NULL
    if kernel == 'sinc':
        kernfunc = &kernel_sinc
    elif kernel == 'lanczos':
        if kernsize == 3:
            kernfunc = &kernel_lanczos3
        elif kernsize == 4:
            kernfunc = &kernel_lanczos4
        elif kernsize == 5:
            kernfunc = &kernel_lanczos5
        elif kernsize == 6:
            kernfunc = &kernel_lanczos6
        elif kernsize == 7:
            kernfunc = &kernel_lanczos7
        elif kernsize == 8:
            kernfunc = &kernel_lanczos8
        elif kernsize == 9:
            kernfunc = &kernel_lanczos9
        elif kernsize == 10:
            kernfunc = &kernel_lanczos10
        else:
            assert False, 'Lanczos only supported for sizes between 3 and 10'
    elif kernel == 'linear':
        kernfunc = &kernel_linear
        assert kernsize == 2, 'kernelsize must be 2 for a linear kernel'
    elif kernel == 'quad':
        kernfunc = &kernel_quad
        assert kernsize == 2, 'kernelsize must be 2 for a quad kernel'
    else:
        assert False, 'Unknown kernel requested'
    return kernfunc

cdef void mult(DTYPE_t[:,:] R, DTYPE_t[:,:] p, DTYPE_t[:,:] out) nogil:
    cdef unsigned int i, j
    cdef DTYPE_t v
    
    for k in xrange(p.shape[1]):
        for i in xrange(R.shape[0]):
            v = R[i,0]*p[0,k]
            for j in xrange(1,R.shape[1]):
                v += R[i,j]*p[j,k]
            out[i,k] = v

cdef void mult_vec(DTYPE_t[:,:] R, DTYPE_t[:] p, DTYPE_t[:] out) nogil:
    cdef unsigned int i, j
    cdef DTYPE_t v
    
    for i in xrange(R.shape[0]):
        v = R[i,0]*p[0]
        for j in xrange(1,R.shape[1]):
            v += R[i,j]*p[j]
        out[i] = v

cdef void mult_vec_ui(DTYPE_t[:,:] R, UITYPE_t[:] p, DTYPE_t[:] out) nogil:
    cdef unsigned int i, j
    cdef DTYPE_t v
    
    for i in xrange(R.shape[0]):
        v = R[i,0]*p[0]
        for j in xrange(1,R.shape[1]):
            v += R[i,j]*p[j]
        out[i] = v

ctypedef void (*truncfptr)(DTYPE_t[:], ITYPE_t[:]) nogil

cdef void round_vec(DTYPE_t[:] p, ITYPE_t[:] pi) nogil:
    cdef unsigned int i
    
    for i in xrange(p.shape[0]):
        pi[i] = <ITYPE_t>round(p[i])

cdef void floor_vec(DTYPE_t[:] p, ITYPE_t[:] pi) nogil:
    cdef unsigned int i
    
    for i in xrange(p.shape[0]):
        pi[i] = <ITYPE_t>floor(p[i])

def compute_interpolation_matrix(DTYPE_t[:,:,:] Rs, int N_dst, int N_src, float rad,
                                 kernel, int kernsize,
                                 DTYPE_t[:,:,:] symRs = None):
    """
    Compute the (sparse) interpolation matrix corresponding to a set of transformation matrices
    R using a particular kernel and radius in Fourier space.
    
    Returns P, where P represents a (N_src x N_src x N_src) -> (N_R x N_T) mapping
    where the output is in row-major order (ie, order='C' with all N_T coeffs corresponding to
    Rs[0] stored first).
    """
    
    cdef unsigned int si, ri, r, i, k

    cdef unsigned int imD = Rs.shape[2]
    cdef unsigned int intD = Rs.shape[1]
    
    cdef kernfptr kernfunc = get_kernel_func(kernel,kernsize)
    cdef truncfptr truncfunc
    if kernsize % 2 != 0:
        # odd kernel size - round the location to the nearest grid
        truncfunc = &round_vec
    else:  
        # even kernel size - floor the location to the nearest grid
        truncfunc = &floor_vec
    
    cdef DTYPE_t[:,:] im_pts = gencoords(N_dst,imD,rad)
    cdef unsigned int N_T = im_pts.shape[0]
    cdef unsigned int N_R = Rs.shape[0]
    cdef unsigned int N_sym = 1+symRs.shape[0] if symRs is not None else 1
    
    if N_sym > 1:
        assert symRs.shape[1] == intD
        assert symRs.shape[2] == intD

    cdef unsigned int intksize = kernsize**intD
    # interpolation tap points (about 0,0,0)
    cdef ITYPE_t[:,:] p = n.require(gencoords(kernsize,intD).reshape((intksize,intD)),dtype=ITYPE)+1
    # WARNING: This assumes that the model is stored with order='C'
    cdef ITYPE_t[:] strides = n.array([N_src**(intD-1-i) for i in range(intD)], dtype=ITYPE)
    # pidx is the linear index offsets of the tap points
    cdef ITYPE_t[:] pidx = n.require(n.dot(p, strides),dtype=ITYPE)
    
    # Allocate memory for the sparse matrix output
    cdef ITYPE_t[:] indptrs = n.empty(N_R*N_T+1, dtype=ITYPE)
    cdef ITYPE_t[:] indices = n.empty(N_R*N_T*intksize*N_sym, dtype=ITYPE)
    cdef DTYPE_t[:] vals = n.empty(N_R*N_T*intksize*N_sym, dtype=DTYPE)
    
    # Temporary storage for computation
    cdef DTYPE_t[:] kvals = n.empty(intksize, dtype=DTYPE)
    cdef DTYPE_t[:] point = n.empty(intD, dtype=DTYPE)
    cdef DTYPE_t[:] int_pt = n.empty(intD, dtype=DTYPE)
    cdef ITYPE_t[:] int_pti = n.empty(intD, dtype=ITYPE)

    cdef unsigned int spidx, rcount
    cdef ITYPE_t center
    cdef DTYPE_t scale = ((<DTYPE_t>N_src)/N_dst)*(n.sqrt(<DTYPE_t>N_dst)**(intD-imD))/N_sym
    cdef int inbounds, cpti
    
    cdef DTYPE_t[:,:] cR, cSR
    cdef DTYPE_t[:,:] cRtmp = n.empty((intD,imD),dtype=DTYPE)

    spidx = 0
    rcount = 0
    with nogil:
        for ri in xrange(N_R):
            cR = Rs[ri,:,:]
            for r in xrange(N_T):
                indptrs[rcount] = spidx
                rcount += 1
                
                for si in xrange(N_sym):
                    if si == 0:
                        cSR = cR
                    else:
                        mult(symRs[si-1,:,:],cR,cRtmp)
                        cSR = cRtmp

                    mult_vec(cSR,im_pts[r],int_pt)
                    truncfunc(int_pt,int_pti)

                    center = 0
                    for i in xrange(intD):
                        center += (int_pti[i] + N_src/2)*strides[i]
                        point[i] = int_pt[i] - int_pti[i]
                
                    kernfunc(point, p, kvals)
                    for k in xrange(intksize):
                        inbounds = 1
                        for i in xrange(intD):
                            cpti = int_pti[i] + p[k,i] + N_src/2
                            inbounds = inbounds and cpti >= 0 and cpti < N_src

                        if kvals[k] != 0 and inbounds:
                            indices[spidx] = pidx[k] + center
                            vals[spidx] = kvals[k]
                            spidx += 1
        indptrs[rcount] = spidx

        if scale != 1:
            # Needed due to the use of a unitary FFT and/or symmetry
            for r in xrange(spidx):
                vals[r] *= scale

    P = sp.csr_matrix( (vals[0:spidx], indices[0:spidx], indptrs),
                       (int(N_R*N_T),N_src**intD), dtype = n.float32 )
    return P

def map_fspace_coordinates(CDTYPE_t[:,:,:] V, DTYPE_t[:,:] pts,
                           kernel, int kernsize):
    output_ary = n.empty(pts.shape[0],dtype=CDTYPE)
    cdef CDTYPE_t[:] output = output_ary

    cdef kernfptr kernfunc = get_kernel_func(kernel,kernsize)
    cdef truncfptr truncfunc
    if kernsize % 2 != 0:
        # odd kernel size - round the location to the nearest grid
        truncfunc = &round_vec
    else:  
        # even kernel size - floor the location to the nearest grid
        truncfunc = &floor_vec

    cdef unsigned int intksize = kernsize**3
    cdef DTYPE_t[:] point = n.empty((3), dtype=DTYPE)
    cdef ITYPE_t[:,:] p = n.require(gencoords(kernsize,3).reshape((intksize,3)),dtype=ITYPE)+1
    cdef ITYPE_t[:] int_pti = n.empty((3), dtype=ITYPE)
    cdef UITYPE_t[:] vpti = n.empty((3), dtype=UITYPE)
    cdef DTYPE_t[:] vals = n.empty((intksize), dtype=DTYPE)

    cdef CDTYPE_t cV
    cdef unsigned int pi, i, k
    cdef int tmp

    with nogil:
        for pi in range(pts.shape[0]):
            truncfunc(pts[pi],int_pti)

            for i in xrange(3):
                point[i] = pts[pi,i] - int_pti[i]
                
            kernfunc(point, p, vals)

            cV = 0
            for k in xrange(intksize):
                inbounds = 1
                for i in xrange(3):
                    tmp = int_pti[i] + p[k,i] 
                    if (tmp < 0) or (tmp >= V.shape[i]):
                        inbounds = 0
                        break
                    vpti[i] = <UITYPE_t>tmp
                if inbounds:
                    cV += vals[k]*V[vpti[0],vpti[1],vpti[2]]
                    
            output [pi] = cV

    return output_ary
            

def symmetrize_fspace_volume(CDTYPE_t[:,:,:] V,
                             float rad, kernel, int kernsize,
                             DTYPE_t[:,:,:] symRs,
                             out_ary = None,
                             unsigned int nthreads = 0):
    cdef kernfptr kernfunc = get_kernel_func(kernel,kernsize)
    cdef truncfptr truncfunc
    if kernsize % 2 != 0:
        # odd kernel size - round the location to the nearest grid
        truncfunc = &round_vec
    else:  
        # even kernel size - floor the location to the nearest grid
        truncfunc = &floor_vec
    
    cdef unsigned int intksize = kernsize**3
    # interpolation tap points (about 0,0,0)
    cdef ITYPE_t[:,:] p = n.require(gencoords(kernsize,3).reshape((intksize,3)),dtype=ITYPE)+1

    cdef ITYPE_t N = V.shape[0] # THIS MUST BE A SIGNED TYPE
    assert V.shape[1] == N and V.shape[2] == N

#     cdef int maxthreads = openmp.omp_get_max_threads()
#     if nthreads == 0:
#         nthreads = maxthreads 
    nthreads = 1

    if out_ary is None:
        out_ary = n.empty_like(V)

    cdef CDTYPE_t[:,:,:] out = out_ary

    assert out.shape[0] == N
    assert out.shape[1] == N
    assert out.shape[2] == N
    
    assert symRs.shape[1] == 3 and symRs.shape[2] == 3
    
    cdef unsigned int rI, x, y, z, vi, i, k
    cdef DTYPE_t[:] point = n.empty((3), dtype=DTYPE)
    cdef DTYPE_t[:] pt = n.empty((3),dtype=DTYPE)
    cdef DTYPE_t[:] int_pt = n.empty((3),dtype=DTYPE)
    cdef ITYPE_t[:] int_pti = n.empty((3), dtype=ITYPE)
    cdef UITYPE_t[:] vpti = n.empty((3), dtype=UITYPE)
    cdef DTYPE_t[:] vals = n.empty((intksize), dtype=DTYPE)
    cdef unsigned int N_sym = symRs.shape[0]
    cdef unsigned int N2 = N**2
    cdef int tmp
    cdef int N_2 = N/2
    cdef DTYPE_t rad2_thresh = ((rad*N/2.0)+(kernsize/2)+1)**2
    cdef CDTYPE_t cV
    cdef int inbounds
    
    with nogil:
        for vi in xrange(N**3):
            x = vi/N2
            y = (vi % N2)/N
            z = (vi % N)
            pt[0] = <ITYPE_t>(x) - N_2
            pt[1] = <ITYPE_t>(y) - N_2
            pt[2] = <ITYPE_t>(z) - N_2
    
            if pt[0]**2 + pt[1]**2 + pt[2]**2 > rad2_thresh:
                out[x,y,z] = V[x,y,z]
                continue
            
            cV = V[x,y,z]
            for rI in xrange(N_sym):
                mult_vec(symRs[rI,:,:],pt,int_pt)
                truncfunc(int_pt,int_pti)
    
                for i in xrange(3):
                    point[i] = int_pt[i] - int_pti[i]
                    
                kernfunc(point, p, vals)
                for k in xrange(intksize):
                    inbounds = 1
                    for i in xrange(3):
                        tmp = int_pti[i] + p[k,i] + N_2 
                        if (tmp < 0) or (tmp >= N):
                            inbounds = 0
                            break
                        vpti[i] = <UITYPE_t>tmp
                    if inbounds:
                        cV += vals[k]*V[vpti[0],vpti[1],vpti[2]]

            out[x,y,z] = cV
    return out_ary

cdef DTYPE_t trilin_interp(DTYPE_t[:,:,:] V, DTYPE_t[:] p) nogil:
    cdef DTYPE_t x = p[0]
    cdef DTYPE_t y = p[1]
    cdef DTYPE_t z = p[2]
    cdef UITYPE_t px0 = <UITYPE_t>floor(x)
    cdef UITYPE_t py0 = <UITYPE_t>floor(y)
    cdef UITYPE_t pz0 = <UITYPE_t>floor(z)
    cdef UITYPE_t px1 = <UITYPE_t>ceil(x)
    cdef UITYPE_t py1 = <UITYPE_t>ceil(y)
    cdef UITYPE_t pz1 = <UITYPE_t>ceil(z)
    cdef DTYPE_t rx = x - px0
    cdef DTYPE_t ry = y - py0
    cdef DTYPE_t rz = z - pz0

    cdef DTYPE_t V000 = V[px0,py0,pz0]
    cdef DTYPE_t V001 = V[px0,py0,pz1] 
    cdef DTYPE_t V010 = V[px0,py1,pz0] 
    cdef DTYPE_t V011 = V[px0,py1,pz1] 
    cdef DTYPE_t V100 = V[px1,py0,pz0]
    cdef DTYPE_t V101 = V[px1,py0,pz1] 
    cdef DTYPE_t V110 = V[px1,py1,pz0] 
    cdef DTYPE_t V111 = V[px1,py1,pz1]
    
    cdef DTYPE_t V_00 = V000 * (1 - rx) + V100 * rx 
    cdef DTYPE_t V_01 = V001 * (1 - rx) + V101 * rx 
    cdef DTYPE_t V_10 = V010 * (1 - rx) + V110 * rx 
    cdef DTYPE_t V_11 = V011 * (1 - rx) + V111 * rx
    
    cdef DTYPE_t V__0 = V_00 * (1 - ry) + V_10 * ry 
    cdef DTYPE_t V__1 = V_01 * (1 - ry) + V_11 * ry
    
    return V__0 * (1 - rz) + V__1 * rz

def symmetrize_volume(DTYPE_t[:,:,:] V,
                      DTYPE_t[:,:,:] symRs,
                      out_ary = None,
                      unsigned int nthreads = 0):
    
    cdef unsigned int N = V.shape[0]
    assert V.shape[1] == N and V.shape[2] == N

    if out_ary is None:
        out_ary = n.empty_like(V)
        
    cdef DTYPE_t[:,:,:] out = out_ary

    assert out.shape[0] == N
    assert out.shape[1] == N
    assert out.shape[2] == N
    
    assert symRs.shape[1] == 3 and symRs.shape[2] == 3
    
#     cdef int maxthreads = openmp.omp_get_max_threads()
#     if nthreads == 0:
#         nthreads = maxthreads 
    nthreads = 1
    
    cdef unsigned int rI, x, y, z, vi
    cdef DTYPE_t[:,:] pt = n.empty((nthreads,3),dtype=DTYPE)
    cdef DTYPE_t[:,:] Rpt = n.empty((nthreads,3),dtype=DTYPE)
    cdef unsigned int N_sym = symRs.shape[0]
    cdef unsigned int thId
    cdef unsigned int N2 = N**2
    cdef DTYPE_t N_2 = N/2.0
    cdef DTYPE_t cV
    
    with nogil:
        thId = 0
        for vi in xrange(N**3):
#         for vi in prange(N**3,schedule='static',num_threads=nthreads,nogil=True):
#             thId = threadid()
            
            x = vi/N2
            y = (vi % N2)/N
            z = (vi % N)
            pt[thId,0] = x - N_2
            pt[thId,1] = y - N_2
            pt[thId,2] = z - N_2
            cV = V[x,y,z]
            for rI in xrange(N_sym):
                mult_vec(symRs[rI,:,:],pt[thId],Rpt[thId])
                Rpt[thId,0] += N_2
                Rpt[thId,1] += N_2
                Rpt[thId,2] += N_2
                if Rpt[thId,0] >= 0 and Rpt[thId,0] <= (N-1) and Rpt[thId,1] >= 0 and Rpt[thId,1] <= (N-1) and Rpt[thId,2] >= 0 and Rpt[thId,2] <= (N-1):
#                     cV = cV + trilin_interp(V,Rpt[thId])
                    cV += trilin_interp(V,Rpt[thId])
            out[x,y,z] = cV

    return out_ary

cdef DTYPE_t bilin_interp(DTYPE_t[:,:] V, DTYPE_t[:] p) nogil:
    cdef DTYPE_t x = p[0]
    cdef DTYPE_t y = p[1]
    cdef UITYPE_t px0 = <UITYPE_t>floor(x)
    cdef UITYPE_t py0 = <UITYPE_t>floor(y)
    cdef UITYPE_t px1 = <UITYPE_t>ceil(x)
    cdef UITYPE_t py1 = <UITYPE_t>ceil(y)
    cdef DTYPE_t rx = x - px0
    cdef DTYPE_t ry = y - py0

    cdef DTYPE_t V00 = V[px0,py0]
    cdef DTYPE_t V01 = V[px0,py1] 
    cdef DTYPE_t V10 = V[px1,py0]
    cdef DTYPE_t V11 = V[px1,py1] 
    
    cdef DTYPE_t V_0 = V00 * (1 - rx) + V10 * rx 
    cdef DTYPE_t V_1 = V01 * (1 - rx) + V11 * rx 
    
    return V_0 * (1 - ry) + V_1 * ry 

def symmetrize_volume_z(DTYPE_t[:,:,:] V,
                        DTYPE_t[:,:,:] symRs,
                        out_ary = None,
                        unsigned int nthreads = 0):
    
    cdef unsigned int N = V.shape[0]
    assert V.shape[1] == N and V.shape[2] == N

    if out_ary is None:
        out_ary = n.empty_like(V)
        
    cdef DTYPE_t[:,:,:] out = out_ary

    assert out.shape[0] == N
    assert out.shape[1] == N
    assert out.shape[2] == N
    
    assert (symRs.shape[1] == 3 and symRs.shape[2] == 3) or (symRs.shape[1] == 2 and symRs.shape[2] == 2)

#     cdef int maxthreads = openmp.omp_get_max_threads()
#     if nthreads == 0:
#         nthreads = maxthreads 
    nthreads = 1
    
    cdef unsigned int rI, x, y, z, vi
    cdef DTYPE_t[:,:] pt = n.empty((nthreads,2),dtype=DTYPE)
    cdef DTYPE_t[:,:] Rpt = n.empty((nthreads,2),dtype=DTYPE)
    cdef unsigned int N_sym = symRs.shape[0]
    cdef unsigned int thId
    cdef unsigned int N2 = N**2
    cdef DTYPE_t N_2 = N/2.0
    cdef DTYPE_t cV
    
    with nogil:
        thId = 0
        for vi in xrange(N**2):
#         for vi in prange(N**2,schedule='static',num_threads=nthreads,nogil=True):
#             thId = threadid()
            
            x = vi/N
            y = vi % N
            pt[thId,0] = x - N_2
            pt[thId,1] = y - N_2
            for z in xrange(N):
                cV = V[x,y,z]
                for rI in xrange(N_sym):
                    mult_vec(symRs[rI,:2,:2],pt[thId],Rpt[thId])
                    Rpt[thId,0] += N_2
                    Rpt[thId,1] += N_2
                    if Rpt[thId,0] >= 0 and Rpt[thId,0] <= (N-1) and Rpt[thId,1] >= 0 and Rpt[thId,1] <= (N-1):
    #                     cV = cV + trilin_interp(V,Rpt[thId])
                        cV += bilin_interp(V[:,:,z],Rpt[thId])
                out[x,y,z] = cV

    return out_ary

def gentrunctofull (N=128, rad=0.3):
    """ Generates a sparse matrix operator that maps truncated image fourier coefficients (R) back to a full N**2 vector """
    xy = gencoords(N,2)
    r2 = n.sum(xy**2,axis=1)
    active_xy = r2 < (rad*N/2.0)**2
    R = sum(active_xy)
    splil = sp.lil_matrix((N**2, R), dtype=n.float32)

    j=0
    for i,v in enumerate(active_xy):
        if v:
            splil[i, j] = 1.0
            j += 1
    
    spcsr = splil.tocsr()
    spcsr.eliminate_zeros()
    return spcsr
    
def genfulltotrunc (N=128, rad=0.3):
    """ Generates a sparse matrix operator that maps full N**2 vector into truncated image fourier coefficients (R) """
    return gentrunctofull(N,rad).T

