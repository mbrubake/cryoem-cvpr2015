#cython: boundscheck=False

import numpy as n
cimport numpy as n
from libc.math cimport exp, log, log1p, isfinite

DTYPE = n.float64
ctypedef double DTYPE_t
ctypedef int ITYPE_t

cdef DTYPE_t dot(DTYPE_t[:] p, DTYPE_t[:] q) nogil:
    cdef DTYPE_t v
    cdef unsigned int i
    cdef unsigned int N = p.shape[0]

    v = p[0]*q[0]
    for i in xrange(1,N):
        v += p[i]*q[i]
    return v

cdef DTYPE_t quad_form(DTYPE_t[:] q, DTYPE_t[:,:] R, DTYPE_t[:] p) nogil:
    cdef unsigned int i, j
    cdef DTYPE_t v, out
    
    out = 0
    for i in xrange(R.shape[0]):
        v = R[i,0]*p[0]
        for j in xrange(1,R.shape[1]):
            v += R[i,j]*p[j]
        out += v*q[i]
    return out

cdef void mult(DTYPE_t[:,:] R, DTYPE_t[:] p, DTYPE_t[:] out) nogil:
    cdef unsigned int i, j
    cdef DTYPE_t v
    
    for i in xrange(R.shape[0]):
        v = R[i,0]*p[0]
        for j in xrange(1,R.shape[1]):
            v += R[i,j]*p[j]
        out[i] = v

cdef DTYPE_t logaddexp(DTYPE_t a, DTYPE_t b) nogil:
    if not isfinite(a):
        return b
    elif not isfinite(b):
        return a
    elif a > b:
        return a + log1p(exp(b-a))
    else:
        return b + log1p(exp(a-b))

cdef DTYPE_t logsumexp(DTYPE_t[:] a) nogil:
    cdef DTYPE_t a_max, a_sum
    cdef unsigned int i
    cdef unsigned int N = a.shape[0]

    if N == 1:
        return a[0]
    elif N == 2:
        return logaddexp(a[0],a[1])
    else:
        a_max = a[0]
        for i in xrange(1,N):
            if a[i] > a_max:
                a_max = a[i]
    
        a_sum = exp(a[0] - a_max)
        for i in xrange(1,N):
            a_sum += exp(a[i] - a_max)
    
        return a_max + log(a_sum)

def compute_fisher_kernel(DTYPE_t[:,:] dirs1, DTYPE_t[:,:] dirs2,
                          DTYPE_t[:] vals,
                          DTYPE_t[:,:,:] Rs, 
                          DTYPE_t kappa, int chiral_flip, int normalize,
                          int logspace,
                          out_ary):
    cdef int apply_kern = vals is not None

    cdef unsigned int r, c, bufI
    cdef unsigned int N1, N2, dim, N_R, N_Total
    cdef DTYPE_t cosang, maxcosang, logkval, nrm
    cdef DTYPE_t[:] d1, d2, buf
    cdef DTYPE_t[:,:] out
    cdef DTYPE_t ninf = -n.infty
    
    if not apply_kern:
        assert not normalize

    assert dirs1.shape[1] == dirs2.shape[1]
    dim = dirs1.shape[1]
    
    if Rs is None:
        N_R = 0
    else:
        N_R = Rs.shape[0]
        assert Rs.shape[1] == dim and Rs.shape[2] == dim
    
    N_Total = 1 + N_R
    if chiral_flip:
        N_Total *= 2

    N1 = dirs1.shape[0]
    N2 = dirs2.shape[0]
    if apply_kern:
        assert N2 == vals.shape[0]
        if out_ary is None:
            out_ary = n.empty((N1,),dtype=DTYPE)
        else:
            assert out_ary.shape[0] == N1
            assert out_ary.shape[1] == 1
        if logspace:
            buf = n.empty(N2,dtype=DTYPE)
        out = out_ary.reshape((N1,1))
    else:
        if out_ary is None:
            out_ary = n.empty((N1,N2),dtype=DTYPE)
        else:
            assert out_ary.shape[0] == N1
            assert out_ary.shape[1] == N2
        out = out_ary

    with nogil:
        nrm = 0
        for r in xrange(N1):
            d1 = dirs1[r]

            if apply_kern and not logspace:
                out[r,0] = 0

            for c in xrange(N2):
                d2 = dirs2[c]
                
                if apply_kern:
                    if logspace and not isfinite(vals[c]):
                        buf[c] = ninf
                        continue
                    elif (not logspace) and vals[c] == 0:
                        continue
                
                cosang = dot(d1,d2)
                maxcosang = cosang
                if chiral_flip:
                    if -cosang > maxcosang:
                        maxcosang = -cosang

                for i in xrange(N_R):
                    cosang = quad_form(d1,Rs[i],d2)
                    if cosang > maxcosang:
                        maxcosang = cosang
                    if chiral_flip:
                        if -cosang > maxcosang:
                            maxcosang = -cosang
                logkval = (maxcosang - 1.0)*kappa

                if apply_kern:
                    if logspace:
                        buf[c] = logkval + vals[c]
                    else:
                        out[r,0] += vals[c]*exp(logkval)
                else:
                    if logspace:
                        out[r,c] = logkval
                    else:
                        out[r,c] = exp(logkval)

            if apply_kern and logspace:
                out[r,0] = logsumexp(buf)

            if normalize and not logspace:
                nrm += out[r,0] 
                
        if normalize:
            if logspace:
                nrm = logsumexp(out[:,0])
                for r in xrange(N1):
                    out[r,0] -= nrm
            else:
                for r in xrange(N1):
                    out[r,0] /= nrm

    return out_ary

# cdef void qchiral_flip(DTYPE_t[:] q, DTYPE_t[:] out) nogil:
#     out[0] =  q[1]
#     out[1] = -q[0]
#     out[2] =  q[3]
#     out[3] = -q[2]
#     
# cdef void qmult(DTYPE_t[:] p, DTYPE_t[:] q, DTYPE_t[:] out) nogil:
#     out[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
#     out[1] = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
#     out[2] = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
#     out[3] = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
# 
# def compute_bingham_kernel(DTYPE_t[:,:] quats1, DTYPE_t[:,:] quats2,
#                            DTYPE_t[:] vals,
#                            DTYPE_t[:,:] quats_sym, 
#                            DTYPE_t kappa, int chiral_flip, int normalize,
#                            int logspace,
#                            out_ary):
#     cdef int apply_kern = vals is not None
# 
#     cdef unsigned int r, c, bufI
#     cdef unsigned int N1, N2, N_R, N_Total
#     cdef DTYPE_t csq, maxcsq, logkval, nrm
#     cdef DTYPE_t[:] q1, q2, buf, qtmp, q2tmp
#     cdef DTYPE_t[:,:] out
#     cdef DTYPE_t ninf = -n.infty
#     
#     if not apply_kern:
#         assert not normalize
# 
#     assert quats1.shape[1] == 4
#     assert quats2.shape[1] == 4
#     
#     if quats_sym is None:
#         N_R = 0
#     else:
#         qtmp = n.empty(3,dtype=DTYPE)
#         N_R = quats_sym.shape[0]
#         assert quats_sym.shape[1] == 4
#     
#     N_Total = 1 + N_R
#     if chiral_flip:
#         q2tmp = n.empty(3,dtype=DTYPE)
#         N_Total *= 2
# 
#     N1 = quats1.shape[0]
#     N2 = quats2.shape[0]
#     if apply_kern:
#         assert N2 == vals.shape[0]
#         if out_ary is None:
#             out_ary = n.empty((N1,),dtype=DTYPE)
#         else:
#             assert out_ary.shape[0] == N1
#             assert out_ary.shape[1] == 1
#         if logspace:
#             buf = n.empty(N2,dtype=DTYPE)
#         out = out_ary.reshape((N1,1))
#     else:
#         if out_ary is None:
#             out_ary = n.empty((N1,N2),dtype=DTYPE)
#         else:
#             assert out_ary.shape[0] == N1
#             assert out_ary.shape[1] == N2
#         out = out_ary
# 
#     with nogil:
#         nrm = 0
#         for r in xrange(N1):
#             q1 = quats1[r]
# 
#             if apply_kern and not logspace:
#                 out[r,0] = 0
# 
#             for c in xrange(N2):
#                 q2 = quats2[c]
#                 
#                 if apply_kern:
#                     if logspace and not isfinite(vals[c]):
#                         buf[c] = ninf
#                         continue
#                     elif (not logspace) and vals[c] == 0:
#                         continue
#                 
#                 csq = dot(q1,q2)**2
#                 maxcsq = csq
#                 if chiral_flip:
#                     qchiral_flip(q2,q2tmp)
#                     csq = dot(q1,q2tmp)**2
#                     if csq > maxcsq:
#                         maxcsq = csq
# 
#                 for i in xrange(N_R):
#                     qmult(quats[i], q2, qtmp)
#                     csq = dot(q1,qtmp)**2
#                     if csq > maxcsq:
#                         maxcsq = csq
#                     if chiral_flip:
#                         qchiral_flip(qtmp,q2tmp)
#                         csq = dot(q1,q2tmp)**2
#                         if csq > maxcsq:
#                             maxcsq = csq
#                 logkval = (maxcsq - 1.0)*kappa
# 
#                 if apply_kern:
#                     if logspace:
#                         buf[c] = logkval + vals[c]
#                     else:
#                         out[r,0] += vals[c]*exp(logkval)
#                 else:
#                     if logspace:
#                         out[r,c] = logkval
#                     else:
#                         out[r,c] = exp(logkval)
# 
#             if apply_kern and logspace:
#                 out[r,0] = logsumexp(buf)
# 
#             if normalize and not logspace:
#                 nrm += out[r,0] 
#                 
#         if normalize:
#             if logspace:
#                 nrm = logsumexp(out[:,0])
#                 for r in xrange(N1):
#                     out[r,0] -= nrm
#             else:
#                 for r in xrange(N1):
#                     out[r,0] /= nrm
# 
#     return out_ary

