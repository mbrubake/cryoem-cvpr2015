#cython: boundscheck=False

# Code ported from
# https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h#L1010

import numpy as n
cimport numpy as n
from cython.parallel import *
import scipy.sparse as sp

def spdot (spmat, n.ndarray[n.float32_t, ndim=1] Xx, n.ndarray[n.float32_t, ndim=1] Yx):
    """ Computes the dot product sum of a sparse matrix in CSR format with a numpy vector. All float32.
    Equivalent to Y += spmat.dot(X) """
    
    assert (spmat.format == 'csr')
        
    cdef unsigned int n_row = spmat.shape[0]
    cdef n.ndarray[n.int32_t, ndim=1] Ap = spmat.indptr 
    cdef n.ndarray[n.int32_t, ndim=1] Aj = spmat.indices
    cdef n.ndarray[n.float32_t, ndim=1] Ax = spmat.data
    
    cdef unsigned int i, jj
    cdef n.float32_t sum
    
    with nogil:    
        for i in xrange(n_row):
            sum = Yx[i]
            jj = Ap[i]
            while jj < Ap[i+1]:
                sum = sum + Ax[jj] * Xx[Aj[jj]]
                jj = jj + 1
            Yx[i] = sum
        
    return


#/*
#* Compute Y += A*X for CSR matrix A and dense vectors X,Y
#*
#*
#* Input Arguments:
#* I n_row - number of rows in A
#* I n_col - number of columns in A
#* I Ap[n_row+1] - row pointer
#* I Aj[nnz(A)] - column indices
#* T Ax[nnz(A)] - nonzeros
#* T Xx[n_col] - input vector
#*
#* Output Arguments:
#* T Yx[n_row] - output vector
#*
#* Note:
#* Output array Yx must be preallocated
#*
#* Complexity: Linear. Specifically O(nnz(A) + n_row)
#*
#*/
#template <class I, class T>
#void csr_matvec(const I n_row,
#const I n_col,
#const I Ap[],
#const I Aj[],
#const T Ax[],
#const T Xx[],
#T Yx[])
#{
#    for(I i = 0; i < n_row; i++){
#        T sum = Yx[i];
#        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
#            sum += Ax[jj] * Xx[Aj[jj]];
#        }
#        Yx[i] = sum;
#    }
#}


