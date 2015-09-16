import numpy as n
from threading import Thread, Lock
from Queue import Queue
import time,  multiprocessing

from cryoem import getslices

import pyximport; pyximport.install(setup_args={"include_dirs":n.get_include()},reload_support=True)
import objective_kernels

from objectives.likelihood import UnknownRSKernel


class UnknownRSThreadedCPUKernel(UnknownRSKernel):
    def __init__(self):
        UnknownRSKernel.__init__(self)

        self.numthreads = None
        self.threads = None
        self.q = Queue()

        self.G_lock = Lock()
        self.sigma_lock = Lock()

    def setup(self,params,diagout,statout,ostream):
        UnknownRSKernel.setup(self, params, diagout, statout, ostream)

        numthreads = params.get('num_threads','auto')
        
        det_numthreads = multiprocessing.cpu_count()
        if numthreads == 'auto':
            numthreads = n.inf
        use_numthreads = min(det_numthreads,numthreads)
        print "Detected {0} cores, using {1} threads".format(det_numthreads,use_numthreads)

        if self.threads == None:
            self.numthreads = use_numthreads
            self.threads = [Thread(target=self.worker) for i in range(self.numthreads)]
            for th in self.threads:
                th.daemon = True 
                th.start()

    def precompute_projections(self,fM):
        if self.using_precomp_slicing:
            # precompute all possible slices
            getslices(fM.reshape((-1,)), self.slice_ops, res=self.slices)
            self.precomp_slices = self.slices.reshape((-1, self.N_T))
        else:
            # do on-the-fly slicing
            self.precomp_slices = None

    def worker(self):
        g_tmp = None
        lcl_sigma2_est = None
        lcl_correlation = None
        lcl_power = None
        lcl_G = None
        workspace = None
        while True:
            idxs, fM, res, compute_grad = self.q.get()

            sigma2 = self.inlier_sigma2_trunc 
            inlier_const = self.inlier_const - res['totallike_logscale']

            if lcl_sigma2_est is None or lcl_sigma2_est.shape[0] != self.N_T:
                lcl_sigma2_est = n.zeros(self.N_T,dtype=n.float64)
            else:
                lcl_sigma2_est[:] = 0

            if lcl_correlation is None or lcl_correlation.shape[0] != self.N_T:
                lcl_correlation = n.zeros(self.N_T,dtype=n.float64)
            else:
                lcl_correlation[:] = 0

            if lcl_power is None or lcl_power.shape[0] != self.N_T:
                lcl_power = n.zeros(self.N_T,dtype=n.float64)
            else:
                lcl_power[:] = 0

            # Result buffers
            like = res['like']
            Evar_like = res['Evar_like']

            if compute_grad:
                if lcl_G is None or lcl_G.shape != self.G.shape:
                    lcl_G = n.zeros_like(self.G)
                else:
                    lcl_G[:] = 0

            for idx in idxs:
                tic = time.time()
                slice_ops, envelope, \
                W_R_sampled, sampleinfo_R, slices_sampled, slice_inds, \
                W_I_sampled, sampleinfo_I, rotd_sampled, rotc_sampled, \
                W_S_sampled, sampleinfo_S, S_sampled = \
                    self.prep_operators(fM,idx,res=res)

                N_slices = slices_sampled.shape[0]
                
                log_W_R = n.log(W_R_sampled)
                log_W_I = n.log(W_I_sampled)
                log_W_S = n.log(W_S_sampled)

                if compute_grad:
                    if g_tmp is None or g_tmp.shape[0] < N_slices or g_tmp.shape[1] != self.N_T:
                        g_tmp = n.empty((N_slices,self.N_T), dtype=self.G_datatype)
                    else:
                        g_tmp[:] = 0.0
                    g = g_tmp[0:N_slices]
                    g[:] = 0
                else:
                    g = None
                res['kern_timing']['prep'][idx] = time.time() - tic

                tic = time.time()
                if len(W_I_sampled) == 1:
                    like[idx], (cphi_S,cphi_R), csigma2_est, ccorrelation, cpower, workspace = \
                         objective_kernels.doimage_RS(slices_sampled, \
                                             S_sampled, envelope, \
                                             rotc_sampled.reshape((-1,)), rotd_sampled.reshape((-1,)), \
                                             log_W_S, log_W_R, \
                                             sigma2, g, workspace )
                    cphi_I = n.array([0.0])
                else:
                    like[idx], (cphi_S,cphi_I,cphi_R), csigma2_est, ccorrelation, cpower, workspace = \
                         objective_kernels.doimage_RIS(slices_sampled, \
                                             S_sampled, envelope, \
                                             rotc_sampled, rotd_sampled, \
                                             log_W_S, log_W_I, log_W_R, \
                                             sigma2, g, workspace )
                res['kern_timing']['work'][idx] = time.time() - tic

                tic = time.time()
                # like[idx] is the negative log likelihood of the image
                like[idx] += self.inlier_like_trunc[idx]

                # Evar_like[idx] is the expected error
                Evar_like[idx] = (csigma2_est.sum() + self.imgpower_trunc[idx]) / self.N**2
                
                lcl_sigma2_est += csigma2_est
                lcl_correlation += ccorrelation
                lcl_power += cpower

                like[idx] += inlier_const

                if compute_grad:
                    if self.using_precomp_slicing:
                        lcl_G[slice_inds] += g
                    else:
                        lcl_G += slice_ops.T.dot(g.reshape((-1,))).reshape(lcl_G.shape)
                res['kern_timing']['proc'][idx] = time.time() - tic

                tic = time.time()
                self.store_results(idx, 1, \
                      cphi_R,sampleinfo_R, \
                      cphi_I,sampleinfo_I, \
                      cphi_S,sampleinfo_S, res, \
                      logspace_phis = True)
                res['kern_timing']['store'][idx] = time.time() - tic

            if compute_grad:
                self.G_lock.acquire()
                self.G += lcl_G
                self.G_lock.release()

            self.sigma_lock.acquire()
            res['sigma2_est'][self.truncmask] += lcl_sigma2_est/self.minibatch['N_M']
            res['correlation'][self.truncmask] += lcl_correlation/self.minibatch['N_M']
            res['power'][self.truncmask] += lcl_power/self.minibatch['N_M']
            self.sigma_lock.release()

            self.q.task_done()

    def eval(self, fM, compute_gradient=True, M=None):
        tic = time.time()

        N_M = self.minibatch['N_M']

        outputs = self.get_result_struct()
        outputs['like_timing'] = {}

        if compute_gradient:
            self.G[:] = 0
        outputs['like_timing']['setup'] = time.time() - tic

        tic = time.time()
        self.precompute_projections(fM)
        outputs['like_timing']['slice'] = time.time() - tic

        tic = time.time()
        numJobs = min(N_M,3*self.numthreads)
        #numJobs = int(self.numthreads + self.numthreads/2)
        imsPerJob = int(n.ceil(float(N_M)/numJobs))
        for jobId in xrange(numJobs):
            idxs = range(imsPerJob*jobId, \
                         min(imsPerJob*(jobId+1),N_M))
            self.q.put(( idxs, fM, outputs, compute_gradient ))
        outputs['like_timing']['queue'] = time.time() - tic

        tic = time.time()
        self.q.join()
        outputs['like_timing']['join'] = time.time() - tic
        outputs['kern_timing'] = dict([(k,n.sum(v)/self.numthreads) for k,v in outputs['kern_timing'].iteritems()])

        # compute gradient of likelihood
        if compute_gradient:
            tic = time.time()
            if self.using_precomp_slicing:
                dLdfM = self.slice_ops.T.dot(self.G.reshape((-1,))).reshape((self.N,self.N,self.N))
                dLdfM /= N_M
            else:
                dLdfM = self.G/N_M
            outputs['like_timing']['unslice'] = time.time() - tic

        outputs['N_R'] = self.N_R
        outputs['N_I'] = self.N_I
        outputs['N_S'] = self.N_S
        outputs['N_Total'] = self.N_R*self.N_I*self.N_S
        outputs['N_R_sampled_total'] = float(n.sum(outputs['N_R_sampled']))/self.N_R
        outputs['N_I_sampled_total'] = float(n.sum(outputs['N_I_sampled']))/self.N_I
        outputs['N_S_sampled_total'] = float(n.sum(outputs['N_S_sampled']))/self.N_S
        outputs['N_Total_sampled_total'] = float(n.sum(outputs['N_Total_sampled']))/(self.N_R*self.N_I*self.N_S)

        L = outputs['like'].sum(dtype=n.float64)/N_M
        outputs['L'] = L

        if compute_gradient:
            return L, dLdfM, outputs
        else:
            return L, outputs
