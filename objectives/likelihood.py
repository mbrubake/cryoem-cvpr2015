from objective import Objective

import numpy as n

import sys, time

from util import FiniteRunningSum, logsumexp

from cryoio.imagestack import FourierStack
from cryoio import ctf
import cryoem, quadrature, density, cryoops

from symmetry import get_symmetryop
from geom import gencoords
from cryoem import getslices

class UnknownRSLikelihood(Objective):
    def __init__(self):
        Objective.__init__(self,False)

    def setup(self,params,diagout,statout,ostream):
        Objective.setup(self,params,diagout,statout,ostream)

        if params['kernel'] == 'multicpu':
            from cpu_kernel import UnknownRSThreadedCPUKernel
            self.kernel = UnknownRSThreadedCPUKernel()
        else:
            assert False
        self.kernel.setup(params,diagout,statout,ostream)
        
    def get_sigma2_map(self,nu,model,rotavg=True):
        N = self.cryodata.N
        N_D = float(self.cryodata.N_D_Train)
        num_batches = float(self.cryodata.num_batches)
        base_sigma2 = self.cryodata.get_noise_std()**2

        mean_sigma2 = self.error_history.get_mean().reshape((N,N))
        mean_mask = self.mask_history.get_mean().reshape((N,N))
        mask_w = self.mask_history.get_wsum() * (N_D / num_batches)
        
        if rotavg:
            mean_sigma2 = cryoem.rotational_average(mean_sigma2,normalize=True,doexpand=True)
            mean_mask = cryoem.rotational_average(mean_mask,normalize=False,doexpand=True)

        obsw = mask_w * mean_mask
        map_sigma2 = (mean_sigma2 * obsw + nu * base_sigma2) / (obsw + nu)

        assert n.all(n.isfinite(map_sigma2))

        if model == 'coloured':
            map_sigma2 = map_sigma2
        elif model == 'white':
            map_sigma2 = n.mean(map_sigma2)
        else:
            assert False, 'model must be one of white or coloured'

        return map_sigma2

    def get_envelope_map(self,sigma2,rho,env_lb=None,env_ub=None,minFreq=None,bfactor=None,rotavg=True):
        N = self.cryodata.N
        N_D = float(self.cryodata.N_D_Train)
        num_batches = float(self.cryodata.num_batches)
        psize = self.params['pixel_size']

        mean_corr = self.correlation_history.get_mean().reshape((N,N))
        mean_power = self.power_history.get_mean().reshape((N,N))
        mean_mask = self.mask_history.get_mean().reshape((N,N))
        mask_w = self.mask_history.get_wsum() * (N_D / num_batches)
        
        if rotavg:
            mean_corr = cryoem.rotational_average(mean_corr,normalize=True,doexpand=True)
            mean_power = cryoem.rotational_average(mean_power,normalize=True,doexpand=True)
            mean_mask = cryoem.rotational_average(mean_mask,normalize=False,doexpand=True)

        if isinstance(sigma2,n.ndarray):
            sigma2 = sigma2.reshape((N,N))

        if bfactor is not None:
            coords = gencoords(N,2).reshape((N**2,2))
            freqs = n.sqrt(n.sum(coords**2,axis=1))/(psize*N)
            prior_envelope = ctf.envelope_function(freqs,bfactor).reshape((N,N))
        else:
            prior_envelope = 1.0

        obsw = (mask_w * mean_mask / sigma2)
        exp_env = (mean_corr * obsw + prior_envelope*rho) / (mean_power * obsw + rho)
        
        if minFreq is not None:
            # Only consider envelope parameters for frequencies above a threshold
            minRad = minFreq*2.0*psize
    
            _, _, minRadMask = gencoords(N, 2, minRad, True)
            
            exp_env[minRadMask.reshape((N,N))] = 1.0
        
        if env_lb is not None or env_ub is not None:
            n.clip(exp_env,env_lb,env_ub,out=exp_env)

        return exp_env


    def get_rmse(self):
        return n.sqrt(self.get_sigma2_mle().mean())

    def get_sigma2_mle(self,noise_model='coloured'):
        N = self.cryodata.N
        sigma2 = self.error_history.get_mean()
        mean_mask = self.mask_history.get_mean()
#         mse = mean_mask*sigma2 + (1-mean_mask)*self.cryodata.data['imgvar_freq']
        mse = mean_mask*sigma2 + (1-mean_mask)*self.cryodata.data_var
        if noise_model == 'coloured':
            return mse.reshape((N,N))
        elif noise_model == 'white':
            return mse.mean()
            

    def get_envelope_mle(self,rotavg=False):
        N = self.cryodata.N

        mean_corr = self.correlation_history.get_mean()
        mean_power = self.power_history.get_mean()
        mean_mask = self.mask_history.get_mean()
        
        if rotavg:
            mean_corr = cryoem.rotational_average(mean_corr.reshape((N,N)),doexpand=True)
            mean_power = cryoem.rotational_average(mean_power.reshape((N,N)),doexpand=True)
            mean_mask = cryoem.rotational_average(mean_mask.reshape((N,N)),doexpand=True)

        obs_mask = mean_mask > 0
        exp_env = n.ones_like(mean_corr)
        exp_env[obs_mask] = (mean_corr[obs_mask] / mean_power[obs_mask])

        return exp_env.reshape((N,N))
        

    def set_samplers(self,sampler_R,sampler_I,sampler_S):
        self.kernel.set_samplers(sampler_R,sampler_I,sampler_S)

    def set_dataset(self,cryodata):
        Objective.set_dataset(self,cryodata)
        self.kernel.set_dataset(cryodata)

        self.error_history = FiniteRunningSum(second_order=False)
        self.correlation_history = FiniteRunningSum(second_order=False)
        self.power_history = FiniteRunningSum(second_order=False)
        self.mask_history = FiniteRunningSum(second_order=False)
        
    def set_data(self,cparams,minibatch):
        Objective.set_data(self,cparams,minibatch)
        self.kernel.set_data(cparams,minibatch)
        
    def eval(self,M=None, compute_gradient=True, fM=None, **kwargs):
        tic_start = time.time()
        
        if self.kernel.slice_premult is not None:
            pfM = density.real_to_fspace(self.kernel.slice_premult * M)
        else:
            pfM = density.real_to_fspace(M)
        pmtime = time.time() - tic_start

        ret = self.kernel.eval(fM=pfM,M=None,compute_gradient=compute_gradient)
        
        if not self.minibatch['test_batch'] and not kwargs.get('intermediate',False):
            tic_record = time.time()
            curr_var = ret[-1]['sigma2_est']
            assert n.all(n.isfinite(curr_var))
            if self.error_history.N_sum != self.cryodata.N_batches:
                self.error_history.setup(curr_var,self.cryodata.N_batches,allow_decay=False)
            self.error_history.set_value(self.minibatch['id'],curr_var)

            curr_corr = ret[-1]['correlation']
            assert n.all(n.isfinite(curr_corr))
            if self.correlation_history.N_sum != self.cryodata.N_batches:
                self.correlation_history.setup(curr_corr,self.cryodata.N_batches,allow_decay=False)
            self.correlation_history.set_value(self.minibatch['id'],curr_corr)

            curr_power = ret[-1]['power']
            assert n.all(n.isfinite(curr_power))
            if self.power_history.N_sum != self.cryodata.N_batches:
                self.power_history.setup(curr_power,self.cryodata.N_batches,allow_decay=False)
            self.power_history.set_value(self.minibatch['id'],curr_power)

            curr_mask = self.kernel.truncmask
            if self.mask_history.N_sum != self.cryodata.N_batches:
                self.mask_history.setup(n.require(curr_mask,dtype=n.float32),self.cryodata.N_batches,allow_decay=False)
            self.mask_history.set_value(self.minibatch['id'],curr_mask)
            ret[-1]['like_timing']['record'] = time.time() - tic_record
        
        if compute_gradient and self.kernel.slice_premult is not None:
            tic_record = time.time()
            ret = (ret[0],self.kernel.slice_premult * density.fspace_to_real(ret[1]),ret[2])
            ret[-1]['like_timing']['premult'] = pmtime + time.time() - tic_record
            
        ret[-1]['like_timing']['total'] = time.time() - tic_start
        
        return ret


class UnknownRSKernel:
    def __init__(self,factoredRI=False):
        self.slice_params = None
        self.slice_interp = None
        self.inplane_params = None
        self.inplane_interp = None
        self.proj_params = None
        self.proj_interp = None
        self.shift_params = None
        self.shift_interp = None

        self.rad = None
        self.factoredRI = None
        
        self.sampler_R = None
        self.sampler_I = None
        self.sampler_S = None
        
        self.G_datatype = n.complex64
        

    def set_samplers(self,sampler_R,sampler_I,sampler_S):
        self.sampler_R = sampler_R
        self.sampler_I = sampler_I
        self.sampler_S = sampler_S

    def setup(self,params,diagout,statout,ostream):
        # If there are more than this number of quadrature points, do OTF slicing
        # FIXME: Eventually do this adaptively based on the amount of memory and
        # effectiveness of IS because, once IS kicks in, OTF slicing may be faster.
        self.otf_thresh_RI = params.get('otf_thresh_RI',60000)
        self.otf_thresh_R = params.get('otf_thresh_R',5000)
        self.otf_thresh_I = params.get('otf_thresh_I',500)
        self.fspace_premult_stack_caching = params.get('interp_cache_fspace', True)
        
    def set_dataset(self,cryodata):
        self.cryodata = cryodata

        self.fspace_stack = FourierStack(self.cryodata.imgstack,
                                         caching = self.fspace_premult_stack_caching)

        self.quad_domain_RI = None
        self.quad_domain_R = None
        self.quad_domain_I = None
        self.quad_domain_S = None

        self.N = self.cryodata.N
        self.N_D = self.cryodata.N_D
        self.N_D_Train = self.cryodata.N_D_Train

        self.outlier_model = None

    def set_proj_quad(self,rad):
        # Get (and generate if needed) the quadrature scheme for slicing
        params = self.params

        tic = time.time()

        N = self.N

        quad_scheme_R = params.get('quad_type_R','sk97')
        quad_R = quadrature.quad_schemes[('dir',quad_scheme_R)]

        degree_R = params.get('quad_degree_R','auto')
        degree_I = params.get('quad_degree_I','auto')

        usFactor_R = params.get('quad_undersample_R',params.get('quad_undersample',1.0))
        usFactor_I = params.get('quad_undersample_I',params.get('quad_undersample',1.0))

        kern_R = params.get('interp_kernel_R',params.get('interp_kernel',None))
        kernsize_R = params.get('interp_kernel_size_R',params.get('interp_kernel_size',None))
        zeropad_R = params.get('interp_zeropad_R',params.get('interp_zeropad',0))
        dopremult_R = params.get('interp_premult_R',params.get('interp_premult',True))

        sym = get_symmetryop(params.get('symmetry',None)) if params.get('perfect_symmetry',True) else None

        maxAngle = quadrature.compute_max_angle(self.N,rad,usFactor_I)
        if degree_I == 'auto':
            degree_I = n.uint32(n.ceil(2.0 * n.pi / maxAngle))

        if degree_R == 'auto':
            degree_R,resolution_R = quad_R.compute_degree(N,rad,usFactor_R)

        resolution_R = max(0.5*quadrature.compute_max_angle(self.N,rad), resolution_R)
        resolution_I = max(0.5*quadrature.compute_max_angle(self.N,rad), 2.0*n.pi / degree_I)

        slice_params = { 'quad_type':quad_scheme_R, 'degree':degree_R, 
                         'sym':sym }
        inplane_params = { 'degree':degree_I }
        proj_params = { 'quad_type_R':quad_scheme_R, 'degree_R':degree_R, 
                         'sym':sym, 'degree_I':degree_I }
        interp_params_RI = { 'N':self.N, 'kern':kern_R, 'kernsize':kernsize_R, 'rad':rad, 'zeropad':zeropad_R, 'dopremult':dopremult_R }
        interp_change_RI = self.proj_interp != interp_params_RI
        
        transform_change = self.slice_interp is None or \
                        self.slice_interp['kern'] != interp_params_RI['kern'] or \
                        self.slice_interp['kernsize'] != interp_params_RI['kernsize'] or \
                        self.slice_interp['zeropad'] != interp_params_RI['zeropad']

        domain_change_R = self.slice_params != slice_params
        domain_change_I = self.inplane_params != inplane_params
        domain_change_RI = self.proj_params != proj_params  
        
        if domain_change_RI:
            proj_quad = {}

            proj_quad['resolution'] = max(resolution_R,resolution_I)
            proj_quad['degree_R'] = degree_R
            proj_quad['degree_I'] = degree_I
            proj_quad['symop'] = sym

            proj_quad['dir'],proj_quad['W_R'] = quad_R.get_quad_points(degree_R,proj_quad['symop'])
            proj_quad['W_R'] = n.require(proj_quad['W_R'], dtype=n.float32)

            proj_quad['thetas'] = n.linspace(0, 2.0*n.pi, degree_I, endpoint=False)
            proj_quad['thetas'] += proj_quad['thetas'][1]/2.0
            proj_quad['W_I'] = n.require((2.0*n.pi/float(degree_I))*n.ones((degree_I,)),dtype=n.float32)

            self.quad_domain_RI = quadrature.FixedSO3Domain( proj_quad['dir'],
                                                            -proj_quad['thetas'],
                                                             proj_quad['resolution'],
                                                             sym=sym)
            self.N_RI = len(self.quad_domain_RI)
            self.proj_quad = proj_quad
            self.proj_params = proj_params


            if domain_change_R:
                self.quad_domain_R = quadrature.FixedSphereDomain(proj_quad['dir'],
                                                                  proj_quad['resolution'],
                                                                  sym=sym)
                self.N_R = len(self.quad_domain_R)
                self.sampler_R.setup(params, self.N_D, self.N_D_Train, self.quad_domain_R)
                self.slice_params = slice_params

            if domain_change_I:
                self.quad_domain_I = quadrature.FixedCircleDomain(proj_quad['thetas'],
                                                                  proj_quad['resolution'])
                self.N_I = len(self.quad_domain_I)
                self.sampler_I.setup(params, self.N_D, self.N_D_Train, self.quad_domain_I)
                self.inplane_params = inplane_params

        if domain_change_RI or interp_change_RI:
            symorder = 1 if self.proj_quad['symop'] is None else self.proj_quad['symop'].get_order()
            print "  Projection Ops: %d (%d slice, %d inplane), " % (self.N_RI, self.N_R, self.N_I), ; sys.stdout.flush()
            if self.N_RI*symorder < self.otf_thresh_RI:
                self.using_precomp_slicing = True
                print "generated in", ; sys.stdout.flush()
                self.slice_ops = self.quad_domain_RI.compute_operator(interp_params_RI)
                print " {0} secs.".format(time.time() - tic)

                Gsz = (self.N_RI,self.N_T)
                self.G = n.empty(Gsz, dtype=self.G_datatype)
                self.slices = n.empty(n.prod(Gsz), dtype=n.complex64)
            else:
                self.using_precomp_slicing = False
                print "generating OTF."
                self.slice_ops = None
                self.G = n.empty((N,N,N),dtype=self.G_datatype)
                self.slices = None
            self.using_precomp_inplane = False
            self.inplane_ops = None
            
            self.proj_interp = interp_params_RI

        if transform_change:
            if dopremult_R:
                premult = cryoops.compute_premultiplier(self.N + 2*int(interp_params_RI['zeropad']*(self.N/2)), 
                                                        interp_params_RI['kern'],interp_params_RI['kernsize'])
                premult = premult.reshape((-1,1,1)) * premult.reshape((1,-1,1)) * premult.reshape((1,1,-1))
            else:
                premult = None
            self.slice_premult = premult
            self.slice_zeropad = interp_params_RI['zeropad']
            assert interp_params_RI['zeropad'] == 0,'Zero padding for slicing not yet implemented'

    def set_slice_quad(self,rad):
        # Get (and generate if needed) the quadrature scheme for slicing
        params = self.params

        tic = time.time()

        N = self.N
        degree_R = params.get('quad_degree_R','auto')
        quad_scheme_R = params.get('quad_type_R','sk97')
        sym = get_symmetryop(params.get('symmetry',None)) if params.get('perfect_symmetry',True) else None
        usFactor_R = params.get('quad_undersample_R',params.get('quad_undersample',1.0))

        kern_R = params.get('interp_kernel_R',params.get('interp_kernel',None))
        kernsize_R = params.get('interp_kernel_size_R',params.get('interp_kernel_size',None))
        zeropad_R = params.get('interp_zeropad_R',params.get('interp_zeropad',0))
        dopremult_R = params.get('interp_premult_R',params.get('interp_premult',True))

        quad_R = quadrature.quad_schemes[('dir',quad_scheme_R)]

        if degree_R == 'auto':
            degree_R,resolution_R = quad_R.compute_degree(N,rad,usFactor_R)
        resolution_R = max(0.5*quadrature.compute_max_angle(self.N,rad), resolution_R)

        slice_params = { 'quad_type':quad_scheme_R, 'degree':degree_R, 
                         'sym':sym }
        interp_params_R = { 'N':self.N, 'kern':kern_R, 'kernsize':kernsize_R, 'rad':rad, 'zeropad':zeropad_R, 'dopremult':dopremult_R }
        
        domain_change_R = slice_params != self.slice_params
        interp_change_R = self.slice_interp != interp_params_R
        transform_change = self.slice_interp is None or \
                        self.slice_interp['kern'] != interp_params_R['kern'] or \
                        self.slice_interp['kernsize'] != interp_params_R['kernsize'] or \
                        self.slice_interp['zeropad'] != interp_params_R['zeropad']

        if domain_change_R:
            slice_quad = {}

            slice_quad['resolution'] = resolution_R
            slice_quad['degree'] = degree_R
            slice_quad['symop'] = sym

            slice_quad['dir'],slice_quad['W'] = quad_R.get_quad_points(degree_R,slice_quad['symop'])
            slice_quad['W'] = n.require(slice_quad['W'], dtype=n.float32)

            self.quad_domain_R = quadrature.FixedSphereDomain(slice_quad['dir'],
                                                              slice_quad['resolution'],\
                                                              sym=sym)
            self.N_R = len(self.quad_domain_R)
            self.sampler_R.setup(params, self.N_D, self.N_D_Train, self.quad_domain_R)
            
            self.slice_quad = slice_quad
            self.slice_params = slice_params

        if domain_change_R or interp_change_R:
            symorder = 1 if self.slice_quad['symop'] is None else self.slice_quad['symop'].get_order()
            print "  Slice Ops: %d, " % self.N_R, ; sys.stdout.flush()
            if self.N_R*symorder < self.otf_thresh_R:
                self.using_precomp_slicing = True
                print "generated in", ; sys.stdout.flush()
                self.slice_ops = self.quad_domain_R.compute_operator(interp_params_R)
                print " {0} secs.".format(time.time() - tic)

                Gsz = (self.N_R,self.N_T)
                self.G = n.empty(Gsz, dtype=self.G_datatype)
                self.slices = n.empty(n.prod(Gsz), dtype=n.complex64)
            else:
                self.using_precomp_slicing = False
                print "generating OTF."
                self.slice_ops = None
                self.G = n.empty((self.N,self.N,self.N),dtype=self.G_datatype)
                self.slices = None
            self.slice_interp = interp_params_R

        if transform_change:
            if dopremult_R:
                premult = cryoops.compute_premultiplier(self.N + 2*int(interp_params_R['zeropad']*(self.N/2)), 
                                                        interp_params_R['kern'],interp_params_R['kernsize'])
                premult = premult.reshape((-1,1,1)) * premult.reshape((1,-1,1)) * premult.reshape((1,1,-1))
            else:
                premult = None
            self.slice_premult = premult
            self.slice_zeropad = interp_params_R['zeropad']
            assert interp_params_R['zeropad'] == 0,'Zero padding for slicing not yet implemented'

    def set_inplane_quad(self,rad):
        # Get (and generate if needed) the quadrature scheme for inplane rotation 
        params = self.params

        tic = time.time()
        
        degree_I = params.get('quad_degree_I','auto')
        usFactor_I = params.get('quad_undersample_I',params.get('quad_undersample',1.0))

        kern_I = params.get('interp_kernel_I',params.get('interp_kernel',None))
        kernsize_I = params.get('interp_kernel_size_I',params.get('interp_kernel_size',None))
        zeropad_I = params.get('interp_zeropad_I',params.get('interp_zeropad',0))
        dopremult_I = params.get('interp_premult_I',params.get('interp_premult',True))

        maxAngle = quadrature.compute_max_angle(self.N,rad,usFactor_I)
        if degree_I == 'auto':
            degree_I = n.uint32(n.ceil(2.0 * n.pi / maxAngle))
        resolution_I = max(0.5*quadrature.compute_max_angle(self.N,rad), 2.0*n.pi / degree_I)

        inplane_params = { 'degree':degree_I }
        interp_params_I = { 'N':self.N, 'kern':kern_I, 'kernsize':kernsize_I, 'rad':rad, 'zeropad':zeropad_I, 'dopremult':dopremult_I }
        
        domain_change_I = self.inplane_params != inplane_params
        interp_change_I = self.inplane_interp != interp_params_I
        transform_change = self.inplane_interp is None or \
                        self.inplane_interp['kern'] != interp_params_I['kern'] or \
                        self.inplane_interp['kernsize'] != interp_params_I['kernsize'] or \
                        self.inplane_interp['zeropad'] != interp_params_I['zeropad']

        if domain_change_I:
            inplane_quad = {}
            inplane_quad['resolution'] = resolution_I
            inplane_quad['thetas'] = n.linspace(0, 2.0*n.pi, degree_I, endpoint=False)
            inplane_quad['thetas'] += inplane_quad['thetas'][1]/2.0
            inplane_quad['W'] = n.require((2.0*n.pi/float(degree_I))*n.ones((degree_I,)),dtype=n.float32)

            self.quad_domain_I = quadrature.FixedCircleDomain(inplane_quad['thetas'],
                                                              inplane_quad['resolution'])
            
            self.N_I = len(self.quad_domain_I)
            self.sampler_I.setup(params, self.N_D, self.N_D_Train, self.quad_domain_I)
            self.inplane_quad = inplane_quad
            self.inplane_params = inplane_params

        if domain_change_I or interp_change_I:
            print "  Inplane Ops: %d, " % self.N_I, ; sys.stdout.flush()
            if self.N_I < self.otf_thresh_I:
                self.using_precomp_inplane = True
                print "generated in", ; sys.stdout.flush()
                self.inplane_ops = self.quad_domain_I.compute_operator(interp_params_I)
                print " {0} secs.".format(time.time() - tic)
            else:
                self.using_precomp_inplane = False
                print "generating OTF."
                self.inplane_ops = None
            self.inplane_interp = interp_params_I

                    
        if transform_change:
            if dopremult_I:
                premult = cryoops.compute_premultiplier(self.N + 2*int(interp_params_I['zeropad']*(self.N/2)),
                                                        interp_params_I['kern'],interp_params_I['kernsize'])
                premult = premult.reshape((-1,1)) * premult.reshape((1,-1))
            else:
                premult = None
            self.fspace_stack.set_transform(premult,interp_params_I['zeropad'])

    def set_shift_quad(self,rad):
        # Get (and generate if needed) the quadrature scheme for inplane shifts
        params = self.params
        
        tic = time.time()

        N = self.N
        quad_scheme = params.get('quad_type_S','hermite')
        shiftdegree = params.get('quad_degree_S','auto')
        shiftextent = params['quad_shiftextent']/params['pixel_size']
        shiftsigma = params['quad_shiftsigma']/params['pixel_size']
        shifttrunc = params.get('quad_shifttrunc','circ')
        usFactor = params.get('quad_undersample_S',params.get('quad_undersample',1.0))

        quad = quadrature.quad_schemes[('shift',quad_scheme)]
        if shiftdegree == 'auto':
            shiftdegree = quad.get_degree(N,rad,shiftsigma,shiftextent,usFactor)

        assert shiftdegree > 0

        shift_params = { 'quad_type':quad_scheme, 'degree':shiftdegree,
                         'shiftsigma':shiftsigma,
                         'shiftextent':shiftextent,
                         'shifttrunc':shifttrunc, }
        interp_params = {'N':N, 'rad':rad}
        domain_change = shift_params != self.shift_params
        interp_change = interp_params != self.shift_interp
        if domain_change:
            shift_quad = {}
            shift_quad['pts'], shift_quad['W'] = quad.get_quad_points(shiftdegree,shiftsigma,shiftextent,shifttrunc)
            shift_quad['resolution'] = shiftextent / shiftdegree

            self.quad_domain_S = quadrature.FixedPlanarDomain(shift_quad['pts'],
                                                              shift_quad['resolution'])
            self.N_S = len(self.quad_domain_S)
            self.sampler_S.setup(params, self.N_D, self.N_D_Train, self.quad_domain_S)
            self.shift_params = shift_params
            self.shift_quad = shift_quad

        if domain_change or interp_change:
            print "  Shift Ops: %d, generated in" % self.N_S, ; sys.stdout.flush()
            self.shift_ops = self.quad_domain_S.compute_operator(interp_params)
            print " {0} secs.".format(time.time() - tic)
            self.shift_interp = interp_params

    def set_data(self,cparams,minibatch):
        self.params = cparams
        self.minibatch = minibatch

        factoredRI = cparams.get('likelihood_factored_slicing',True)
        max_freq = cparams['max_frequency']
        psize = cparams['pixel_size']
        rad_cutoff = cparams.get('rad_cutoff', 1.0)
        rad = min(rad_cutoff,max_freq*2.0*psize)

        self.xy, self.trunc_xy, self.truncmask = gencoords(self.N, 2, rad, True)
        self.trunc_freq = n.require(self.trunc_xy / (self.N*psize), dtype=n.float32) 
        self.N_T = self.trunc_xy.shape[0]

        interp_change = self.rad != rad or self.factoredRI != factoredRI
        if interp_change:
            print "Iteration {0}: freq = {3}, rad = {1}, N_T = {2}".format(cparams['iteration'], rad, self.N_T, max_freq)
            self.rad = rad
            self.factoredRI = factoredRI

        # Setup the quadrature schemes
        if not factoredRI:
            self.set_proj_quad(rad)
        else:
            self.set_slice_quad(rad)
            self.set_inplane_quad(rad)

        # Check shift quadrature
        self.set_shift_quad(rad)
        
        # Setup inlier model
        self.inlier_sigma2 = cparams['sigma']**2
        base_sigma2 = self.cryodata.noise_var
        if isinstance(self.inlier_sigma2,n.ndarray):
            self.inlier_sigma2 = self.inlier_sigma2.reshape(self.truncmask.shape)
            self.inlier_sigma2_trunc = self.inlier_sigma2[self.truncmask != 0]
            self.inlier_const = (self.N_T/2.0)*n.log(2.0*n.pi) + 0.5*n.sum(n.log(self.inlier_sigma2_trunc))
        else:
            self.inlier_sigma2_trunc = self.inlier_sigma2 
            self.inlier_const = (self.N_T/2.0)*n.log(2.0*n.pi*self.inlier_sigma2)

        # Compute the likelihood for the image content outside of rad
        _,_,fspace_truncmask = gencoords(self.fspace_stack.get_num_pixels(), 2, rad*self.fspace_stack.get_num_pixels()/self.N, True)
        self.imgpower = n.empty((self.minibatch['N_M'],),dtype=density.real_t)
        self.imgpower_trunc = n.empty((self.minibatch['N_M'],),dtype=density.real_t)
        for idx,Idx in enumerate(self.minibatch['img_idxs']):
            Img = self.fspace_stack.get_image(Idx)
            self.imgpower[idx] = n.sum(Img.real**2) + n.sum(Img.imag**2)

            Img_trunc = Img[fspace_truncmask.reshape(Img.shape) == 0]
            self.imgpower_trunc[idx] = n.sum(Img_trunc.real**2) + n.sum(Img_trunc.imag**2)
        like_trunc = 0.5*self.imgpower_trunc/base_sigma2
        self.inlier_like_trunc = like_trunc
        self.inlier_const += ((self.N**2 - self.N_T)/2.0)*n.log(2.0*n.pi*base_sigma2)
        
        # Setup the envelope function
        envelope = self.params.get('exp_envelope',None)
        if envelope is not None:
            envelope = envelope.reshape((-1,))
            envelope = envelope[self.truncmask != 0]
            envelope = n.require(envelope,dtype=n.float32)
        else:
            bfactor = self.params.get('learn_like_envelope_bfactor',500.0)
            if bfactor is not None:
                freqs = n.sqrt(n.sum(self.trunc_xy**2,axis=1))/(psize*self.N)
                envelope = ctf.envelope_function(freqs,bfactor)
        self.envelope = envelope

    def get_result_struct(self):
        N_M = self.minibatch['N_M']

        res = { }

        for suff in ['S','I','R']:
            res['CV2_'+suff] = n.zeros(N_M)

        basesigma2 = self.cryodata.noise_var

        res['Evar_like'] = n.zeros(N_M)
        res['Evar_prior']= n.zeros(N_M)
#         res['Evar_prior'] = self.cryodata.data['imgpower'][self.minibatch['I']]/self.N**2

#         res['sigma2_est'] = n.empty_like(self.cryodata.data['imgvar_freq'])
        res['sigma2_est'] = n.empty((self.N**2,),dtype=density.real_t)
        res['sigma2_est'][self.truncmask != 0] = 0
        res['sigma2_est'][self.truncmask == 0] = basesigma2

        res['correlation'] = n.zeros_like(res['sigma2_est'])
        res['power'] = n.zeros_like(res['sigma2_est'])
        
        res['like'] = n.zeros(N_M)
        res['N_R_sampled'] = n.zeros(N_M,dtype=n.uint32)
        res['N_I_sampled'] = n.zeros(N_M,dtype=n.uint32)
        res['N_S_sampled'] = n.zeros(N_M,dtype=n.uint32)
        res['N_Total_sampled'] = n.zeros(N_M,dtype=n.uint32)
        
        # Divide by the normalization constant with sigma=noise_std to keep it from being huge
        res['totallike_logscale'] = (self.N**2/2.0)*n.log(2.0*n.pi*basesigma2)
        
        res['kern_timing'] = {'prep_sample_R':n.empty(N_M),'prep_sample_I':n.empty(N_M),'prep_sample_S':n.empty(N_M),
                              'prep_slice':n.empty(N_M), 'prep_rot_img':n.empty(N_M), 'prep_rot_ctf':n.empty(N_M),
                              'prep':n.empty(N_M),'work':n.empty(N_M),'proc':n.empty(N_M),'store':n.empty(N_M)}

        return res

    def prep_operators(self,fM,idx, slicing = True, res=None):
        
        Idx = self.minibatch['img_idxs'][idx]
        CIdx = self.minibatch['ctf_idxs'][idx]
        cCTF = self.cryodata.ctfstack.get_ctf(CIdx)
        Img = self.fspace_stack.get_image(Idx)
        
        factoredRI = self.factoredRI

        if not factoredRI:
            W_R = self.proj_quad['W_R']
            W_I = self.proj_quad['W_I']
        else:
            W_R = self.slice_quad['W']
            W_I = self.inplane_quad['W']
        W_S = self.shift_quad['W']

        envelope = self.envelope

        tic = time.time()
        samples_R, sampleweights_R = self.sampler_R.sample(Idx)
        if samples_R is None:
            N_R_sampled = self.N_R
            W_R_sampled = W_R
        else:
            N_R_sampled = len(samples_R)
            W_R_sampled = n.require(W_R[samples_R] * sampleweights_R, dtype = W_R.dtype)
        sampleinfo_R = N_R_sampled, samples_R, sampleweights_R
        if res is not None:
            res['kern_timing']['prep_sample_R'][idx] = time.time() - tic 

        tic = time.time()
        samples_I, sampleweights_I = self.sampler_I.sample(Idx)
        if samples_I is None:
            N_I_sampled = self.N_I
            W_I_sampled = W_I
        else:
            N_I_sampled = len(samples_I)
            W_I_sampled = n.require(W_I[samples_I] * sampleweights_I, dtype = W_I.dtype)
        sampleinfo_I = N_I_sampled, samples_I, sampleweights_I
        if res is not None:
            res['kern_timing']['prep_sample_I'][idx] = time.time() - tic 

        tic = time.time()
        samples_S, sampleweights_S = self.sampler_S.sample(Idx)
        if samples_S is None:
            N_S_sampled = self.N_S
            S_sampled = self.shift_ops 
            W_S_sampled = W_S
        else:
            N_S_sampled = len(samples_S)
            S_sampled = self.shift_ops[samples_S]
            W_S_sampled = n.require( W_S[samples_S] * sampleweights_S , dtype = W_S.dtype)
        sampleinfo_S = N_S_sampled, samples_S, sampleweights_S
        if res is not None:
            res['kern_timing']['prep_sample_S'][idx] = time.time() - tic 

        if slicing:
            if not factoredRI:
                if samples_R is None and samples_I is None:
                    full_samples = None
                else:
                    full_samples = []
                    it_R = xrange(self.N_R) if samples_R is None else samples_R
                    it_I = xrange(self.N_I) if samples_I is None else samples_I
                    for r in it_R:
                        full_samples += [(r,i) for i in it_I]
                    full_samples = n.array(full_samples)
                    samples_R = self.N_I*full_samples[:,0] + full_samples[:,1]
                    samples_I = n.array(0)

                W_R_sampled = (W_R_sampled.reshape((N_R_sampled,1)) * W_I_sampled.reshape((1,N_I_sampled))).reshape((N_R_sampled*N_I_sampled,))
                W_I_sampled = n.array([1.0], dtype = W_I.dtype)

                N_R_sampled = N_R_sampled*N_I_sampled
                N_I_sampled = 1

                if self.using_precomp_slicing:
                    slice_ops = self.slice_ops
                    if samples_R is None:
                        slices_sampled = self.precomp_slices
                    else:
                        slices_sampled = self.precomp_slices[samples_R]
                else:
                    slice_ops = self.quad_domain_RI.compute_operator(self.interp_params,samples_R)
                    slices_sampled = getslices(fM.reshape((-1,)), slice_ops).reshape((N_R_sampled,self.N_T))
                
                rotd_sampled = Img[self.truncmask.reshape(Img.shape)].reshape((N_I_sampled,self.N_T))
                rotc_sampled = cCTF.compute(self.trunc_freq).reshape((1,self.N_T))
            else:
                tic = time.time()
                if self.using_precomp_slicing:
                    slice_ops = self.slice_ops
                    if samples_R is None:
                        slices_sampled = self.precomp_slices
                    else:
                        slices_sampled = self.precomp_slices[samples_R]
                else:
                    slice_ops = self.quad_domain_R.compute_operator(self.slice_interp,samples_R)
                    slices_sampled = getslices(fM.reshape((-1,)), slice_ops).reshape((N_R_sampled,self.N_T))
                if res is not None:
                    res['kern_timing']['prep_slice'][idx] = time.time() - tic 

                tic = time.time()
                if samples_I is None:
                    rotc_sampled = cCTF.compute(self.trunc_freq,self.quad_domain_I.theta).T
                else:
                    rotc_sampled = cCTF.compute(self.trunc_freq,self.quad_domain_I.theta[samples_I]).T
                if res is not None:
                    res['kern_timing']['prep_rot_ctf'][idx] = time.time() - tic 

                # Generate the rotated versions of the current image
                if self.using_precomp_inplane:
                    if samples_I is None:
                        rotd_sampled = getslices(Img,self.inplane_ops).reshape((N_I_sampled,self.N_T))
                    else:
                        rotd_sampled = getslices(Img,self.inplane_ops).reshape((self.N_I,self.N_T))[samples_I]
                else:
                    inplane_ops = self.quad_domain_I.compute_operator(self.inplane_interp,samples_I)
                    rotd_sampled = getslices(Img,inplane_ops).reshape((N_I_sampled,self.N_T))
                if res is not None:
                    res['kern_timing']['prep_rot_img'][idx] = time.time() - tic 
        else:
            slice_ops = None
            slices_sampled = None

            rotc_sampled = None
            rotd_sampled = None

        return slice_ops, envelope, \
            W_R_sampled, sampleinfo_R, slices_sampled, samples_R, \
            W_I_sampled, sampleinfo_I, rotd_sampled, rotc_sampled, \
            W_S_sampled, sampleinfo_S, S_sampled
            
    def store_results(self, idx, isw, \
                      cphi_R, sampleinfo_R, \
                      cphi_I, sampleinfo_I, \
                      cphi_S, sampleinfo_S, res,
                      logspace_phis = False):
        Idx = self.minibatch['img_idxs'][idx]
        testImg = self.minibatch['test_batch']

        N_R_sampled = sampleinfo_R[0]
        N_I_sampled = sampleinfo_I[0]
        N_S_sampled = sampleinfo_S[0]
        if not self.factoredRI:
            cphi_R = cphi_R.reshape((N_R_sampled,N_I_sampled))
            if logspace_phis:
                cphi_I = logsumexp(cphi_R,axis=0)
                cphi_R = logsumexp(cphi_R,axis=1)
            else:
                cphi_I = n.sum(cphi_R,axis=0)
                cphi_R = n.sum(cphi_R,axis=1)

        self.sampler_R.record_update(Idx, sampleinfo_R[1], cphi_R, sampleinfo_R[2], isw, testImg, logspace = logspace_phis)
        self.sampler_I.record_update(Idx, sampleinfo_I[1], cphi_I, sampleinfo_I[2], isw, testImg, logspace = logspace_phis)
        self.sampler_S.record_update(Idx, sampleinfo_S[1], cphi_S, sampleinfo_S[2], isw, testImg, logspace = logspace_phis)

        res['N_R_sampled'][idx] = N_R_sampled
        res['N_I_sampled'][idx] = N_I_sampled
        res['N_S_sampled'][idx] = N_S_sampled
        res['N_Total_sampled'][idx] = N_R_sampled*N_I_sampled*N_S_sampled
        
        res['Evar_prior'][idx] = self.imgpower[idx]/self.N**2

        if logspace_phis:
            res['CV2_R'][idx] = n.exp(-logsumexp(2*cphi_R,dtype=n.float64))
            res['CV2_I'][idx] = n.exp(-logsumexp(2*cphi_I,dtype=n.float64))
            res['CV2_S'][idx] = n.exp(-logsumexp(2*cphi_S,dtype=n.float64))
        else:
            res['CV2_R'][idx] = (1.0/n.sum(cphi_R**2,dtype=n.float64))
            res['CV2_I'][idx] = (1.0/n.sum(cphi_I**2,dtype=n.float64))
            res['CV2_S'][idx] = (1.0/n.sum(cphi_S**2,dtype=n.float64))



