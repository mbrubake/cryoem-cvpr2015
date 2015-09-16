import time, os, sys
from datetime import datetime
from cryoio.imagestack import MRCImageStack, CombinedImageStack
from cryoio.ctfstack import CTFStack, CombinedCTFStack
from cryoio.dataset import CryoDataset

opj = os.path.join
from copy import copy, deepcopy

import numpy as n


from shutil import copyfile
from util import BackgroundWorker, Output, OutputStream, Params, format_timedelta, gitutil, FiniteRunningSum
import cryoem
from objectives import eval_objective, SumObjectives
from importancesampler.gaussian import FixedGaussianImportanceSampler
from importancesampler.fisher import FixedFisherImportanceSampler
import cPickle
import socket

from threading import Thread
from Queue import Queue

from optimizers.sagd import SAGDStep
from optimizers.sgd import SGDMomentumStep

from cryoio.mrc import writeMRC, readMRC

from symmetry import get_symmetryop
import density

# precond should ideally be set to inv(chol(H)) where H is the Hessian
def density2params(M,fM,xtype,grad_transform = False,precond = None):
    if xtype == 'real':
        if grad_transform:
            x0 = M if precond is None else M * precond
        else:
            x0 = M if precond is None else M / precond
    elif xtype == 'complex':
        if grad_transform:
            x0 = fM if precond is None else fM * precond
        else:
            x0 = fM if precond is None else fM / precond
    elif xtype == 'complex_coeff':
        if grad_transform:
            pfM = fM if precond is None else fM * precond
        else:
            pfM = fM if precond is None else fM / precond

        x0 = n.empty((2*fM.size,),dtype=density.real_t)
        x0[0:fM.size] = pfM.real.reshape((-1,))
        x0[fM.size:] = pfM.imag.reshape((-1,))
    elif xtype == 'complex_herm_coeff':
        assert precond is None, 'Unimplemented'

        N = fM.shape[0]
        NC = N/2 + 1
        startFreq = 1-(N%2)

        herm_freqs = fM[0:NC,:,:]
        if startFreq:
            herm_freqs += n.roll(n.roll(n.roll(fM[::-1, ::-1, ::-1], \
                                               1, axis=0), \
                                        1, axis=1), \
                                 1, axis=2)[0:NC,:,:].conj()
        else:
            herm_freqs += fM[::-1, ::-1, ::-1][0:NC,:,:].conj()

        if not grad_transform:
            herm_freqs *= 0.5

        x0 = n.empty((2*NC*N**2,),dtype=density.real_t)
        x0[0:NC*N**2] = herm_freqs.real.reshape((-1,))
        x0[NC*N**2:] = herm_freqs.imag.reshape((-1,))
 
    return x0


def param2density(x,xtype,sz,precond = None):
    if xtype == 'real':
        M, fM = x.reshape(sz), None
        if precond is not None:
            M = M * precond
    elif xtype == 'complex':
        M, fM = None, x.reshape(sz)
        if precond is not None:
            fM = fM * precond
    elif xtype == 'complex_coeff':
        M, fM = None, density.empty_cplx(sz)

        fM.real = x[0:fM.size].reshape(sz)
        fM.imag = x[fM.size:].reshape(sz)

        if precond is not None:
            fM *= precond
    elif xtype == 'complex_herm_coeff':
        assert precond is None, 'Unimplemented'

        M, fM = None, density.empty_cplx(sz)

        N = sz[0]
        NC = N/2 + 1
        startFreq = 1-(N%2)
        zeroFreq = N/2

        herm_freqs = n.empty((NC,N,N),dtype=density.complex_t)
        herm_freqs.real = x[0:NC*N**2].reshape(herm_freqs.shape)
        herm_freqs.imag = x[NC*N**2:].reshape(herm_freqs.shape)

        fM[0:NC,:,:] = herm_freqs
        if startFreq:
            fM[NC:,:,:] = n.roll(n.roll(herm_freqs[startFreq:zeroFreq,:,:][::-1,::-1,::-1].conj(), \
                                        1, axis=1), 1, axis=2)
        else:
            fM[NC:,:,:] = herm_freqs[startFreq:zeroFreq,:,:][::-1,::-1,::-1].conj()

    return M,fM

"""
This class is meant to wrap an objective function and deal with
reducing FFTs while allowing the optimizers to not need to know anything
about the real-space versus fourier space (or whatever) parameterizations.
"""
class ObjectiveWrapper:
    def __init__(self,xtype,obj = None,arg_dict = None,precond = None):
        self.arg_dict = arg_dict if arg_dict is not None else {}
        self.objective = obj
        self.xtype = xtype
        self.precond = precond

        assert xtype in ['real','complex','complex_coeff','complex_herm_coeff']

    def require_fspace(self): 
        return self.xtype in ['complex','complex_coeff','complex_herm_coeff']

    def set_objective(self,obj,arg_dict = None):
        self.args = arg_dict if arg_dict is not None else {}
        self.objective = obj

        if self.require_fspace():
            assert self.objective.fspace
        else:
            assert not self.objective.fspace

    def get_parameter(self):
        return self.x0

    def convert_parameter(self,x,comp_real=False,comp_fspace=False):
        is_x0 = x is self.x0
        if is_x0:
            M, fM = self.M0, self.fM0
        else:
            M, fM = param2density(x, self.xtype, self.M0.shape, \
                                  precond=self.precond)

            if comp_real and M is None:
                M = density.fspace_to_real(fM)

            if comp_fspace and fM is None:
                fM = density.real_to_fspace(M)
                
        return M, fM

    def set_density(self,M0,fM0):
        self.M0 = M0
        self.fM0 = fM0

        self.x0 = density2params(M0,fM0,self.xtype,precond=self.precond)

        return self.x0

    def eval_obj(self,x,**kwargs):
        M, fM = self.convert_parameter(x)

        cargs = copy(self.args)
        cargs.update(kwargs)
        if cargs.get('compute_gradient',True):
            logP,dlogP,outputs = self.objective.eval(M=M, fM=fM,
                                                     **cargs)
        else:
            logP,outputs = self.objective.eval(M=M, fM=fM,
                                               **cargs)
            return logP,outputs

        if self.xtype in ['complex_coeff','complex_herm_coeff'] :
            if cargs.get('all_grads',False):
                new_dlogPs = []
                for adlogP in outputs['all_dlogPs']:
                    new_dlogP = density2params(None,adlogP.reshape(fM.shape), \
                                               self.xtype,grad_transform=True, \
                                               precond=self.precond)
                    new_dlogPs.append(new_dlogP)
                outputs['all_dlogPs'] = new_dlogPs

            dlogP = density2params(None,dlogP.reshape(fM.shape),self.xtype, \
                                   grad_transform=True,precond=self.precond)

        return logP,dlogP.reshape(x.shape),outputs
    
class CryoOptimizer(BackgroundWorker):
    def outputbatchinfo(self,batch,res,logP,prefix,name):
        diag = {}
        stat = {}
        like = {}
        
        N_M = batch['N_M']
        cepoch = self.cryodata.get_epoch(frac=True)
        epoch = self.cryodata.get_epoch()
        num_data = self.cryodata.N_D_Train
        sigma = n.sqrt(n.mean(res['Evar_like']))
        sigma_prior = n.sqrt(n.mean(res['Evar_prior']))
        
        self.ostream('  {0} Batch:'.format(name))

        for suff in ['R','I','S']:
            diag[prefix+'_CV2_'+suff] = res['CV2_'+suff]

        diag[prefix+'_idxs'] = batch['img_idxs']
        diag[prefix+'_sigma2_est'] = res['sigma2_est']
        diag[prefix+'_correlation'] = res['correlation']
        diag[prefix+'_power'] = res['power']

#         self.ostream("    RMS Error: %g" % (sigma/n.sqrt(self.cryodata.noise_var)))
        self.ostream("    RMS Error: %g, Signal: %g" % (sigma/n.sqrt(self.cryodata.noise_var), \
                                                        sigma_prior/n.sqrt(self.cryodata.noise_var)))
        self.ostream("    Effective # of R / I / S:     %.2f / %.2f / %.2f " %\
                      (n.mean(res['CV2_R']), n.mean(res['CV2_I']),n.mean(res['CV2_S'])))

        # Importance Sampling Statistics
        is_speedups = []
        for suff in ['R','I','S','Total']:
            if self.cparams.get('is_on_'+suff,False) or (suff == 'Total' and len(is_speedups) > 0):
                spdup = N_M/res['N_' + suff + '_sampled_total']
                is_speedups.append((suff,spdup,n.mean(res['N_'+suff+'_sampled']),res['N_'+suff]))
                stat[prefix+'_is_speedup_'+suff] = [spdup]
            else:
                stat[prefix+'_is_speedup_'+suff] = [1.0]

        if len(is_speedups) > 0:
            lblstr = is_speedups[0][0]
            numstr = '%.2f (%d of %d)' % is_speedups[0][1:]
            for i in range(1,len(is_speedups)):
                lblstr += ' / ' + is_speedups[i][0]
                numstr += ' / %.2f (%d of %d)' % is_speedups[i][1:]
            
            self.ostream("    IS Speedup {0}: {1}".format(lblstr,numstr))

        stat[prefix+'_sigma'] = [sigma]
        stat[prefix+'_logp'] = [logP]
        stat[prefix+'_like'] = [res['L']]
        stat[prefix+'_num_data'] = [num_data]
        stat[prefix+'_num_data_evals'] = [self.num_data_evals]
        stat[prefix+'_iteration'] = [self.iteration]
        stat[prefix+'_epoch'] = [epoch]
        stat[prefix+'_cepoch'] = [cepoch],
        stat[prefix+'_time'] = [time.time()]

        for k,v in res['like_timing'].iteritems():
            stat[prefix+'_like_timing_'+k] = [v]
        
        Idxs = batch['img_idxs']
        self.img_likes[Idxs] = res['like']
        like['img_likes'] = self.img_likes
        like['train_idxs'] = self.cryodata.train_idxs
        like['test_idxs'] = self.cryodata.test_idxs
        keepidxs = self.cryodata.train_idxs if prefix == 'train' else self.cryodata.test_idxs
        keeplikes = self.img_likes[keepidxs]
        keeplikes = keeplikes[n.isfinite(keeplikes)]
        quants = n.percentile(keeplikes, range(0,101))
        stat[prefix+'_full_like_quantiles'] = [quants]
        quants = n.percentile(res['like'], range(0,101))
        stat[prefix+'_mini_like_quantiles'] = [quants]
        stat[prefix+'_num_like_quantiles'] = [len(keeplikes)]

        self.diagout.output(**diag)
        self.statout.output(**stat)
        self.likeout.output(**like)

    def ioworker(self):
        while True:
            iotype,fname,data = self.io_queue.get()
            
            try:
                if iotype == 'mrc':
                    writeMRC(fname,*data)
                elif iotype == 'pkl':
                    with open(fname, 'wb') as f:
                        cPickle.dump(data, f, protocol=-1)
                elif iotype == 'cp':
                    copyfile(fname,data)
            except:
                print "ERROR DUMPING {0}: {1}".format(fname, sys.exc_info()[0])
                
            self.io_queue.task_done()
        
    def __init__(self, expbase, cmdparams=None):
        """cryodata is a CryoData instance. 
        expbase is a path to the base of folder where this experiment's files
        will be stored.  The folder above expbase will also be searched
        for .params files. These will be loaded first."""
        BackgroundWorker.__init__(self)

        # Create a background thread which handles IO
        self.io_queue = Queue()
        self.io_thread = Thread(target=self.ioworker)
        self.io_thread.daemon = True
        self.io_thread.start()

        # General setup ----------------------------------------------------
        self.expbase = expbase
        self.outbase = None

        # Paramter setup ---------------------------------------------------
        # search above expbase for params files
        _,_,filenames = os.walk(opj(expbase,'../')).next()
        self.paramfiles = [opj(opj(expbase,'../'), fname) \
                           for fname in filenames if fname.endswith('.params')]
        # search expbase for params files
        _,_,filenames = os.walk(opj(expbase)).next()
        self.paramfiles += [opj(expbase,fname)  \
                            for fname in filenames if fname.endswith('.params')]
        if 'local.params' in filenames:
            self.paramfiles += [opj(expbase,'local.params')]
        # load parameter files
        self.params = Params(self.paramfiles)
        self.cparams = None
        
        if cmdparams is not None:
            # Set parameter specified on the command line
            for k,v in cmdparams.iteritems():
                self.params[k] = v
                
        # Dataset setup -------------------------------------------------------
        self.imgpath = self.params['inpath']
        psize = self.params['resolution']
        if not isinstance(self.imgpath,list):
            imgstk = MRCImageStack(self.imgpath,psize)
        else:
            imgstk = CombinedImageStack([MRCImageStack(cimgpath,psize) for cimgpath in self.imgpath])

        if self.params.get('float_images',True):
            imgstk.float_images()
        
        self.ctfpath = self.params['ctfpath']
        mscope_params = self.params['microscope_params']
         
        if not isinstance(self.ctfpath,list):
            ctfstk = CTFStack(self.ctfpath,mscope_params)
        else:
            ctfstk = CombinedCTFStack([CTFStack(cctfpath,mscope_params) for cctfpath in self.ctfpath])


        self.cryodata = CryoDataset(imgstk,ctfstk)
        self.cryodata.compute_noise_statistics()
        if self.params.get('window_images',True):
            imgstk.window_images()
        minibatch_size = self.params['minisize']
        testset_size = self.params['test_imgs']
        partition = self.params.get('partition',0)
        num_partitions = self.params.get('num_partitions',1)
        seed = self.params['random_seed']
        if isinstance(partition,str):
            partition = eval(partition)
        if isinstance(num_partitions,str):
            num_partitions = eval(num_partitions)
        if isinstance(seed,str):
            seed = eval(seed)
        self.cryodata.divide_dataset(minibatch_size,testset_size,partition,num_partitions,seed)
        
        self.cryodata.set_datasign(self.params.get('datasign','auto'))
        if self.params.get('normalize_data',True):
            self.cryodata.normalize_dataset()

        self.voxel_size = self.cryodata.pixel_size


        # Iterations setup -------------------------------------------------
        self.iteration = 0 
        self.tic_epoch = None
        self.num_data_evals = 0
        self.eval_params()

        outdir = self.cparams.get('outdir',None)
        if outdir is None:
            if self.cparams.get('num_partitions',1) > 1:
                outdir = 'partition{0}'.format(self.cparams['partition'])
            else:
                outdir = ''
        self.outbase = opj(self.expbase,outdir)
        if not os.path.isdir(self.outbase):
            os.makedirs(self.outbase) 

        # Output setup -----------------------------------------------------
        self.ostream = OutputStream(opj(self.outbase,'stdout'))

        self.ostream(80*"=")
        self.ostream("Experiment: " + expbase + \
                     "    Kernel: " + self.params['kernel'])
        self.ostream("Started on " + socket.gethostname() + \
                     "    At: " + time.strftime('%B %d %Y: %I:%M:%S %p'))
        self.ostream("Git SHA1: " + gitutil.git_get_SHA1())
        self.ostream(80*"=")
        gitutil.git_info_dump(opj(self.outbase, 'gitinfo'))
        self.startdatetime = datetime.now()


        # for diagnostics and parameters
        self.diagout = Output(opj(self.outbase, 'diag'),runningout=False)
        # for stats (per image etc)
        self.statout = Output(opj(self.outbase, 'stat'),runningout=True)
        # for likelihoods of individual images
        self.likeout = Output(opj(self.outbase, 'like'),runningout=False)

        self.img_likes = n.empty(self.cryodata.N_D)
        self.img_likes[:] = n.inf

        # optimization state vars ------------------------------------------
        init_model = self.cparams.get('init_model',None)
        if init_model is not None:
            filename = init_model
            if filename.upper().endswith('.MRC'):
                M = readMRC(filename)
            else:
                with open(filename) as fp:
                    M = cPickle.load(fp)
                    if type(M)==list:
                        M = M[-1]['M'] 
            if M.shape != 3*(self.cryodata.N,):
                M = cryoem.resize_ndarray(M,3*(self.cryodata.N,),axes=(0,1,2))
        else:
            init_seed = self.cparams.get('init_random_seed',0)  + self.cparams.get('partition',0)
            print "Randomly generating initial density (init_random_seed = {0})...".format(init_seed), ; sys.stdout.flush()
            tic = time.time()
            M = cryoem.generate_phantom_density(self.cryodata.N, 0.95*self.cryodata.N/2.0, \
                                                5*self.cryodata.N/128.0, 30, seed=init_seed)
            print "done in {0}s".format(time.time() - tic)

        tic = time.time()
        print "Windowing and aligning initial density...", ; sys.stdout.flush()
        # window the initial density
        wfunc = self.cparams.get('init_window','circle')
        cryoem.window(M,wfunc)

        # Center and orient the initial density
        cryoem.align_density(M)
        print "done in {0:.2f}s".format(time.time() - tic)

        # apply the symmetry operator
        init_sym = get_symmetryop(self.cparams.get('init_symmetry',self.cparams.get('symmetry',None)))
        if init_sym is not None:
            tic = time.time()
            print "Applying symmetry operator...", ; sys.stdout.flush()
            M = init_sym.apply(M)
            print "done in {0:.2f}s".format(time.time() - tic)

        tic = time.time()
        print "Scaling initial model...", ; sys.stdout.flush()
        modelscale = self.cparams.get('modelscale','auto')
        mleDC, _, mleDC_est_std = self.cryodata.get_dc_estimate()
        if modelscale == 'auto':
            # Err on the side of a weaker prior by using a larger value for modelscale
            modelscale = (n.abs(mleDC) + 2*mleDC_est_std)/self.cryodata.N
            print "estimated modelscale = {0:.3g}...".format(modelscale), ; sys.stdout.flush()
            self.params['modelscale'] = modelscale
            self.cparams['modelscale'] = modelscale
        M *= modelscale/M.sum()
        print "done in {0:.2f}s".format(time.time() - tic)
        if mleDC_est_std/n.abs(mleDC) > 0.05:
            print "  WARNING: the DC component estimate has a high relative variance, it may be inaccurate!"
        if ((modelscale*self.cryodata.N - n.abs(mleDC)) / mleDC_est_std) > 3:
            print "  WARNING: the selected modelscale value is more than 3 std devs different than the estimated one.  Be sure this is correct."

        self.M = n.require(M,dtype=density.real_t)
        self.fM = density.real_to_fspace(M)
        self.dM = density.zeros_like(self.M)

        self.step = eval(self.cparams['optim_algo'])
        self.step.setup(self.cparams, self.diagout, self.statout, self.ostream)

        # Objective function setup --------------------------------------------
        param_type = self.cparams.get('parameterization','real')
        cplx_param = param_type in ['complex','complex_coeff','complex_herm_coeff']
        self.like_func = eval_objective(self.cparams['likelihood'])
        self.prior_func = eval_objective(self.cparams['prior'])

        if self.cparams.get('penalty',None) is not None:
            self.penalty_func = eval_objective(self.cparams['penalty'])
            prior_func = SumObjectives(self.prior_func.fspace, \
                                       [self.penalty_func,self.prior_func], None)
        else:
            prior_func = self.prior_func

        self.obj = SumObjectives(cplx_param,
                                 [self.like_func,prior_func], [None,None])
        self.obj.setup(self.cparams, self.diagout, self.statout, self.ostream)
        self.obj.set_dataset(self.cryodata)
        self.obj_wrapper = ObjectiveWrapper(param_type)

        self.last_save = time.time()
        
        self.logpost_history = FiniteRunningSum()
        self.like_history = FiniteRunningSum()

        # Importance Samplers -------------------------------------------------
        self.is_sym = get_symmetryop(self.cparams.get('is_symmetry',self.cparams.get('symmetry',None)))
        self.sampler_R = FixedFisherImportanceSampler('_R',self.is_sym)
        self.sampler_I = FixedFisherImportanceSampler('_I')
        self.sampler_S = FixedGaussianImportanceSampler('_S')
        self.like_func.set_samplers(sampler_R=self.sampler_R,sampler_I=self.sampler_I,sampler_S=self.sampler_S)

    def eval_params(self):
        # cvars are state variables that can be used in parameter expressions
        cvars = {}
        cvars['cepoch'] = self.cryodata.get_epoch(frac=True)
        cvars['epoch'] = self.cryodata.get_epoch()
        cvars['iteration'] = self.iteration
        cvars['num_data'] = self.cryodata.N_D_Train
        cvars['num_batches'] = self.cryodata.N_batches
        cvars['noise_std'] = n.sqrt(self.cryodata.noise_var)
        cvars['data_std'] = n.sqrt(self.cryodata.data_var)
        cvars['voxel_size'] = self.voxel_size
        cvars['pixel_size'] = self.cryodata.pixel_size
        cvars['prev_max_frequency'] = self.cparams['max_frequency'] if self.cparams is not None else None

        # prelist fields are parameters that can be used in evaluating other parameter
        # expressions, they can only depend on values defined in cvars
        prelist = ['max_frequency']
        
        skipfields = set(['inpath','ctfpath'])

        cvars = self.params.partial_evaluate(prelist,**cvars)
        if self.cparams is None:
            self.max_frequency_changes = 0
        else:
            self.max_frequency_changes += cvars['max_frequency'] != cvars['prev_max_frequency']
                
        cvars['num_max_frequency_changes'] =  self.max_frequency_changes
        cvars['max_frequency_changed'] = cvars['max_frequency'] != cvars['prev_max_frequency']
        self.cparams = self.params.evaluate(skipfields,**cvars)

        self.cparams['exp_path'] = self.expbase
        self.cparams['out_path'] = self.outbase

        if 'name' not in self.cparams:
            self.cparams['name'] = '{0} - {1} - {2} ({3})'.format(self.cparams['dataset_name'], self.cparams['prior_name'], self.cparams['optimizer_name'], self.cparams['kernel'])

    def run(self):
        while self.dowork(): pass
        print "Waiting for IO queue to clear...",  ; sys.stdout.flush()
        self.io_queue.join()
        print "done."  ; sys.stdout.flush()

    def begin(self):
        BackgroundWorker.begin(self)

    def end(self):
        BackgroundWorker.end(self)

    def dowork(self):
        """Do one atom of work. I.E. Execute one minibatch"""

        timing = {}
        # Time each minibatch
        tic_mini = time.time()

        self.iteration += 1

        # Fetch the current batches
        trainbatch = self.cryodata.get_next_minibatch(self.cparams.get('shuffle_minibatches',True))

        # Get the current epoch
        cepoch = self.cryodata.get_epoch(frac=True)
        epoch = self.cryodata.get_epoch()
        num_data = self.cryodata.N_D_Train

        # Evaluate the parameters
        self.eval_params()
        timing['setup'] = time.time() - tic_mini

        # Do hyperparameter learning
        if self.cparams.get('learn_params',False):
            tic_learn = time.time()
            if self.cparams.get('learn_prior_params',True):
                tic_learn_prior = time.time()
                self.prior_func.learn_params(self.params, self.cparams, M=self.M, fM=self.fM)
                timing['learn_prior'] = time.time() - tic_learn_prior 

            if self.cparams.get('learn_likelihood_params',True):
                tic_learn_like = time.time()
                self.like_func.learn_params(self.params, self.cparams, M=self.M, fM=self.fM)
                timing['learn_like'] = time.time() - tic_learn_like
                
            if self.cparams.get('learn_prior_params',True) or self.cparams.get('learn_likelihood_params',True):
                timing['learn_total'] = time.time() - tic_learn   

        # Time each epoch
        if self.tic_epoch == None:
            self.ostream("Epoch: %d" % epoch)
            self.tic_epoch = (tic_mini,epoch)
        elif self.tic_epoch[1] != epoch:
            self.ostream("Epoch Total - %.6f seconds " % \
                         (tic_mini - self.tic_epoch[0]))
            self.tic_epoch = (tic_mini,epoch)

        sym = get_symmetryop(self.cparams.get('symmetry',None))
        if sym is not None:
            self.obj.ws[1] = 1.0/sym.get_order()

        tic_mstats = time.time()
        self.ostream(self.cparams['name']," Iteration:", self.iteration,\
                     " Epoch:", epoch, " Host:", socket.gethostname())

        # Compute density statistics
        N = self.cryodata.N
        M_sum = self.M.sum(dtype=n.float64)
        M_zeros = (self.M == 0).sum()
        M_mean = M_sum/N**3
        M_max = self.M.max()
        M_min = self.M.min()
#         self.ostream("  Density (min/max/avg/sum/zeros): " +
#                      "%.2e / %.2e / %.2e / %.2e / %g " %
#                      (M_min, M_max, M_mean, M_sum, M_zeros))
        self.statout.output(total_density=[M_sum],
                            avg_density=[M_mean],
                            nonzero_density=[M_zeros],
                            max_density=[M_max],
                            min_density=[M_min])
        timing['density_stats'] = time.time() - tic_mstats

        # evaluate test batch if requested
        if self.iteration <= 1 or self.cparams.get('evaluate_test_set',self.iteration%5):
            tic_test = time.time()
            testbatch = self.cryodata.get_testbatch()

            self.obj.set_data(self.cparams,testbatch)
            testLogP, res_test = self.obj.eval(M=self.M, fM=self.fM,
                                               compute_gradient=False)

            self.outputbatchinfo(testbatch, res_test, testLogP, 'test', 'Test')
            timing['test_batch'] = time.time() - tic_test
        else:
            testLogP, res_test = None, None

        # setup the wrapper for the objective function 
        tic_objsetup = time.time()
        self.obj.set_data(self.cparams,trainbatch)
        self.obj_wrapper.set_objective(self.obj)
        x0 = self.obj_wrapper.set_density(self.M,self.fM)
        evalobj = self.obj_wrapper.eval_obj
        timing['obj_setup'] = time.time() - tic_objsetup

        # Get step size
        self.num_data_evals += trainbatch['N_M']  # at least one gradient
        tic_objstep = time.time()
        trainLogP, dlogP, v, res_train, extra_num_data = self.step.do_step(x0,
                                                         self.cparams,
                                                         self.cryodata,
                                                         evalobj,
                                                         batch=trainbatch)

        # Apply the step
        x = x0 + v
        timing['step'] = time.time() - tic_objstep

        # Convert from parameters to value
        tic_stepfinalize = time.time()
        prevM = n.copy(self.M)
        self.M, self.fM = self.obj_wrapper.convert_parameter(x,comp_real=True)
 
        apply_sym = sym is not None and self.cparams.get('perfect_symmetry',True) and self.cparams.get('apply_symmetry',True)
        if apply_sym:
            self.M = sym.apply(self.M)

        # Truncate the density to bounds if they exist
        if self.cparams['density_lb'] is not None:
            n.maximum(self.M,self.cparams['density_lb']*self.cparams['modelscale'],out=self.M)
        if self.cparams['density_ub'] is not None:
            n.minimum(self.M,self.cparams['density_ub']*self.cparams['modelscale'],out=self.M)

        # Compute net change
        self.dM = prevM - self.M

        # Convert to fourier space (may not be required)
        if self.fM is None or apply_sym \
           or self.cparams['density_lb'] != None \
           or self.cparams['density_ub'] != None:
            self.fM = density.real_to_fspace(self.M)
        timing['step_finalize'] = time.time() - tic_stepfinalize

        # Compute step statistics
        tic_stepstats = time.time()
        step_size = n.linalg.norm(self.dM)
        grad_size = n.linalg.norm(dlogP)
        M_norm = n.linalg.norm(self.M)

        self.num_data_evals += extra_num_data
        inc_ratio = step_size / M_norm
        self.statout.output(step_size=[step_size],
                            inc_ratio=[inc_ratio],
                            grad_size=[grad_size],
                            norm_density=[M_norm])
        timing['step_stats'] = time.time() - tic_stepstats


        # Update import sampling distributions
        tic_isupdate = time.time()
        self.sampler_R.perform_update()
        self.sampler_I.perform_update()
        self.sampler_S.perform_update()

        self.diagout.output(global_phi_R=self.sampler_R.get_global_dist())
        self.diagout.output(global_phi_I=self.sampler_I.get_global_dist())
        self.diagout.output(global_phi_S=self.sampler_S.get_global_dist())
        timing['is_update'] = time.time() - tic_isupdate
        
        # Output basic diagnostics
        tic_diagnostics = time.time()
        self.diagout.output(iteration=self.iteration, epoch=epoch, cepoch=cepoch)

        if self.logpost_history.N_sum != self.cryodata.N_batches:
            self.logpost_history.setup(trainLogP,self.cryodata.N_batches)
        self.logpost_history.set_value(trainbatch['id'],trainLogP)

        if self.like_history.N_sum != self.cryodata.N_batches:
            self.like_history.setup(res_train['L'],self.cryodata.N_batches)
        self.like_history.set_value(trainbatch['id'],res_train['L'])

        self.outputbatchinfo(trainbatch, res_train, trainLogP, 'train', 'Train')

        # Dump parameters here to catch the defaults used in evaluation
        self.diagout.output(params=self.cparams,
                            envelope_mle=self.like_func.get_envelope_mle(),
                            sigma2_mle=self.like_func.get_sigma2_mle(),
                            hostname=socket.gethostname())
        self.statout.output(num_data=[num_data],
                            num_data_evals=[self.num_data_evals],
                            iteration=[self.iteration],
                            epoch=[epoch],
                            cepoch=[cepoch],
                            logp=[self.logpost_history.get_mean()],
                            like=[self.like_history.get_mean()],
                            sigma=[self.like_func.get_rmse()],
                            time=[time.time()])
        timing['diagnostics'] = time.time() - tic_diagnostics

        checkpoint_it = self.iteration % self.cparams.get('checkpoint_frequency',50) == 0 
        save_it = checkpoint_it or self.cparams['save_iteration'] or \
                  time.time() - self.last_save > self.cparams.get('save_time',n.inf)
                  
        if save_it:
            tic_save = time.time()
            self.last_save = tic_save
            if self.io_queue.qsize():
                print "Warning: IO queue has become backlogged with {0} remaining, waiting for it to clear".format(self.io_queue.qsize())
                self.io_queue.join()
            self.io_queue.put(( 'pkl', self.statout.fname, copy(self.statout.outdict) ))
            self.io_queue.put(( 'pkl', self.diagout.fname, deepcopy(self.diagout.outdict) ))
            self.io_queue.put(( 'pkl', self.likeout.fname, deepcopy(self.likeout.outdict) ))
            self.io_queue.put(( 'mrc', opj(self.outbase,'model.mrc'), \
                                (n.require(self.M,dtype=density.real_t),self.voxel_size) ))
            self.io_queue.put(( 'mrc', opj(self.outbase,'dmodel.mrc'), \
                                (n.require(self.dM,dtype=density.real_t),self.voxel_size) ))

            if checkpoint_it:
                self.io_queue.put(( 'cp', self.diagout.fname, self.diagout.fname+'-{0:06}'.format(self.iteration) ))
                self.io_queue.put(( 'cp', self.likeout.fname, self.likeout.fname+'-{0:06}'.format(self.iteration) ))
                self.io_queue.put(( 'cp', opj(self.outbase,'model.mrc'), opj(self.outbase,'model-{0:06}.mrc'.format(self.iteration)) ))
            timing['save'] = time.time() - tic_save
                
            
        time_total = time.time() - tic_mini
        self.ostream("  Minibatch Total - %.2f seconds                         Total Runtime - %s" %
                     (time_total, format_timedelta(datetime.now() - self.startdatetime) ))

        
        return self.iteration < self.cparams.get('max_iterations',n.inf) and \
               cepoch < self.cparams.get('max_epochs',n.inf)
       
