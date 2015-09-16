# visualizer.py
# Ali Punjani 2013
#
# Visualization code for cryoem

from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import colors
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter, LogFormatterExponent, LogFormatterMathtext, LogLocator

from monitor import ExpMonitor, ExpSetMonitor

from cryoio import ctf
import cryoem as c
import numpy as n
from cryoio import mrc
import quadrature
from objectives import eval_objective
import density
from symmetry import get_symmetryop

remotebase = './'
localbase = './'
remoteuserhost = None

class SlicePlot:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.autoscale(True)
        plt.subplots_adjust(left=0.25, bottom=0.25)

        self.M = None

    def __del__ (self):
        self.close()

    def __exit__ (self):
        self.close()

    def close(self):
        plt.close(self.fig)

    def set_data(self,M,vmin = None, vmax = None, bar = True):
        plt.figure(self.fig.number)

        N = M.shape[2]
        self.vmin = M.min() if vmin == None else vmin
        self.vmax = M.max() if vmax == None else vmax

        if self.M is None or self.M.shape[2] != N:
            # make the slider
            self.axframe = plt.axes([0.05, 0.01, 0.91, 0.03])
            self.sframe = Slider(self.axframe, 'Slice', 0, N-1, valinit=0,
                                 valfmt='%.1f')

            plt.subplot(111)
            plt.subplots_adjust(left=0.01,bottom=0.05,right=0.85,top=0.99)
            self.im = plt.imshow(M[:,:,0].T, origin='lower',
                                 interpolation='nearest',
                                 vmin=self.vmin, vmax=self.vmax)
            plt.axis('off')
            if bar:
                self.cbar_ax = plt.gcf().add_axes([0.85,0.15,0.05,0.8])
                plt.gcf().colorbar(self.im, cax=self.cbar_ax)

            self.sframe.on_changed(lambda val: self.update_display())

        self.M = M
        self.update_display()


    def update_display(self):
        if self.M is None: return

#        frame = n.around(self.sframe.val)
#        self.im.set_data(self.M[:,:,frame].T)

        val = self.sframe.val
        frame0 = n.floor(val)
        frame1 = frame0 + 1

        curr_slice = (frame1-val)*self.M[:,:,frame0].T
        if frame1 < self.M.shape[2]:
            curr_slice += (val-frame0)*self.M[:,:,frame1].T

        self.im.set_data(curr_slice)
 

class SetVisualizer(ExpSetMonitor):
    def __init__(self,exps,base,rbase,lbase,ruserhost,show_density = False):
        self.figstats = [ plt.figure() for i in range(5) ]
        if show_density:
            self.figdensity = [ mlab.figure(size=(800,600), bgcolor=(1,1,1), fgcolor=(0,0,0), figure=e) 
                                for e in exps ]
            ExpSetMonitor.__init__(self,exps,base,rbase,lbase,ruserhost,files=['model.mrc','diag','stat'])
        else:
            self.figdensity = []
            ExpSetMonitor.__init__(self,exps,base,rbase,lbase,ruserhost,files=['diag','stat'])


        self.contours = [ ]
        self.curr_contours = [ ]
        self.levels = [0.2,0.5,0.8]

        self.legend_names = None
        self.dataset_name = None
        self.font_size = None

    def __del__ (self):
        self.close()

    def __exit__ (self):
        self.close()

    def close(self):
        for fg in self.figstats: plt.close(fg)
        for ml in self.figdensity: mlab.close(ml)

    def get_legend_names(self):
        if self.legend_names != None:
            return self.legend_names
        else:
            return dict([ (mon.get_name(),mon.get_name()) for mon in self.monitors])
    
    def dowork(self):
        ups = self.fetch_update()
        if any(ups):
            self.update_plots()
            for i in range(len(self.figdensity)):
                if ups[i]: self.update_density(i)

    def updatevis(self):
        self.update_densities()
        self.update_plots()

    def update_density(self,ind):
        if ind >= len(self.figdensity):
            return
        ml = self.figdensity[ind]
        mon = self.monitors[ind]

        mlab.figure(ml)
        mlab.clf()
        if len(self.contours) > 0:
            self.curr_contours = self.contours
        self.curr_contours = plot_density(mon.M, self.curr_contours, self.levels)

    def update_densities(self):
        self.curr_contours = self.contours
        for i in range(len(self.figdensity)):
            self.update_density(i)

    def save_plots(self):
        if self.dataset_name != None: dsname = self.dataset_name + '_'
        else: dsname = ''

        plt.figure(self.figstats[0].number)
        plt.savefig(dsname+'NLP.pdf',bbox_inches='tight')
        plt.figure(self.figstats[1].number)
        plt.savefig(dsname+'StepSize.pdf',bbox_inches='tight')

    def update_plots(self):
        legend_names = self.get_legend_names()
        fsize = self.font_size

        params = self.monitors[0].diag[-1]['params']
        num_batches = params['num_batches']

        xval = 'iteration'

        plt.figure(self.figstats[0].number)
        plt.clf()
        init_lp = 0
        end_lp = 0
        max_it = 1e300
        for mon in self.monitors:
            (x_test,lp_test) = mon.get_logprob(xval = xval, dset = 'test')
            init_lp += lp_test[0]/len(self.monitors)
            end_lp += lp_test[-1]/len(self.monitors)
            plt.plot(x_test,lp_test,label=legend_names[mon.get_name()])
            max_it = min(max_it,x_test[-1])
        plt.ylabel('Test NLP', fontsize=fsize)
        plt.xlabel(xval, fontsize=fsize)
#        plt.ylim((end_lp - 0.05*n.abs(init_lp-end_lp),init_lp - 0.2*n.abs(init_lp - end_lp)))
        plt.xlim((None,max_it))
        plt.grid()
        plt.legend(fontsize=fsize)
        if self.dataset_name != None: plt.title(self.dataset_name, fontsize=fsize)

        plt.figure(self.figstats[1].number)
        plt.clf()
        init_lp = 0
        end_lp = 0
        max_it = 1e300
        for mon in self.monitors:
            (x_test,lp_test) = mon.get_logprob(xval = xval, dset = 'train', \
                                               smooth_window = num_batches)
            init_lp += lp_test[0]/len(self.monitors)
            end_lp += lp_test[-1]/len(self.monitors)
            plt.plot(x_test,lp_test,label=legend_names[mon.get_name()])
            max_it = min(max_it,x_test[-1])
        plt.ylabel('Train NLP', fontsize=fsize)
        plt.xlabel(xval, fontsize=fsize)
        plt.xlim((None,max_it))
        plt.grid()
        plt.legend(fontsize=fsize)
        if self.dataset_name != None: plt.title(self.dataset_name, fontsize=fsize)

        plt.figure(self.figstats[2].number)
        plt.clf()
        init_lp = 0
        end_lp = 0
        max_it = 1e300
        for mon in self.monitors:
            (x_test,lp_test) = mon.get_stepsize(xval = xval)
            init_lp += lp_test[0]/len(self.monitors)
            end_lp += lp_test[-1]/len(self.monitors)
            plt.plot(x_test,lp_test,label=legend_names[mon.get_name()])
            max_it = min(max_it,x_test[-1])
        plt.ylabel('Step Size', fontsize=fsize)
        plt.xlabel(xval, fontsize=fsize)
        plt.xlim((None,max_it))
        plt.grid()
        plt.legend(fontsize=fsize)
        if self.dataset_name != None: plt.title(self.dataset_name, fontsize=fsize)

        plt.figure(self.figstats[3].number)
        plt.clf()
        init_lp = 0
        end_lp = 0
        max_it = 1e300
        for mon in self.monitors:
            (x_test,lp_test) = mon.get_sigma(xval = xval, dset = 'test')
            init_lp += lp_test[0]/len(self.monitors)
            end_lp += lp_test[-1]/len(self.monitors)
            plt.plot(x_test,lp_test,label=legend_names[mon.get_name()])
            max_it = min(max_it,x_test[-1])
        plt.ylabel('Test Error', fontsize=fsize)
        plt.xlabel(xval, fontsize=fsize)
#        plt.ylim((end_lp - 0.05*n.abs(init_lp-end_lp),init_lp - 0.2*n.abs(init_lp - end_lp)))
        plt.xlim((None,max_it))
        plt.grid()
        plt.legend(fontsize=fsize)
        if self.dataset_name != None: plt.title(self.dataset_name, fontsize=fsize)


        plt.figure(self.figstats[4].number)
        plt.clf()
        init_lp = 0
        end_lp = 0
        max_it = 1e300
        for mon in self.monitors:
            (x_test,lp_test) = mon.get_sigma(xval = xval, dset = 'train', \
                                                   smooth_window = num_batches)
            init_lp += lp_test[0]/len(self.monitors)
            end_lp += lp_test[-1]/len(self.monitors)
            plt.plot(x_test,lp_test,label=legend_names[mon.get_name()])
            max_it = min(max_it,x_test[-1])
        plt.ylabel('Train Error', fontsize=fsize)
        plt.xlabel(xval, fontsize=fsize)
#        plt.ylim((end_lp - 0.05*n.abs(init_lp-end_lp),init_lp - 0.2*n.abs(init_lp - end_lp)))
        plt.xlim((None,max_it))
        plt.grid()
        plt.legend(fontsize=fsize)
        if self.dataset_name != None: plt.title(self.dataset_name, fontsize=fsize)


class Visualizer(ExpMonitor):
    def __init__(self, expbase, rbase=remotebase, lbase=localbase, ruserhost=remoteuserhost,extra_plots=False,show_grad=False):
        if show_grad:
            ExpMonitor.__init__(self, expbase,rbase,lbase,ruserhost)
        else:
            ExpMonitor.__init__(self, expbase,rbase,lbase,ruserhost,files=['model.mrc','diag','stat'])

        self.extra_plots = extra_plots

        self.show_grad = show_grad

        self.fig1 = mlab.figure(size=(800,600), bgcolor=(1,1,1), fgcolor=(0,0,0))
        self.figMslices = SlicePlot()
        if self.show_grad:
            self.figdMslices = SlicePlot()
            
        self.figures = {}

        self.contours = []
        self.curr_contours = []

        self.stats_xval = 'iteration'
#        self.updatevis()
    
    def get_figure(self,figid):
        if figid not in self.figures:
            self.figures[figid] = plt.figure()
        else:
            self.figures[figid].show()
        return self.figures[figid]

    def close_figure(self,figid):
        if figid in self.figures:
            plt.close(figid)
            del self.figures['figid']

    def close(self):
        for f in self.figures.itervalues():
            plt.close(f)
        self.figures = {}
        
        self.figMslices.close()
        mlab.close(self.fig1)
        if self.show_grad:
            self.figdMslices.close()

    def dowork(self):
        if self.fetch_update():
            self.updatevis()

    def save_mrc(self, aligned=True, fname=None):
        if fname == None:
            params = self.diag['params']
            name = params['name']
            fname = '{0}.mrc'.format(name)

        mrc.writeMRC(fname,self.alignedM if aligned else self.M, psz=params['resolution'])

    def show_envelope_plot(self,cfig):
        cdiag = self.diag
        cparams = cdiag['params']

        resolution = cparams['pixel_size']
        name = cparams['name']
        maxfreq = cparams['max_frequency']
        N = self.M.shape[0]
        rad_cutoff = cparams.get('rad_cutoff', 1.0)
        rad = min(rad_cutoff,maxfreq*2.0*resolution)
    
        envelope_mle = cdiag['envelope_mle']
        vmin = envelope_mle.min()
        vmax = envelope_mle.max()
        exp_envelope = cparams.get('exp_envelope',None)
        
        have_exp = exp_envelope is not None
        if have_exp:
            vmin = exp_envelope.min()
            vmax = exp_envelope.max()
        
        startI = int((1-rad)*N/2)+1
        endI = N/2 + int(rad*N/2)+1
        imextent = [startI-(N+1.0)/2,endI-(N+1.0)/2,startI-(N+1.0)/2,endI-(N+1.0)/2]
        imextent = [e/(2.0*resolution)/(N/2) for e in imextent]

        cbaxs = []

        plt.figure(cfig.number)
        plt.clf()

        cbaxs.append(plt.subplot(2,1+have_exp,1))
        im = plt.imshow(envelope_mle[startI:endI,startI:endI], interpolation='nearest',
                        vmin=vmin, vmax=vmax, extent=imextent)
        plt.title('ML')
        
        if have_exp:
            cbaxs.append(plt.subplot(2,2,2))
            plt.imshow(exp_envelope[startI:endI,startI:endI], interpolation='nearest',
                       vmin=vmin, vmax=vmax, extent=imextent)
            plt.title('MAP')
        plt.colorbar(im,ax=cbaxs)
        
        env_max = 1.0
        env_min = 0.0
        bfactor = cparams.get('learn_like_envelope_bfactor',500.0)
        have_bfactor = bfactor is not None 
        if have_bfactor:
            plt.subplot(2,1,2)
            (fs,bfactor_env) = get_env_func(N, resolution=resolution, 
                                            bfactor=bfactor)
    
            plt.plot(fs,bfactor_env,label='Prior (bfactor {0})'.format(bfactor),linewidth=2)
            env_max = max(env_max,bfactor_env.max())
            env_min = min(env_min,bfactor_env.min())
        else:
            fs = n.linspace(0,1.0/(2.0*resolution),N/2)
            bfactor_env = 1.0

        ra_mle_envelope = c.rotational_average(envelope_mle,maxRadius=N/2)
        plt.plot(fs[0:int(rad*N/2)],ra_mle_envelope[0:int(rad*N/2)],label='ML')
        env_max = max(env_max,ra_mle_envelope.max())
        env_min = min(env_min,ra_mle_envelope.min())
        if have_exp:
            ra_exp_envelope = c.rotational_average(exp_envelope,maxRadius=N/2) 
            plt.plot(fs[0:N/2],ra_exp_envelope[0:N/2],label='MAP',linewidth=2)
            env_max = max(env_max,ra_exp_envelope.max())
            env_min = min(env_min,ra_exp_envelope.min())
        plt.legend()
        plt.grid()

        plt.plot((rad/(2.0*resolution))*n.ones((2,)), n.array([env_min,env_max]))
            
        plt.suptitle(name + ' Envelope')
        
    def show_error_plot(self,cfig):
        cdiag = self.diag
        cparams = cdiag['params']

        name = cparams['name']

#         (x_train,sigma_train) = self.get_statistic(yval = 'sigma', xval = self.stats_xval, dset = 'train')
        (x_total,sigma_total) = self.get_statistic(yval = 'sigma', xval = self.stats_xval)
        (x_test,sigma_test) = self.get_statistic(yval = 'sigma', xval = self.stats_xval, dset = 'test')

        plt.figure(cfig.number)
        plt.clf()

#         plt.plot(x_train,sigma_train,label='Train',linewidth=2,linestyle='--')
        plt.plot(x_total,sigma_total,label='Total',linewidth=4)
        plt.plot(x_test,sigma_test,label='Test',linewidth=2,marker='o')
#         plt.yscale('log',basey=2)
        plt.legend()
        plt.grid()
        plt.title(name + ' Error')

    def show_noise_plot(self,cfig):
        cdiag = self.diag
        cparams = cdiag['params']

        vox_size = cparams['voxel_size']
        name = cparams['name']
        maxfreq = cparams['max_frequency']
        N = self.M.shape[0]
        rad_cutoff = cparams.get('rad_cutoff', 1.0)
        rad = min(rad_cutoff,maxfreq*2.0*vox_size)

        startI = int((1-rad)*N/2)+1
        endI = N/2 + int(rad*N/2)+1
        imextent = [startI-(N+1.0)/2,endI-(N+1.0)/2,startI-(N+1.0)/2,endI-(N+1.0)/2]
        imextent = [e/(2.0*vox_size)/(N/2) for e in imextent]
        sigma_est = cparams['sigma']
        sigma_mle = n.sqrt(cdiag['sigma2_mle'])
        train_sigma_est = n.sqrt(cdiag['train_sigma2_est']).reshape((N,N))
        test_sigma_est = n.sqrt(cdiag['test_sigma2_est']).reshape((N,N))
        showsigma = isinstance(sigma_est,n.ndarray)
        vmin = min([n.min(sigma_est),sigma_mle[startI:endI,startI:endI].min()])
        vmax = max([n.max(sigma_est),sigma_mle[startI:endI,startI:endI].max()])
        
        imshow_kws = { 'interpolation':'nearest', \
                       'vmin':vmin, 'vmax':vmax, 'extent':imextent, \
                       'norm':LogNorm(vmin=vmin, vmax=vmax) }

        cbaxs = []
        plt.figure(cfig.number)
        plt.clf()

        plt.subplot(2,1,1)
        raps = n.sqrt(c.rotational_average(train_sigma_est**2))
        fs = n.linspace(0,(len(raps)-1)/(N/2.0)/(2.0*vox_size),len(raps))
        plt.plot(fs,raps,label='Training RMSE')

        raps = n.sqrt(c.rotational_average(test_sigma_est**2))
        fs = n.linspace(0,(len(raps)-1)/(N/2.0)/(2.0*vox_size),len(raps))
        plt.plot(fs,raps,label='Testing RMSE')
        plt.legend()
        plt.grid()

        if showsigma:
            cbaxs.append(plt.subplot(2,2,3))
        else:
            cbaxs.append(plt.subplot(2,1,2))
        im = plt.imshow(sigma_mle[startI:endI,startI:endI], **imshow_kws)
        plt.title('Freq RMSE (MLE)')

        if showsigma:
            sigma_est = sigma_est.reshape((N,N))
            cbaxs.append(plt.subplot(2,2,4))
            im = plt.imshow(sigma_est[startI:endI,startI:endI], **imshow_kws)
            plt.title('Coloured Noise Std Dev')

        plt.subplot(2,1,1)
        if showsigma:
            raps = n.sqrt(c.rotational_average(sigma_est**2))
            fs = n.linspace(0,(len(raps)-1)/(N/2.0)/(2.0*vox_size),len(raps))
        else:
            raps = [sigma_est,sigma_est]
            fs = [fs[0],fs[-1]]
        plt.plot(fs,raps,label='Noise Std Dev')
        plt.xlim((0,rad/(2.0*vox_size)))
        plt.yscale('log',basey=2)
        plt.legend()
 
        plt.title(name + ' Noise Levels')

        plt.colorbar(im, ax=cbaxs, ticks=LogLocator(base=2), format=LogFormatterMathtext(base=2))
        
    def show_objective_plot(self,cfig):
        cdiag = self.diag
        cstat = self.stat
        cparams = cdiag['params']
        name = cparams['name']
        num_batches = cparams['num_batches']


        plt.figure(cfig.number)
        plt.clf()
        ax1 = plt.subplot(1,1,1)
        lines = []

        (x_test,lp_test) = self.get_statistic(yval='logp', xval = self.stats_xval, dset = 'test')
        lines += ax1.plot(x_test,lp_test,label='Test LP', linewidth=2, marker='o', color='g')
        
        if 'logp' in cstat:
#             (x_train,lp_train) = self.get_statistic(yval='logp', xval = self.stats_xval, dset = 'train')
#             lines += ax1.plot(x_train,lp_train,label='Train LP',linewidth=2, linestyle='--', color='b')
            (x_train,lp_train) = self.get_statistic(yval='logp', xval = self.stats_xval)
        else:
            (x_train,lp_train) = self.get_statistic(yval='logp', xval = self.stats_xval, dset = 'train', smooth_window=num_batches)
        lines += ax1.plot(x_train,lp_train,label='Total LP',linewidth=4, linestyle='-', color='b')
        ax1.grid()

        ax2 = ax1.twinx()

        (x_stepsize,stepsize) = self.get_statistic(yval = 'step_size', xval = self.stats_xval)
        lines += ax2.plot(x_stepsize,stepsize,label='Step Size', color='r', linestyle='-', linewidth=1)
#         ax2.set_yscale('log',basey=2)

        labels = [l.get_label() for l in lines]
        plt.legend(lines,labels)
        
        plt.title(name)

    def show_density_plot(self,cfig):
        cdiag = self.diag
        cparams = cdiag['params']
        name = cparams['name']
        maxfreq = cparams['max_frequency']
        resolution = cparams['voxel_size']
        prior = eval_objective(cparams['prior'])
        prior.set_params(cparams)

        N = self.M.shape[0]
        rad_cutoff = cparams.get('rad_cutoff', 1.0)
        rad = min(rad_cutoff,maxfreq*2.0*resolution)

        # Statistics of M
        plt.figure(cfig.number)
        plt.clf()
        plt.suptitle(name + ' Density Statistics')

        nHistBins = 0.5*self.M.shape[0]
        logprobScale = n.log(self.M.size/nHistBins)
        plt.subplot(2,1,1)
        plt.hist(self.M.reshape((-1,)),bins=nHistBins,log=True)
        histxLims = plt.xlim()
        histyLims = plt.ylim()
        vals = n.linspace(histxLims[0],histxLims[1],1000)
        plt.plot(vals,n.exp(logprobScale-prior.scalar_eval(vals)))
        plt.xlim(histxLims)
        plt.ylim(histyLims)
        plt.title('Voxel Histogram + Prior')

        plt.subplot(2,2,3)
        plt.hist(n.absolute(self.fM).reshape((-1,)),bins=nHistBins,log=True)
        plt.title('Power Histogram')
        (fs,raps) = rot_power_spectra(self.fM,resolution=resolution)
        plt.subplot(2,2,4)
        plt.plot(fs/(N/2.0)/(2.0*resolution),raps,label='RAPS')
        plt.plot((rad/(2.0*resolution))*n.ones((2,)), 
                  n.array([raps[raps>0].min(),raps.max()]))
        plt.yscale('log')
        plt.title('Rotationally Averaged Power Spectra')

    def updatevis(self, levels=[0.2,0.5,0.8]):
        if self.M is None or self.diag is None or self.stat is None:
            return

        cdiag = self.diag
        cparams = cdiag['params']
        sym = get_symmetryop(cparams.get('symmetry',None))
        quad_sym = sym if cparams.get('perfect_symmetry',True) else None

        resolution = cparams['voxel_size']

        name = cparams['name']
        maxfreq = cparams['max_frequency']
        N = self.M.shape[0]
        rad_cutoff = cparams.get('rad_cutoff', 1.0)
        rad = min(rad_cutoff,maxfreq*2.0*resolution)

        # Show objective function
        self.show_objective_plot(self.get_figure('stats'))
        
        # Show information about noise and error
        self.show_error_plot(self.get_figure('error'))
        self.show_noise_plot(self.get_figure('noise'))
        
        # Plot the envelope function if we have the info
        if 'envelope_mle' in cdiag:
            self.show_envelope_plot(self.get_figure('envelope'))
        else:
            self.close_figure('envelope')

        if sym is None:
            assert quad_sym is None
            alignedM,R = c.align_density(self.M)
            if self.show_grad:
                aligneddM = c.rotate_density(self.dM,R)
            else:
                aligneddM = None
        else:
            alignedM, aligneddM = self.M, self.dM
            R = n.identity(3)

        self.alignedM,self.aligneddM,self.alignedR = alignedM,aligneddM,R
        self.fM = density.real_to_fspace(self.M)

        self.figMslices.set_data(alignedM)

        glbl_phi_R = n.array([cdiag['global_phi_R']]).ravel()
        if len(glbl_phi_R) == 1:
            glbl_phi_R = None
        glbl_phi_I = cdiag['global_phi_I']
        glbl_phi_S = cdiag['global_phi_S']

        # Get direction quadrature
        quad_R = quadrature.quad_schemes[('dir',cparams.get('quad_type_R','sk97'))]
        quad_degree_R = cparams.get('quad_degree_R','auto')
        if quad_degree_R == 'auto':
            usFactor_R = cparams.get('quad_undersample_R',
                                     cparams.get('quad_undersample',1.0))
            quad_degree_R,_ = quad_R.compute_degree(N,rad,usFactor_R)
        origlebDirs,_ = quad_R.get_quad_points(quad_degree_R,quad_sym)
        lebDirs = n.dot(origlebDirs,R)

        # Get shift quadrature
        quad_S = quadrature.quad_schemes[('shift',cparams.get('quad_type_S','hermite'))]
        quad_degree_S = cparams.get('quad_degree_S','auto')
        if quad_degree_S == 'auto':
            usFactor_S = cparams.get('quad_undersample_S',
                                     cparams.get('quad_undersample',1.0))
            quad_degree_S = quad_S.get_degree(N,rad,
                                              cparams['quad_shiftsigma']/resolution,
                                              cparams['quad_shiftextent']/resolution,
                                              usFactor_S)
        pts_S,_ = quad_S.get_quad_points(quad_degree_S,
                                         cparams['quad_shiftsigma']/resolution,
                                         cparams['quad_shiftextent']/resolution,
                                         cparams.get('quad_shifttrunc','circ'))
        vmax_R = 5.0/len(glbl_phi_R)
        vmax_S = 5.0/len(glbl_phi_S)

        # Density visualization
        mlab.figure(self.fig1)
        mlab.clf()
        self.curr_contours = plot_density(alignedM, self.contours, levels)
#         dispPhiR = glbl_phi_R
#         dispDirs = lebDirs
#         plot_directions(alignedM.shape[0]*dispDirs + alignedM.shape[0]/2.0,
#                         dispPhiR,
#                         0, vmax_R)
        mlab.view(focalpoint=[alignedM.shape[0]/2.0,alignedM.shape[0]/2.0,alignedM.shape[0]/2.0],distance=1.5*alignedM.shape[0])


        if glbl_phi_R is not None:
            plt.figure(self.get_figure('global_is_dists').number)
            plt.clf()
            plot_importance_dists(name,lebDirs,pts_S*resolution,glbl_phi_R,glbl_phi_I,glbl_phi_S,vmax_R,vmax_S)

        if self.show_grad:
            # Statistics of dM
            self.figdMslices.set_data(aligneddM)

            plt.figure(self.get_figure('step_stats').number)
            plt.clf()
            plt.suptitle(name + ' Step Statistics')

            plt.subplot(1,2,1)
            plt.hist(self.dM.reshape((-1,)),bins=0.5*self.dM.shape[0],log=True)
            plt.title('Voxel Histogram')

            (fs,raps) = rot_power_spectra(self.dM,resolution=resolution)
            plt.subplot(1,2,2)
            plt.plot(fs/(N/2.0)/(2.0*resolution),raps,label='RAPS')
            plt.plot((rad/(2.0*resolution))*n.ones((2,)), 
                     n.array([raps[raps > 0].min(),raps.max()]))
            plt.yscale('log')
            plt.title('RAPS Step')


        if not self.extra_plots:
            self.close_figure('density_stats')
            return

        # Statistics of M
        self.show_density_plot(self.get_figure('density_stats'))


    def __del__ (self):
        self.close()

    def __exit__ (self):
        self.close()

### ------------------------- UTILITY FUNCTIONS FOR PLOTTING ------------------------
def plot_importance_dists(name,quadDirs,pts,phi_R,phi_I,phi_S,vmax_R,vmax_S):
    if isinstance(phi_I,n.ndarray):
        dispI = True
    else:
        dispI = False

    if isinstance(phi_S,n.ndarray) and pts.shape[0] > 2:
        dispS = True
    else:
        dispS = False

    if dispI or dispS:
        plt.subplot(2,1,1)
    else:
        plt.subplot(1,1,1)
    plotwinkeltriple(quadDirs,phi_R,vmin=0,vmax=vmax_R)
    plt.title(name + ' Global Direction Distribution')

    if dispI:
        plt.subplot(2,2,3,polar=True)
        plt.plot(n.linspace(0, 2.0*n.pi, phi_I.size+1, endpoint=True),
                 n.hstack([phi_I,phi_I[0]]))
        plt.gca().set_rmax(1.1*phi_I.max())
        plt.title('Inplane Distribution')

    if dispS:
        if dispI:
            plt.subplot(2,2,4)
        else:
            plt.subplot(2,1,2)
        plotshifts(pts[:,0],pts[:,1],phi_S,vmin=0,vmax=vmax_S)
        plt.title('Shift Distribution')

def plot_density(s, contours, levels=[0.2,0.5,0.8], colors= [(0,1,0),(0,0,1),(1,0,0)], opacity=[0.1,0.5,0.1]):
    "Makes a nice plot of a density in the current mlab.figure"
    src = mlab.pipeline.scalar_field(s)
    mlab.gcf().scene.background = (1,1,1)
    mlab.gcf().scene.foreground = (0,0,0)
    import itertools

    if contours == []:
        mins = s.min()
        ptps = s.ptp()
        curr_contours = [mins+l*ptps for l in levels]
    else:
        curr_contours = [c for c in contours]

    for cont,c,o in zip(curr_contours, itertools.cycle(colors), itertools.cycle(opacity)):
        mlab.pipeline.iso_surface(src, contours=[cont,], opacity=o, color=c)

#    mlab.text(0.1,0.9,'min: %15.2e' % (s.min()), color=(0,0,0), width=0.2)
#    mlab.text(0.1,0.85,'max: %15.2e' % (s.max()), color=(0,0,0), width=0.2)
    print s.min(), s.max()
    return curr_contours

def plot_directions(dirs,vals,vmin=None,vmax=None):
    if vmin != None or vmax != None:
        vals = n.clip(vals,vmin,vmax)
    mlab.points3d(dirs[:,0],dirs[:,1],dirs[:,2],n.log(1e-10+vals),scale_mode='none',scale_factor=5.0,opacity=0.2)
#    pts = mlab.pipeline.scalar_scatter(dirs[:,0],dirs[:,1],dirs[:,2],vals)
#    mesh = mlab.pipeline.delaunay3d(pts)
#    surf = mlab.pipeline.surface(mesh, opacity=0.1)


def get_env_func(N,resolution,bfactor = None):
    freq_radius = n.linspace(0,N/2,N/2+1)/(N*resolution)
    env = ctf.envelope_function(freq_radius,bfactor)

    return freq_radius, env

def rot_power_spectra(fM,powerLen = None,resolution = None):
    if resolution == None:
        resolution = 1

    powerfM = fM.real**2 + fM.imag**2 
    raps = c.rotational_average(powerfM,powerLen)
    radius = n.linspace(0,len(raps)-1,len(raps))

    return (radius,raps)

def winkeltriple(t,ph):
    ph1 = n.arccos(2.0/n.pi)
    a = n.arccos(n.cos(ph)*n.cos(t/2.0))
    x = 0.5*( t*n.cos(ph1) + 2.0*n.cos(ph)*n.sin(t/2.0) / n.sinc(a/n.pi) )
    y = 0.5*( ph + n.sin(ph)/n.sinc(a/n.pi) )
    return x,y
    
def plotshifts(x,y,v, vmin=None, vmax=None):
    plt.tripcolor(x,y,1e-10 + v,shading='gourad',vmin=vmin+1e-10,vmax=vmax+1e-10,norm=colors.LogNorm())
    plt.axis('equal')
    plt.colorbar()

def plotwinkeltriple(d,v, vmin=None, vmax=None):
    """ Plots a winkel projection of a function on a sphere evaluated at directions d
    v - values
    """
    
#     phi = n.arctan2(d[:,2],n.linalg.norm(d[:,0:2],axis=1)).reshape((-1,))
    phi = n.arctan2(d[:,2],n.linalg.norm(d[:,0:2],axis=1)).reshape((-1,))
    theta = n.arctan2(d[:,1],d[:,0]).reshape((-1,))

    x,y = winkeltriple(theta,phi)
    
    t_border = n.concatenate( [ n.linspace(n.pi,-n.pi,50), n.ones(50)*-n.pi, n.linspace(-n.pi,n.pi,50), n.ones(50)*n.pi ] )
    ph_border = n.concatenate( [ n.ones(50)*-n.pi/2.0, n.linspace(-n.pi/2,n.pi/2.0,50), n.ones(50)*n.pi/2.0, n.linspace(n.pi/2.0,-n.pi/2.0,50) ] )
    x_border,y_border = winkeltriple(t_border,ph_border)

    plt.hold(True)
    plt.tripcolor(x,y,1e-10 + v,shading='gourad',vmin=vmin+1e-10,vmax=vmax+1e-10,norm=colors.LogNorm())
    plt.plot(x_border,y_border,'-k')
    plt.colorbar()
    plt.show()

