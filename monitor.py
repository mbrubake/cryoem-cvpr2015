from util import *
import cPickle, os, shutil, subprocess, shlex, time, platform
from cryoio import mrc
import numpy as n

default_files = ['model.mrc','dmodel.mrc','diag','stat']

class ExpSetMonitor():
    def __init__(self,exps,base,rbase,lbase,ruserhost,*args,**kwargs):
        self.expbases = [ os.path.join(base,e) for e in exps ]
        self.monitors = [ ExpMonitor(expbase,rbase,lbase,ruserhost,*args,**kwargs) for expbase in self.expbases ]

    def fetch_update(self):
        anyup = [ m.fetch_update() for m in self.monitors ]
        return anyup

class ExpMonitor():
    def __init__(self, expbase, rbase = './', lbase = './', ruserhost = None, files = None):
        if files is None:
            self.files = set(default_files)
        else:
            self.files = set(files)

        self.expbase = expbase

        if ruserhost == None:
            self.lbase = os.path.join(lbase, expbase)
        else:
            self.lbase = os.path.join(lbase, ruserhost, expbase)

        self.rbase = os.path.join(rbase, expbase)
        self.ruserhost = ruserhost

        if not os.path.isdir(self.lbase):
            os.makedirs(self.lbase)

        self.load_data_files()

    def get_name(self):
        return self.diag['params']['name']
    
    def get_statistic(self,yval,xval='iteration',dset=None,smooth_window=1):
        assert 'stat' in self.files

        cstat = self.stat
        
        x = self.get_xval(xval,dset=dset)
        if dset is None:
            y = cstat[yval]
        else:
            y = cstat[dset+'_'+yval]

        if smooth_window > 1:
            y = moving_average(n.array(y),smooth_window)
            
        return x, y

    def get_xval(self, xval='num_data', dset='train'):
        if dset is None or dset+'_'+xval not in self.stat:
            x = n.array(self.stat[xval])
        else:
            x = n.array(self.stat[dset+'_'+xval])
#        if smooth_window < n.size(x):
#            lp = smooth(n.array(lp),smooth_window)
        if xval=='time':
            x -= x[0]
            x /= 60*60
        return x

    def get_throughput(self, xval='num_data', smooth_window=1):
        x = self.get_xval(xval)
        times = self.stat['time']
        x = x[1:]
        dt = n.diff(times)
        nd = self.stat['num_data_evals']
        dnd = n.diff(nd)
        tp = dnd/dt
        tp = moving_average(n.array(tp), smooth_window)
        return x,tp
    
    def get_like_timing(self, xval='num_data', dset='train'):
        x = self.get_xval(xval)
        ks = [k for k in self.stat.keys() if k.startswith(dset+'_like_timing_')]
        like_timing = {k[len(dset+'_like_timing_'):] : self.stat[k] for k in ks}
        return x, like_timing

    def supports_like_timing(self):
        return True if 'train_like_timing_total' in self.stat else False

    def get_isperf(self, xval='num_data', dset='train'):
        x = self.get_xval(xval)
        is_speedup_R = self.stat[dset+'_is_speedup_R']
        is_speedup_I = self.stat[dset+'_is_speedup_I']
        is_speedup_S = self.stat[dset+'_is_speedup_S']
        is_speedup   = self.stat[dset+'_is_speedup_Total']
        return x, is_speedup_R, is_speedup_I, is_speedup_S, is_speedup

    def get_likestats(self, xval='num_data', dset='train'):
        x = self.get_xval(xval, dset)
        fullquants = n.vstack(self.stat[dset+'_full_like_quantiles'])
        miniquants = n.vstack(self.stat[dset+'_mini_like_quantiles'])
        return x, fullquants, miniquants

    def get_dataset_size(self):
        return n.sum(n.array(self.stat['epoch']) == 0)*self.diag['params']['minisize']

    def load_data_files(self):
        try:
            self.M = self.loaddensity('model.mrc', 'M') if 'model.mrc' in self.files else None
            self.dM = self.loaddensity('dmodel.mrc', 'dM') if 'dmodel.mrc' in self.files else None
            self.diag = self.loadfile('diag') if 'diag' in self.files else None
            self.stat = self.loadfile('stat') if 'stat' in self.files else None

            if isinstance(self.diag,list):
                self.diag = self.diag[-1]
            if isinstance(self.stat,list):
                self.stat = self.stat[-1]
        except:
            pass

    def loaddensity(self, fname, key):
        if os.path.isfile(os.path.join(self.lbase,fname)):
            if fname.upper().endswith('.MRC'):
                return mrc.readMRC(os.path.join(self.lbase,fname))
            else:
                with open(os.path.join(self.lbase,fname), 'rb') as f:
                    return cPickle.load(f)[-1][key]
        else:
            return None

    def loadfile(self, fname):
        if os.path.isfile(os.path.join(self.lbase,fname)):
            with open(os.path.join(self.lbase,fname), 'rb') as f:
                return cPickle.load(f)
        else:
            return None

    def fetch_update(self, force=False):
        print "Checking for updates to {0}...".format(self.expbase)
        anyup = False
        for fname in self.files:
            update = checkremotefile(os.path.join(self.rbase,fname), \
                                     os.path.join(self.lbase,fname), \
                                     self.ruserhost)
            anyup = anyup or update
            if update or force:
                print "  Fetching update of {0}...".format(fname),
                getremotefile(os.path.join(self.rbase,fname), \
                              os.path.join(self.lbase,fname), \
                              self.ruserhost)
                print "done."

        if anyup or force:
            self.load_data_files()

        return anyup

def checkremotefile(remotepath, localpath, remoteuserhost):
    "Check if remote file has newer time than local file"
    if remoteuserhost != None:
        args = shlex.split("ssh "+ remoteuserhost + " 'stat -c %Y "+ remotepath + "' ")
    else:
        if platform.system() == 'Darwin':
            args = shlex.split('stat -f "%m" ' + remotepath)
        else:
            args = shlex.split("stat -c %Y " + remotepath)
    remotetime = time.localtime(float(subprocess.check_output(args).rstrip()))

    if not os.path.isfile(localpath):
        return True # update if local doesn't exist

    if platform.system() == 'Darwin':
        args = shlex.split('stat -f "%m" ' + localpath)
    else:
        args = shlex.split("stat -c %Y " + localpath)
    localtime = time.localtime(float(subprocess.check_output(args).rstrip()))
    
    return remotetime >= localtime

def getremotefile(remotepath, localpath, remoteuserhost):
    if remoteuserhost != None:
        args = shlex.split("scp "+ remoteuserhost+":"+remotepath + " " + localpath)
        subprocess.check_output(args)
    elif not os.path.samefile(remotepath,localpath):
        shutil.copy(remotepath,localpath)

def moving_average(a, winSz):
    if n == 1: return a
    winSz = min(winSz,a.size)
    ret = n.cumsum(a, dtype=float)
    ret[winSz:] = ret[winSz:] - ret[:-winSz]
    return n.r_[ret[0:(winSz-1)] / n.arange(1,winSz), ret[(winSz-1):] / winSz]

#def smooth(x, window_len=10, window='hanning', boundary='reflect'):
def smooth(x, window_len=10, window='flat', boundary='zeros'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    """

    x = n.asarray(x)

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if window_len < 3:
        return x

    if boundary != 'reflect' and x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    if not boundary in ['reflect','extend','zeros']:
        raise ValueError, "boundary is one of 'reflect', 'extend'"

    if boundary == 'reflect':
        s=n.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    elif boundary == 'extend':
        s=n.r_[x[0]*n.ones(window_len-1), x, x[-1]*n.ones(window_len-1)]
    elif boundary == 'zeros':
        s=n.r_[n.zeros(window_len-1), x, n.ones(window_len-1)]
    
    if window == 'flat': #moving average
        w = n.ones(window_len,'d')
    else:
        w = getattr(n, window)(window_len)
    y = n.convolve(w/w.sum(), s, mode='same')

    return y[window_len-1:-window_len+1]
