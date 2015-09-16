import numbers 

from finitesum import FiniteRunningSum
from logsumexp import logsumexp
from backgroundworker import BackgroundWorker
from params import Params
from output import Output,OutputStream
from gitutil import git_info_dump, git_get_SHA1

def format_timedelta(diff):
    if isinstance(diff, numbers.Real):
        s = diff
        d = 0
    else:
        s = diff.seconds
        d = diff.days
    h,r = divmod(s, 3600)
    m,s = divmod(r, 60)
    if d > 0:
        return '%02d:%02d:%02d:%02ds' % (diff.days, h,m,s)
    else:
        return '%02d:%02d:%02ds' % (h,m,s)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        return self[attr]
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__
    
def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self.f(*key)
            self[key] = ret
            return ret
    return memodict(f)

