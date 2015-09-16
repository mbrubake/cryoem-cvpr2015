import sys, cPickle

class OutputStream():
    "An output stream, for redirecting print statements. Contains a file object and flushes after each write"
    def __init__(self, fname=None):
        self.fname = fname
        if fname==None:
            self.f = sys.stdout
        else:
            self.f = open(fname,'a')

    def __call__(self, *arg):
        for a in arg:
            print >>self.f, a,
        print >>self.f, ''
        self.f.flush()

    def __del__(self):
        if self.fname is not None:
            self.f.close()

class Output():
    def __init__(self, fname, runningout=True):
        self.fname = fname
        self.outdict = {}
        self.runningout = runningout

    def resetoutput(self):
        self.outdict = {}

    def output(self, **kwargs):
        "Adds the arguments to the current (last) output dict. If you output the same keyword twice, the outputs will get summed."
        for k,v in kwargs.iteritems():
            if k in self.outdict and self.runningout:
                self.outdict[k] += v
            else:
                self.outdict[k] = v

    def write(self, fname=None):
        "Writes the stored outputs to file. If idx is not None, only one index from the list is written."
        if fname is None:
            fname = self.fname

        with open(fname, 'wb') as f:
            cPickle.dump(self.outdict, f, protocol=2)

    def load(self):
        "Loads outputs from a file for resuming. Must set fname first."
        with open(self.fname) as f:
            self.outdict = cPickle.load(f)
