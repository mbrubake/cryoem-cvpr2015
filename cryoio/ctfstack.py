import numpy as n
from copy import copy

import ctf

def readctfstxt(parfile,num=None):
    labels = ('MAG', 'FILM', 'DF1', 'DF2', 'ANGAST', 'FLAG')

    data = []
    total = 0
    with open(parfile,'r') as f:
        for l in f:
            if num != None and total >= num:
                break
            fields = l.split(',')
            cnt = int(fields[0])
            cpars = [float(cf) for cf in fields[1:]]
            assert(len(cpars) == len(labels))
            if num != None and total+cnt > num:
                cnt = num - total
            data.extend(cnt*[ cpars ])
            total += cnt

    d = { lab : [ val[j] for val in data ] for j,lab in enumerate(labels) }
    d['ANGAST'] = [ n.pi*angast/180.0 for angast in d['ANGAST'] ]

    return d
    
def readdefocustxt(parfile, num=None):
    labels = ('FILM', 'DF1', 'DF2', 'ANGAST')

    data = []
    total = 0
    with open(parfile,'r') as f:
        for l in f:
            if l[0] != ' ':
                continue    # skip headers
            if num != None and total >= num:
                break
            fields = l.split()
            cpars = [float(cf) for cf in fields]
            assert(len(cpars) == len(labels))
            
            data += [ cpars ]
            total += 1

    d = { lab : [ val[j] for val in data ] for j,lab in enumerate(labels) }
    d['ANGAST'] = [ n.pi*angast/180.0 for angast in d['ANGAST'] ]

    return d

def readctfspar(parfile, num=None):
    "Reads a parfile. Reads the whole file, and returns a dict of lists of values."
    fmt = ('I7', 'F8', 'F8', 'F8', 'F8', 'F8', 'F8', 'I6', 'F9', 'F9', 'F8', 'F7')
    labels = ('IDX','PSI', 'THETA', 'PHI', 'SHX', 'SHY', 'MAG', 'FILM', 'DF1', 'DF2', 'ANGAST', 'CCMAX')

    ls = [int(s[1]) for s in fmt]
    splits = n.concatenate([[0,], n.cumsum(ls)])
    fs = {'I':int, 'F':float, 'S':str}

    data = []
    with open(parfile,'r') as f:
        while num is None or len(data) < num:
            l = f.readline()
            if l == '':
                break
            if l[0].lower() == 'c':
                continue
            data += [[fs[str.upper(fm[0])](l[s:e]) for fm,s,e in zip(fmt, splits[:-1], splits[1:]) ]]

    d = { lab : [ val[j] for val in data ] for j,lab in enumerate(labels) }
    d['ANGAST'] = [ n.pi*angast/180.0 for angast in d['ANGAST'] ]

    return d

class CTFStackBase:
    def __init__(self):
        pass

    def get_num_ctfs(self):
        return len(self.CTF)

    def get_num_images(self):
        return self.CTFmap.shape[0]

    def get_ctf(self,idx):
        return self.CTF[idx]
    
    def get_ctf_idx_for_image(self,idx):
        return self.CTFmap[idx]

    def get_ctf_for_image(self,idx):
        return self.CTF[self.CTFmap[idx]]
    
    def get_image_idxs_for_ctf(self,idx):
        return self.IMGmap[idx]
    
    def scale_ctfs(self,scale):
        for C in self.CTF:
            C.params['dscale'] *= scale
    
    def set_datasign(self,dsign):
        for C in self.CTF:
            C.params['dscale'] = dsign*n.abs(C.params['dscale'])

    def flip_datasign(self):
        for C in self.CTF:
            C.params['dscale'] *= -1

    def write_defocus_txt(self,fname):
        with open(fname,'wt') as f:
            f.write('Image#  Defocus1  Defocus2  Astig\n')
            for ctfI in self.CTFmap:
                cCTF = self.CTF[ctfI]
                df1 = cCTF.params['df1']
                df2 = cCTF.params['df2']
                angast = (180.0/n.pi)*cCTF.params['angast']

                f.write('{0: 5d}  {1: 8.1f}  {2: 8.1f}  {3: 6.2f}\n'.format(ctfI,df1,df2,angast))

    def write_pardata(self,fname):
        with open(fname,'wt') as f:
            f.write('{0: <10s}'.format('C'))
            for pf in self.parfields + ['FILM','DF1','DF2','ANGAST']:
                f.write(' {0: >10s}'.format(pf))
            f.write('\n')

            for i,ctfI in enumerate(self.CTFmap):
                cCTF = self.CTF[ctfI]
                f.write('{0: 10d}'.format(i))
                
                for pf in self.parfields:
                    f.write(' {0: 10.6g}'.format(self.pardata[pf][i]))

                df1 = cCTF.params['df1']
                df2 = cCTF.params['df2']
                angast = (180.0/n.pi)*cCTF.params['angast']

                f.write(' {0: 10d} {1: 10.1f} {2: 10.1f} {3: 10.2f}\n'.format(ctfI,df1,df2,angast))


class GeneratedCTFStack(CTFStackBase):
    def __init__(self,mscope_params,parfields=[]):
        CTFStackBase.__init__(self)

        assert 'wgh' in mscope_params
        assert 'cs' in mscope_params
        assert 'akv' in mscope_params
        
        if 'dscale' not in mscope_params:
            mscope_params['dscale'] = 1
            
        self.mscope_params = mscope_params

        # self.CTF contains a list of CTFs
        self.CTF = []
        # self.CTFmap contains a mapping which specifies the CTF of each image,
        # ie, image i has CTF self.CTF[self.CTFmap[i]]
        self.CTFmap = []
        # self.IMGmap contains a listing of which images are associated with 
        # each CTF, ie, self.CTF[j] is used by images self.IMGmap[j]
        self.IMGmap = []
        
        # self.pardata has extra, per-image information like ground truth poses
        self.pardata = dict([(i,[]) for i in parfields])
        self.parfields = parfields
        
    def add_ctf(self,ctf):
        self.CTF.append(ctf)
        self.IMGmap.append([])
        return len(self.CTF)-1
    
    def add_img(self,ctfI,**kwargs):
        assert ctfI < len(self.CTF)
        
        imgI = len(self.CTFmap)
        self.CTFmap.append(ctfI)
        self.IMGmap[ctfI].append(imgI)
        for i in self.parfields:
            self.pardata[i].append(kwargs[i])
        
        return imgI
    
class CTFStack(CTFStackBase):
    def __init__(self,parfile,mscope_params):
        CTFStackBase.__init__(self)

        assert 'wgh' in mscope_params
        assert 'cs' in mscope_params
        assert 'akv' in mscope_params
        
        if 'dscale' not in mscope_params:
            mscope_params['dscale'] = 1

        self.parfile = parfile
        if self.parfile[-3:].lower() == 'txt':
            with open(self.parfile, 'r') as f:
                numcols = max(len(f.readline().split()),len(f.readline().split(',')))
            if numcols == 7:
                self.pardata = readctfstxt(parfile)
            else:
                self.pardata = readdefocustxt(parfile)
        elif self.parfile[-3:].lower() == 'par':
                self.pardata = readctfspar(parfile)
        else:
            assert False
            
        CTF = []
        CTFmap = []
        IMGmap = []
        cDF1 = None; cDF2 = None; cANGAST = None; idx = 0;
        for DF1, DF2, ANGAST in zip(self.pardata['DF1'], self.pardata['DF2'], self.pardata['ANGAST']):
            # if the current params dont match, generate a new CTF
            if (cDF1 != DF1 or cDF2 != DF2 or cANGAST != ANGAST):
                params = copy(mscope_params)
                params['df1'] = DF1
                params['df2'] = DF2
                params['angast'] = ANGAST
                
                CTF.append(ctf.ParametricCTF(params))

                CTFmap.append(len(CTF)-1)
                IMGmap.append([ idx ])
                
                cDF1, cDF2, cANGAST = DF1, DF2, ANGAST
            else:
                CTFmap.append(CTFmap[-1])
                IMGmap[-1].append(idx)
            idx += 1

        # self.CTF contains a list of CTFs
        self.CTF = CTF
        # self.CTFmap contains a mapping which specifies the CTF of each image,
        # ie, image i has CTF self.CTF[self.CTFmap[i]]
        self.CTFmap = n.array(CTFmap)
        # self.IMGmap contains a listing of which images are associated with 
        # each CTF, ie, self.CTF[j] is used by images self.IMGmap[j]
        self.IMGmap = IMGmap


class CombinedCTFStack(CTFStackBase):
    def __init__(self,stacks):
        CTFStackBase.__init__(self)

        pardata = None
        CTF = []
        CTFmap = []
        IMGmap = []

        num_imgs = 0
        num_ctfs = 0
        for stk in stacks:
            if pardata is None:
                pardata = stk.pardata
            else:
                for k,v in stk.pardata.iteritems():
                    if k not in pardata:
                        pardata[k] = num_imgs*[None]
                    pardata[k] += v
                for k in pardata.iterkeys():
                    if k not in stk.pardata:
                        pardata[k] += stk.get_num_images()*[None]
                        
            CTF += stk.CTF
            CTFmap += [ c+num_ctfs for c in stk.CTFmap]
            for idxs in stk.IMGmap:
                IMGmap.append([ i+num_imgs for i in idxs ])
            num_ctfs += stk.get_num_ctfs()
            num_imgs += stk.get_num_images()
        
        self.pardata = pardata
        
        self.CTF = CTF
        self.CTFmap = n.array(CTFmap)
        self.IMGmap = IMGmap



        
        
    
