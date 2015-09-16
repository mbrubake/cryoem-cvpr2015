import mrc, density, geom
from exceptions import NotImplementedError
from cryoem import resize_ndarray

from threading import Lock

import numpy as n

class ImageStackBase:
    def __init__(self):
        pass

    def get_pixel_size(self):
        return self.pixel_size
    
    def get_num_images(self):
        return self.num_images
    
    def get_num_pixels(self):
        return self.num_pixels
    
    def get_image(self,idxs=None):
        raise NotImplementedError()

    def float_images(self,rad = 0.99):
        N = self.get_num_pixels()
        coords = geom.gencoords(N,2).reshape((N**2,2))
        Cs = n.sum(coords**2,axis=1).reshape((N,N)) > (rad*N/2.0 - 1.5)**2
        
        vals = []
        for img in self:
            corner_pixels = img[Cs]

            float_val = n.mean(corner_pixels)
            img -= float_val
            
            vals.append(float_val)
        
        return vals
    
    def window_images(self,rad = 0.99):
        N = self.get_num_pixels()
        coords = geom.gencoords(N,2).reshape((N**2,2))
        Cs = n.sum(coords**2,axis=1).reshape((N,N)) > (rad*N/2.0 - 1.5)**2
        
        for img in self:
            img[Cs] = 0
        
    
    def scale_images(self,scale):
        for img in self:
            img *= scale
        
            
    def compute_variance(self):
        vals = []
        for img in self:
            vals.append(n.mean(img**2,dtype=n.float64))
        return n.mean(vals,dtype=n.float64)

    def estimate_noise_variance(self,esttype='robust',zerosub=False,rad = 1.0):
        N = self.get_num_pixels()
        Cs = n.sum(geom.gencoords(N,2).reshape((N**2,2))**2,axis=1).reshape((N,N)) > (rad*N/2.0 - 1.5)**2
        vals = []
        for img in self:
            cvals = img[Cs]
            vals.append(cvals)

        if esttype == 'robust':
            if zerosub:
                var = (1.4826*n.median(n.abs(n.asarray(vals) - n.median(vals))))**2
            else:
                var = (1.4826*n.median(n.abs(vals)))**2
        elif esttype == 'mle':
            var = n.mean(n.asarray(vals)**2,dtype=n.float64)
            if zerosub:
                var -= n.mean(vals,dtype=n.float64)**2
        return var


class MRCImageStack(ImageStackBase):
    def __init__(self,stkfile,psz,scale=1.0):
        ImageStackBase.__init__(self)
        self.stkfile = stkfile
        raw_imgdata, hdr = mrc.readMRC(stkfile, inc_header=True)
        
        assert hdr['nx'] == hdr['ny']
        
        if psz is None:
            assert hdr['xlen'] == hdr['ylen']
            psz = hdr['xlen']/hdr['nx']
            print 'No pixel size specified, using the one defined by the MRC header: {0}A'.format(psz)
            print '   WARNING: This may be inaccurate!'

        if scale != 1.0:
            Nsz = [ n.round(scale*raw_imgdata.shape[0]), \
                    n.round(scale*raw_imgdata.shape[1]), \
                    raw_imgdata.shape[2] ]
            imgdata = resize_ndarray(raw_imgdata,Nsz,axes=(0,1))
            self.pixel_size = (psz*raw_imgdata.shape[0]) / imgdata.shape[1] 
        else:
            imgdata = raw_imgdata
            self.pixel_size = psz
        
        self.imgdata = n.transpose(imgdata,axes=(2,0,1))
        
        self.num_images = self.imgdata.shape[0]
        self.num_pixels = self.imgdata.shape[1]

    def scale_images(self,scale):
        self.imgdata *= scale
        
    def __iter__(self):
        return self.imgdata.__iter__()

    def get_image(self,idx):
        return self.imgdata[idx]


class CombinedImageStack(ImageStackBase):
    def __init__(self,stacks):
        ImageStackBase.__init__(self)

        self.stacks = stacks
        
        self.pixel_size = self.stacks[0].get_pixel_size()
        self.num_pixels = self.stacks[0].get_num_pixels()
        self.num_images = 0
        self.stack_idxs = n.empty((len(self.stacks),2),dtype=n.uint32)
        for i,stk in enumerate(self.stacks):
            assert stk.get_pixel_size() == self.pixel_size
            assert stk.get_num_pixels() == self.num_pixels
            self.stack_idxs[i,0] = self.num_images
            self.num_images += stk.get_num_images()
            self.stack_idxs[i,1] = self.num_images

    def scale_images(self,scale):
        for stk in self.stacks:
            stk.scale_images(scale)

    def __iter__(self):
        return CombinedImageStack_Iter(self)

    def get_image(self,idx):
        for stk,(startI,endI) in zip(self.stacks,self.stack_idxs):
            if startI <= idx and idx < endI:
                return stk.get_image(idx-startI)


class CombinedImageStack_Iter:
    def __init__(self,stk,stkI=0,imgI=-1):
        self.stack = stk
        self.stkI = stkI
        self.imgI = imgI

    def __iter__(self):
        return self
    
    def next(self):
        self.imgI += 1
        if self.imgI == self.stk.stack_idxs[self.stkI,1]:
            self.stkI += 1
            self.imgI = 0
        
        if self.stkI == self.stk.stack_idxs.shape[0]:
            raise StopIteration
        else:
            return self.stk.stacks[self.stkI].get_image(self.imgI)


class FourierStack:
    def __init__(self,basestack,premult=None,zeropad=0, caching=True):
        self.stack = basestack
        self.fft_lock = Lock()
        self.set_transform(premult,zeropad)
        self.caching = caching

    def set_transform(self,premult,zeropad):
        self.transformed = {}
        self.premult = premult
        self.zeropad = int(zeropad*(self.stack.get_num_pixels()/2))

        Nzp = self.get_num_pixels()
        if self.premult is not None:
            assert self.premult.shape[0] == Nzp  
            assert self.premult.shape[1] == Nzp  

        if self.zeropad:
            self.zpimg = n.zeros((Nzp,Nzp),dtype=density.real_t)

        self.fspacesum = n.zeros(2*(self.get_num_pixels(),),
                                dtype = density.complex_t)
        self.powersum = n.zeros(2*(self.get_num_pixels(),),
                                dtype = density.real_t)
        self.nsum = 0

    def get_mean(self):
        return self.fspacesum/self.nsum
    
    def get_variance(self):
        return self.powersum/self.nsum

    def get_pixel_size(self):
        return self.stack.pixel_size
    
    def get_num_images(self):
        return self.stack.num_images
    
    def get_num_pixels(self):
        return 2*self.zeropad + self.stack.num_pixels
    
    def get_image(self,idx):
        if not self.caching:
            self.transformed = {}
        if idx not in self.transformed:
            self.fft_lock.acquire()
            if self.zeropad:
                N = self.stack.get_num_pixels()
                img = self.zpimg
                img[self.zeropad:(N+self.zeropad),self.zeropad:(N+self.zeropad)] = self.stack.get_image(idx)
            else:
                img = self.stack.get_image(idx)

            if self.premult is not None:
                img = self.premult*img

            self.transformed[idx] = density.real_to_fspace(img)
            self.fft_lock.release()

            self.fspacesum += self.transformed[idx]
            self.powersum += self.transformed[idx].real**2 + self.transformed[idx].imag**2
            self.nsum += 1 
             
        return self.transformed[idx]
