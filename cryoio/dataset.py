import numpy as n

from numpy.random import RandomState

def estimate_mean_std(vals,esttype):
    if esttype == 'robust':
        mean = n.median(vals)
        std = 1.4826*n.median(n.abs(vals - mean))
    elif esttype == 'mle':
        mean = n.mean(vals)
        std = n.std(vals - mean)
    return mean, std

class CryoDataset:
    def __init__(self,imgstack,ctfstack):
        self.imgstack = imgstack
        self.ctfstack = ctfstack
        
        assert self.imgstack.get_num_images() == self.ctfstack.get_num_images()
        
        self.N = self.imgstack.get_num_pixels()
        self.pixel_size = self.imgstack.get_pixel_size()

    
    def compute_noise_statistics(self):
        self.mleDC_est = self.estimate_dc()
        self.noise_var = self.imgstack.estimate_noise_variance()
        self.data_var = self.imgstack.compute_variance()
        
        print 'Dataset noise profile'
        print '  Noise: {0:.3g}'.format(n.sqrt(self.noise_var))
        print '  Data: {0:.3g}'.format(n.sqrt(self.data_var))
        assert self.data_var > self.noise_var
        self.signal_var = self.data_var - self.noise_var 
        print '  Signal: {0:.3g}'.format(n.sqrt(self.signal_var))
        print '  Signal-to-Noise Ratio: {0:.1f}% ({1:.1f}dB)'.format(100*self.signal_var/self.noise_var, 10*n.log10(self.signal_var/self.noise_var))

    def normalize_dataset(self):
        self.imgstack.scale_images(1.0/n.sqrt(self.noise_var))
        self.ctfstack.scale_ctfs(1.0/n.sqrt(self.noise_var))
    
        self.data_var = self.data_var/self.noise_var
        self.signal_var = self.signal_var/self.noise_var    
        self.noise_var = 1.0


    def divide_dataset(self,minibatch_size,testset_size,partition,num_partitions,seed):
        self.rand = RandomState(seed)
        
        self.N_D = self.imgstack.get_num_images()
        self.idxs = self.rand.permutation(self.N_D)
        
        print "Dividing dataset of {0} images with minisize of {1}".format(self.N_D,minibatch_size)
        if testset_size != None:
            print "  Test Images: {0}".format(testset_size)
            self.test_idxs = self.idxs[0:testset_size]
            self.train_idxs = self.idxs[testset_size:]
        else:
            self.train_idxs = self.idxs
            self.test_idxs = []
        
        if num_partitions > 1:
            print "  Partition: {0} of {1}".format(partition+1,num_partitions)
            N_D = len(self.train_idxs)
            partSz = N_D/num_partitions
            self.train_idxs = self.train_idxs[partition*partSz:(partition+1)*partSz]

        self.N_D_Test = len(self.test_idxs)
        self.N_D_Train = len(self.train_idxs)
        numBatches = int(n.floor(float(self.N_D_Train)/minibatch_size))
        real_minisize = int(n.floor(float(self.N_D_Train)/numBatches))
        N_Rem = self.N_D_Train - real_minisize*numBatches
        numRegBatches = numBatches - N_Rem
        batchInds = [ (real_minisize*i, real_minisize*(i+1)) \
                      for i in xrange(numRegBatches) ] + \
                    [ (real_minisize*numRegBatches + (real_minisize+1)*i,
                       min(real_minisize*numRegBatches + (real_minisize+1)*(i+1),self.N_D_Train)) \
                      for i in xrange(N_Rem) ]
        self.batch_idxs = n.array(batchInds)
        self.N_batches = self.batch_idxs.shape[0]
        self.batch_order = self.rand.permutation(self.N_batches)

        batch_sizes = self.batch_idxs[:,1] - self.batch_idxs[:,0]

        print "  Train Images: {0}".format(self.N_D_Train)
        print "  Minibatches: {0}".format(self.N_batches)
        print "  Batch Size Range: {0} - {1}".format(batch_sizes.min(),batch_sizes.max())
        
        self.minibatch_size = minibatch_size
        self.testset_size = testset_size
        self.partition = partition
        self.num_partitions = num_partitions

        self.reset_minibatches(True)

    def get_dc_estimate(self):
        return self.mleDC_est

    def estimate_dc(self,esttype='robust'):
        N = self.N
        
        obs = []
        ctf_dcs = {}
        zeros = n.zeros((1,2))
        for img_i,img in enumerate(self.imgstack):
            ctf_i = self.ctfstack.get_ctf_idx_for_image(img_i)
            if ctf_i not in ctf_dcs:
                ctf_dcs[ctf_i] = self.ctfstack.get_ctf(ctf_i).compute(zeros)
                 
            obs.append(n.mean(img) * n.sqrt(float(N)) / ctf_dcs[ctf_i])
            
        obs = n.array(obs)
        mleDC, mleDC_std = estimate_mean_std(obs, esttype)
        mleDC_est_std = mleDC_std /  n.sqrt(len(obs))
        
        return mleDC, mleDC_std, mleDC_est_std
    
    def set_datasign(self,datasign):
        mleDC, _, mleDC_est_std = self.get_dc_estimate()
        datasign_est = 1 if mleDC > 2*mleDC_est_std else -1 if mleDC < -2*mleDC_est_std else 0
        print "Estimated DC Component: {0:.3g} +/- {1:.3g}".format(mleDC,mleDC_est_std)

        if datasign == 'auto':
            if datasign_est == 0:
                print "  WARNING: estimated DC component has large variance, detected sign could be wrong."
                datasign = n.sign(mleDC)
            else:
                datasign = datasign_est
        else:
            if datasign_est*datasign < 0:
                print "  WARNING: estimated DC component and specified datasign disagree; be sure this is correct!"
            
        if datasign != 1:
            print "  Using negative datasign"
            assert datasign == -1
            self.ctfstack.flip_datasign()
        else:
            print "  Using positive datasign"
            assert datasign == 1

    def reset_minibatches(self,epochReset=True):
        self.curr_batch = None
        self.epoch_frac = 0

        if epochReset:
            self.epoch = 0
            self.data_visits = 0

    def get_testbatch(self):
        miniidx = self.test_idxs
        ret = {'img_idxs':miniidx, 
               'ctf_idxs':self.ctfstack.get_ctf_idx_for_image(miniidx),
               'N_M':len(miniidx), 'test_batch':True}
        
        return ret
    
    def get_next_minibatch(self,shuffle_minibatches):
        if self.curr_batch == None:
            self.curr_batch = 1
            batchInd = 0
            newepoch = False
        else:
            batchInd = self.curr_batch
            self.curr_batch = (self.curr_batch+1)%self.N_batches
            newepoch = batchInd == 0

        if newepoch:
            if shuffle_minibatches:
                self.batch_order = self.rand.permutation(self.N_batches)
            self.epoch = self.epoch + 1
            self.epoch_frac = 0

        batch_id = self.batch_order[batchInd]

        startI = self.batch_idxs[batch_id,0]
        endI = self.batch_idxs[batch_id,1]
        miniidx = self.train_idxs[startI:endI]

        self.data_visits += endI - startI
        self.epoch_frac += float(endI - startI)/self.N_D_Train
      
        ret = {'img_idxs':miniidx, 
               'ctf_idxs':self.ctfstack.get_ctf_idx_for_image(miniidx),
               'N_M':len(miniidx), 'id':batch_id, 'epoch':self.epoch + self.epoch_frac,
               'num_batches': self.N_batches, 'newepoch':newepoch, 'test_batch':False }

        return ret

    def get_epoch(self,frac=False):
        if self.epoch == None: # Data not yet loaded
            return 0

        if frac:
            return self.epoch + self.epoch_frac
        else:
            return self.epoch


