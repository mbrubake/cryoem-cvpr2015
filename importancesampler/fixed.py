import numpy as n
from util import logsumexp

def multinomial_sample_residual(N,probs):
    """
    This implements residual sampling from a multinomial distribution.
    The net result is a samplecount value with the same expected values but
    with lower variance.
    """
    Nprobs = N*probs
    samplecount = n.require(n.floor(Nprobs),dtype=n.int32)
    R = samplecount.sum()
    pbar = (Nprobs - samplecount) / (N - R)
    samplecount += n.random.multinomial(N - R,pbar)
    return samplecount

class FixedImportanceSampler():
    """Class which draws weighted samples from a fixed set of points"""
    def __init__(self, suffix):
        self.num_dist = None
        self.domain = None
        self.prev_vals = None
        self.globalvals = None
        self.prev_globalvals = None
        self.suffix = suffix
        self.params = None

        self.updates = {}
        self.cached_samples = {}

    def set_domain(self,domain):
        self.domain = domain

    def setup(self, params, num_dist, num_train_dist, domain):
        self.params = params
        self.num_train_dist = num_train_dist
        do_importance_sampling = params.get('is_on'+self.suffix, False)
        if self.num_dist != num_dist:
            self.num_dist = num_dist    # number of distributions to store 
            self.set_domain(domain)
            if do_importance_sampling:
                self.prev_vals = {}

                self.prev_globalvals = []
                self.prev_globalw = []
                self.prev_globalcnts = []

                self.globalvals = n.zeros(len(domain), dtype=n.float64)
                self.globalw = 0
                self.globalcnt = 0
                self.global_contrib = n.empty(self.num_dist, dtype = n.int32)
                self.global_contrib[:] = 0
                self.global_upd = n.zeros(len(domain), dtype=n.float64)
            else:
                self.prev_vals = None
                self.globalvals = None
                self.prev_globalvals = None
                self.global_upd = None

        elif self.domain != domain:
            old_domain = self.domain
            self.set_domain(domain)

            # The domain changed, need to refine the existing values
            if do_importance_sampling:
                # Convert the old global distributions if they're still used
                self.prev_globalvals = [ self.evaluate_kernel(None, vals,
                                                              old_domain, self.params) \
                                         for (w,cnt,vals) in zip(self.prev_globalw,self.prev_globalcnts,self.prev_globalvals) \
                                         if cnt > 0 ]
                self.prev_globalw = [ w for (w,cnt) in zip(self.prev_globalw,self.prev_globalcnts) if cnt > 0 ]
                self.prev_globalcnts = [ cnt for cnt in self.prev_globalcnts if cnt > 0 ]

                # Convert the previous global distribution
                self.prev_globalvals.append(self.evaluate_kernel(None, self.globalvals,
                                                                 old_domain, self.params))
                self.prev_globalw.append(self.globalw)
                self.prev_globalcnts.append(self.globalcnt)

                # Allocate the new global distribution
                self.globalvals = n.zeros(len(domain), dtype=n.float64)
                self.globalw = 0
                self.globalcnt = 0
                self.global_contrib[self.global_contrib < 0] -= 1
                self.global_contrib[self.global_contrib == 1] = -1
                self.global_upd = n.zeros(len(domain), dtype=n.float64)

        self.global_cache = None

    def get_global_dist(self):
        if self.global_cache is not None:
            return self.global_cache

        totalw = n.sum(self.prev_globalw) + self.globalw
        if totalw == 0:
            return 1.0/len(self.domain)

#        print >>sys.stderr, "{3}: {0} / {1} / {2}".format(self.num_dist - totalw, self.prev_globalw, self.globalw, self.suffix)
        globaldist = float(self.num_train_dist - totalw)/len(self.domain)

        for (w,vals) in zip(self.prev_globalw,self.prev_globalvals):
            if n.isfinite(w) and w > 0:
                globaldist = globaldist + w*vals

        if n.isfinite(self.globalw) and self.globalw > 0:
            globaldist = globaldist + self.globalvals

        globaldist /= n.sum(globaldist)

        self.global_cache = globaldist

        return globaldist

    def get_global_logdist(self):
        return n.log(self.get_global_dist())

    def get_image_dist(self,im,params = None):
        if params is None:
            params = self.params

        if im in self.prev_vals:
            samples, _, expvals, pdom, _, _, _ = self.prev_vals[im]

            ret = self.evaluate_kernel(samples,expvals,pdom,params,logspace=False)
        else:
            ret = n.empty(len(self.domain), 
                          dtype=n.float64)                
            ret[:] = 1.0/len(self.domain)
        return ret

    def get_image_logdist(self,im,params = None):
        if params is None:
            params = self.params

        if im in self.prev_vals:
            samples, logvals, _, pdom, _, _, _ = self.prev_vals[im]

            ret = self.evaluate_kernel(samples,logvals,pdom,params,logspace=True)
        else:
            ret = n.empty(len(self.domain), 
                          dtype=n.float64)                
            ret[:] = -n.log(len(self.domain))
        return ret

    def get_image_priorprob(self,im):
        if im in self.prev_vals:
            # FIXME: Idea: Adjust prior prob based on inlier probability!
            # Maybe not the best idea, would become very slow for outliers.
            # Need to instead measure stability of the importance distribution.
            # Maybe compute the ESS of the final sample set?
            _,_,_,_,pparams,_,_ = self.prev_vals[im]
            return pparams.get('is_prior_prob'+self.suffix,pparams.get('is_prior_prob',0.05))
        else:
            return 1.0

    def sample(self, dist):
        # sample from dist (index) using params (is_*)
        # returns sample indices and sample weights (may be smaller than
        # computed num_samples if there are overlaps)
        # the sample weights are not the inverse probabilities, but rather
        # the actual weights you need to multiply the weighted function
        # evaluations by.
        # So before we computed sum_i w_i f(x_i)
        # now we compute        sum_j sampleweight_j w_sample_j f(x_sample_j)
        # for instance, if the number of samples == domain and the importance
        # distribution is uniform then the sampleweights are all 1. 

        # 1) generate prob dist over domain
        # 2) sample from this dist
        # 3) compute weights from samples and probs (weight = 1 / prob / numsamples)
        # 4) remove dups and multiply weights by number of dups
        # 5) return final samples and weights

        if dist in self.cached_samples:
            # If we've already drawn samples since our last update, use those.
            # This ensures that multiple evaluations of the objective uses a
            # consistent set of IS samples.
            return self.cached_samples[dist][0:2]

        params = self.params
        global_prob = params.get('is_global_prob'+self.suffix,params.get('is_global_prob',0))
        num_samples = params.get('is_num_samples'+self.suffix,params.get('is_num_samples','auto'))
        ess_scale = params.get('is_ess_scale'+self.suffix,params.get('is_ess_scale',5.0))
        do_importance_sampling = params.get('is_on'+self.suffix, False) and len(self.domain) > 1

        if do_importance_sampling:
            prior_prob = self.get_image_priorprob(dist)
            if prior_prob < 1.0:
                # Use some of the prior and some of the previous distribution
                logprobs = self.get_image_logdist(dist)

                # Mix the current estimate with a prior distribution
                # probs *= (1.0 - prior_prob)
                logprobs += n.log(1.0-prior_prob)
                # probs += (1.0 - global_prob) * prior_prob / float(len(self.domain))
                logprobs = n.logaddexp(logprobs,n.log((1.0 - global_prob) * prior_prob / float(len(self.domain))))
            else:
                # Using only the prior distribution
                logprobs = n.empty(len(self.domain))
                logprobs[:] = n.log((1.0 - global_prob) / float(len(self.domain)))

            if global_prob > 0 and prior_prob > 0:
                # Part of the prior prob is the global estimate of directions
                # probs += (prior_prob * global_prob) * self.globalvals
                logprobs = n.logaddexp(logprobs, n.log(prior_prob * global_prob) + self.get_global_logdist())

            # Normalize the distribution
            # probs /= probs.sum()
            logprobs[n.logical_not(n.isfinite(logprobs))] = -n.inf
            lse = logsumexp(logprobs)
            if n.isfinite(lse):
                logprobs -= lse
            else:
                print "WARNING: lse for {1} is not finite: {0}".format(lse,self.suffix)
                print "logprobs = {0}".format(logprobs)
                logprobs[:] = -n.log(len(self.domain))

            # Compute effective sample size.
            # ess ~=~ the number of significant components in probs.
            # Note that large values of prior_prob will cause ess to be large.
#            ess = 1.0/n.sum(probs**2)
#            logess = -n.log(n.sum(n.exp(2*logprobs))
            logess = -logsumexp(2*logprobs)
            if not (n.isfinite(logess) and logess > 0):
                print "WARNING: logess is not finite and positive: {0}, {1}".format(logess,n.exp(logess))
                ess = 1
            else:
                ess = n.exp(logess)

            # Tune the number of samples based on the effective sample size.
            if num_samples == 'auto':
                num_samples = min(n.ceil(ess_scale*ess),len(self.domain))

            # Only do sampling if it's likely to save us at least 5%
            if min(ess,num_samples)/len(self.domain) < (1 - 0.05):
                probs = n.exp(logprobs)
                samplecount = n.random.multinomial(num_samples,probs)
#                samplecount = multinomial_sample_residual(num_samples,probs)

                # only evaluate samples with nonzero counts
                samples = n.where(samplecount)[0] 
                weights = samplecount[samples] / (probs[samples] * num_samples)
                fullsampled = False
            else:
                fullsampled = True

        if (not do_importance_sampling) or fullsampled: 
            samples = None
            weights = 1.0
            fullsampled = True

        self.cached_samples[dist] = (samples, weights)
        return self.cached_samples[dist][0:2]

    def record_update(self, dist, samples, samplevalues, samplews, pinlier, testImg, logspace=False):
        do_importance_sampling = self.params.get('is_on'+self.suffix, False)
        if not do_importance_sampling or dist in self.updates:
            return

        # IMPORTANT: Must make sure our stored memory here is distinct from the memory passed in with
        # samplevalues, otherwise it may get changed out from underneath us!!!
        samplevalues = n.require(samplevalues,dtype=n.float64)
        if not logspace:
            logsamplevalues = n.log(n.abs(samplevalues)/samplews)
        else:
            logsamplevalues = samplevalues - n.log(samplews)

        # Anneal the new values a bit
        temp = self.params.get('is_temperature'+self.suffix, self.params.get('is_temperature',1.0))
        assert n.isfinite(temp) and temp > 0
        if temp != 1.0:
            logsamplevalues *= (1.0/temp)
        logsamplevalues[n.logical_not(n.isfinite(logsamplevalues))] = -n.inf
        lse = logsumexp(logsamplevalues)
        assert n.isfinite(lse)
        logsamplevalues -= lse
        
        expsamplevalues = n.exp(logsamplevalues)

        self.updates[dist] = (samples,logsamplevalues,expsamplevalues,self.domain,self.params,pinlier,testImg)

        # Compute the global update here to allow it to be done in parallel
        # NOTE: there is some threading level non-determinism here due to
        # numerical precision and the fact that global_upd may get added in 
        # different orders with different thread timings.
        global_prob = self.params.get('is_global_prob'+self.suffix,self.params.get('is_global_prob',0))
        if not testImg and pinlier > 0 and global_prob > 0:
            self.global_upd += pinlier*self.evaluate_kernel(samples,
                                                            expsamplevalues,
                                                            self.domain,
                                                            self.params,
                                                            logspace=False)
        if self.global_contrib[dist] == 1:
            psamples, _, pexpsamplevalues, pdom, pparams, ppinlier, ptestImg = self.prev_vals[dist]
            global_prob = pparams.get('is_global_prob'+self.suffix,pparams.get('is_global_prob',0))
            if not ptestImg and ppinlier > 0 and global_prob > 0:
                self.global_upd -= ppinlier*self.evaluate_kernel(psamples, pexpsamplevalues, \
                                                                 pdom, pparams, logspace=False)

    def clear_updates(self):
        self.updates.clear()
        self.global_upd[:] = 0
        self.cached_samples.clear()

    def perform_update(self):
        do_importance_sampling = self.params.get('is_on'+self.suffix, False)
        # Don't update if IS isn't on
        if not do_importance_sampling:
            self.clear_updates()
            return

        # Get the update for the global distribution
        for (im,upd) in self.updates.iteritems():
            if upd[6]:
                # Test images aren't included in the global distribution
                continue 
            global_prob = upd[4].get('is_global_prob'+self.suffix,upd[4].get('is_global_prob',0))
            if self.global_contrib[im] < 0:
                # Contribution is in the previous global dist
                _, _, _, _, pparams, ppinlier, ptestImg = self.prev_vals[im]
                assert not ptestImg
                pglobal_prob = pparams.get('is_global_prob'+self.suffix,pparams.get('is_global_prob',0))
                if pglobal_prob > 0:
                    self.prev_globalw[self.global_contrib[im]] -= ppinlier 
                    self.prev_globalcnts[self.global_contrib[im]] -= 1
            if self.global_contrib[im] != 1 and global_prob > 0:
                self.globalw += upd[5]
                self.globalcnt += 1
            self.global_contrib[im] = 1

        # Update the stored values
        self.prev_vals.update(self.updates)

        # Update the global distribution
        self.globalvals += self.global_upd

        # Reset the updates
        self.clear_updates()

    def stats(self):
        if self.sparsevals == None:
            avg_active = 1.0
        else:
            avg_active = (self.sparsevals > 0).sum() / float(self.sparsevals.shape[0] * self.sparsevals.shape[1])
        return {'avg_active': avg_active}

