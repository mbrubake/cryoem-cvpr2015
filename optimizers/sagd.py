import numpy as n
import base

import os
import cPickle as pkl

from util import FiniteRunningSum


def mypolyfit(x,y,dx,dy,deg=None):
    if deg == None:
        deg = len(x) + len(dx) - 1
    x = n.asarray(x).reshape((-1,1))
    y = n.asarray(y).reshape((-1,1))
    dx = n.asarray(dx).reshape((-1,1))
    dy = n.asarray(dy).reshape((-1,1))

    A = n.vstack([ n.hstack([ x**d for d in reversed(range(1,deg+1)) ] + [n.ones(x.shape)]), \
                   n.hstack([ d*(dx**(d-1)) for d in reversed(range(1,deg+1)) ] + [n.zeros(dx.shape)]) ])
    b = n.vstack([y,dy])

    p, _, _, _ = n.linalg.lstsq(A,b)
    p = p.reshape((-1,))

    return p


def find_L(x, f, dfdx, evalobj, L, maxIts = None, \
           gradientCheck = False, ostream = None, precond = None, \
           minStep = 2, maxStep = 10, optThresh = 1.1, maxPolyVals = 10):
    """
    Perform a really simple line search to estimate the Lipschitz
    constant which is used to set the step-size
    """
    initL = n.float32(L)

    if precond is None:
        g = dfdx
        gstep = dfdx
    else:
        g = precond*dfdx
        gstep = precond**2*dfdx

    gnorm2 = n.sum(g**2,dtype=n.float64)
    gnorm = n.sqrt(gnorm2)

#    print 'gnorm = {0}, relgnorm = {1}'.format(gnorm,gnorm/n.linalg.norm(M))

    if gradientCheck:
        eps = 5e-4
        dfdxnorm = n.linalg.norm(dfdx)
        d = dfdx * ( eps / dfdxnorm )
        
        fp, _ = evalobj( x + d.reshape(x.shape), compute_gradient = False, intermediate=True )
        fm, _ = evalobj( x - d.reshape(x.shape), compute_gradient = False, intermediate=True )
        outstr = 'grad = {0}, fdgrad = {1}'.format(dfdxnorm,(fp - fm)/(2*eps))
        if ostream != None:
            ostream(outstr)
        else:
            print outstr

    goodL = None # holds the smallest known good L
    badL = None # holds the largest known bad L
    # note that badL < goodL

    base_invLVals = [0.0]
    base_condVals = [0.0]
    invLVals = []
    condVals = []
    invLDVals = [0]
    condDVals = [0.5*gnorm2]
    its = 0
    while maxIts == None or its < maxIts:
        its += 1
        if goodL == None and badL == None:
            currL = initL
            predFitL = None
        else:
            # Fit a polynomial and extract its roots
#            fitP = n.polyfit(invLVals,condVals,len(invLVals)-1)
            fitP = mypolyfit(invLVals + base_invLVals,
                             condVals + base_condVals,
                             invLDVals,condDVals)
            fitRoots = n.roots(fitP)

            # Weed out real and negative roots
            fitRoots = fitRoots[n.isreal(fitRoots)].real
            fitRoots = fitRoots[fitRoots > 0]

            # Convert the roots to their corresponding L values
            fitL = 1.0/fitRoots

            # Remove L values that are not within the bounds
            if badL != None:
                fitL = fitL[fitL > badL]
            if goodL != None:
                fitL = fitL[fitL < goodL]

            if len(fitL) > 0:
                predFitL = n.min(fitL)
                if goodL != None:
                    predFitL = 0.95*predFitL + 0.05*goodL
                else:
                    predFitL *= 1.05
            else:
                predFitL = None

            if goodL == None: # and badL != None
                if predFitL != None:
                    currL = max(minStep*badL,min(maxStep*badL,predFitL))
                else:
                    currL = minStep*badL
            elif badL == None: # and goodL != None
                if predFitL != None:
                    currL = min(goodL/minStep,max(goodL/maxStep,predFitL))
                else:
                    currL = goodL/minStep
            else: # goodL != None and badL != None
                if predFitL != None:
                    minPredL = badL + 0.01*(goodL-badL)
                    maxPredL = goodL - 0.01*(goodL-badL)
                    currL = min(maxPredL,max(minPredL,predFitL))
                else:
                    currL = n.sqrt(badL*goodL)

        xp = x - (gstep.reshape(x.shape)/currL)
        fp, _ = evalobj( xp, compute_gradient = False, intermediate=True )
        condVal = (f - 0.5*gnorm2/currL) - fp
        relCondVal = (f - fp) / (0.5*gnorm2/currL)
        good = condVal >= 0

        # Only keep the most recent evaluations
        if maxPolyVals is not None and len(invLVals) >= maxPolyVals:
            invLVals.pop(0)
            condVals.pop(0)
        invLVals.append(1.0/currL)
        condVals.append(f - fp - 0.5*gnorm2/currL)

        eps = gnorm/currL
        if ostream != None:
            ostream('{its}: L = {0}, abs cond = {1}, rel cond = {2}, predL = {3}'.format(currL,condVal,relCondVal,predFitL,its=its))

        if its == 1:
            initfp = fp

        if good:
            goodL = currL
            goodfp = fp

            if abs(relCondVal) < optThresh: # optThres of optimal
                break
        else:
            if f - fp == 0:
                break

            badL = currL
            badfp = fp

            if 0.5*gnorm2/L < 1e-8:
                break

    if goodL != None:
        finalL = goodL
        finalfp = goodfp
        solnType = 'good'
    elif badL != None:
        finalL = badL
        finalfp = badfp
        solnType = 'better'
    else:
        finalL = initL
        finalfp = initfp
        solnType = 'init'

    if ostream != None:
        ostream("Found {4} L = {0} (f - fp = {2}, 0.5*gnorm2/L = {3}) after {1} its".format(finalL,its,f-finalfp,0.5*gnorm2/finalL,solnType))

    return finalL



class SAGDStep(base.BaseStep):
    def __init__(self,ind_L = False, ng = False, alt_sum = True):
        base.BaseStep.__init__(self)

        self.g_history = FiniteRunningSum(second_order=ng)

        self.L = None
        
        self.alt_sum = alt_sum # FIXME: currently ignored
        self.ind_L = ind_L # FIXME: currently ignored

        self.precond = None
        self.ng = ng
        
        self.prev_max_freq = None


    def add_batch_gradient(self,batch,curr_g,params):
        mu = params.get('sagd_momentum',None)

        if self.g_history.N_sum != params['num_batches']:
            # Reset gradient g_history if minibatch size changes.
            self.g_history.setup(curr_g, batch['num_batches'])
            
        return self.g_history.set_value(batch['id'], curr_g, mu)

    def save_L0(self,params):
        with open(os.path.join(params['exp_path'],'sagd_L0.pkl'),'wb') as fp:
            pkl.dump(self.L,fp,protocol=2)

    def load_L0(self,params):
        try:
            with open(os.path.join(params['exp_path'],'sagd_L0.pkl'),'rb') as fp:
                L0 = pkl.load(fp)
        except:
            L0 = params.get('sagd_L0',1.0)
        return L0

    def do_step(self, x, params, cryodata, evalobj, batch, **kwargs):
#         ostream = self.ostream
#         ostream = None
        ostream = self.ostream if self.L is None else None
        inc_L = params.get('sagd_incL',False)
        do_ls = self.L is None or params.get('sagd_linesearch',False)
        max_ls_its = params.get('sagd_linesearch_maxits',5) if self.L is not None else None
        grad_check = params.get('sagd_gradcheck',False)
        minStep = 1.05 if self.L is not None else 2
        maxStep = 100.0 if self.L is not None else 10
        optThresh = params.get('sagd_linesearch_accuracy',1.01)
        g2_lambdaw = params.get('sagd_lambdaw',10) # weight of the prior which biases the covariance estimate towards lambda 
        eps0 = params.get('sagd_learnrate',1.0/16.0)
        reset_precond = params.get('sagd_reset_precond',False)
        F_max_range = params.get('sagd_precond_maxrange',1000.0)
        use_saga = params.get('sagd_sagastep',False)

        if self.g_history is not None and reset_precond:
            self.g_history.reset_meansq()
            do_ls = True

        # Evaluate the gradient
        f, g, res_train = evalobj( x, all_grads = True )

        assert len(res_train['all_dlogPs']) == 2
        g_like = res_train['all_dlogPs'][0]
        g_prior = res_train['all_dlogPs'][1]

        if use_saga:
            prev_g_hat = self.g_history.get_mean()

        # Add the gradient to the g_history
        prev_g_like, _ = self.add_batch_gradient(batch,g_like,params)

        # Get the current average g
        if not use_saga:
            curr_g_hat = self.g_history.get_mean()
            totghat = curr_g_hat + g_prior
        else:
            totghat = prev_g_hat + g_prior + (g_like - prev_g_like)

        if self.ng:
            # Update the preconditioner when we're going to do a linesearch
            if do_ls:
                curr_g_var = n.maximum(0,self.g_history.get_meansq())
                curr_w2sum = self.g_history.get_w2sum()
                mean_var = n.mean(curr_g_var)
                if curr_w2sum > 1:
                    F =   mean_var * (g2_lambdaw / (curr_w2sum + g2_lambdaw)) + \
                        curr_g_var * (curr_w2sum / (curr_w2sum + g2_lambdaw))
                else:
                    F = mean_var
                    
                minF = F.min()
                maxF = F.max()
                minF_trunc = n.maximum(minF,maxF/F_max_range)


                self.precond = n.sqrt(minF_trunc)/n.sqrt(n.maximum(minF_trunc,F))
                if ostream is not None:
                    ostream("  F min/max = {0} / {1}, var min/max/mean = {2} / {3} / {4}".format(n.min(F),n.max(F),
                                                                    n.min(curr_g_var),n.max(curr_g_var),mean_var))
#                    ostream("  precond range = ({0},{1}), Lscale = {2}".format(precond.min(),precond.max(),Lscale))
            precond = self.precond
        else:
            precond = None

        init_L = self.L is None

        # Gradually increase the current L if requested
        if inc_L != 1.0 and not init_L:
            self.L *= inc_L

        if not init_L:
            L0 = self.L
        else:
            L0 = self.load_L0(params)

        if do_ls:
            # Perform line search if we haven't found a value of L yet
            # and/or check that the current L satisfies the conditions
            self.L = find_L( x, f, g, evalobj, L0, max_ls_its, \
                             gradientCheck=grad_check, \
                             ostream=ostream, precond=precond, \
                             minStep=minStep, maxStep=maxStep, \
                             optThresh=optThresh)

        currL = self.L

        if init_L:
            self.save_L0(params)

        eps = eps0/currL

        dx = -eps*totghat
        if self.ng:
            self.statout.output(sagd_precond_min=[precond.min()],
                                sagd_precond_max=[precond.max()])
            dx *= precond**2
#             if ostream is not None: 
#                 ostream("  step size range = ({0},{1})".format(precond.min()**2/currL,precond.max()**2/currL))
#         else:
#             if ostream is not None:
#                 ostream("  step size = {0}".format(1.0/currL))

#        dMnorm = n.linalg.norm(dM)
#        print "||dM||/eps = {0}, ||dM|| / (||M|| * eps) = {1}".format(dMnorm/eps,dMnorm/(Mnorm*eps))
        ghatnorm = n.linalg.norm(totghat)
        self.statout.output(sagd_L=[self.L],
                            sagd_gnorm=[ghatnorm],
                            sagd_eps=[eps])
        

        return f, g.reshape((-1,1)), dx, res_train, 0  # 0 extra operations on data


