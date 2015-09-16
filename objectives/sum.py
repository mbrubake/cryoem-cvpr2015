from objective import Objective

import density

class SumObjectives(Objective):
    def __init__(self,fspace,objs,ws=None):
        Objective.__init__(self,fspace)
        self.objs = objs
        if ws == None:
            self.ws = [None]*len(objs)
        else:
            self.ws = ws

    def setup(self,params,diagout,statout,ostream):
        Objective.setup(self,params,diagout,statout,ostream)
        for obj in self.objs: obj.setup(params,diagout,statout,ostream)

    def set_dataset(self,cryodata):
        Objective.set_dataset(self,cryodata)
        for obj in self.objs: obj.set_dataset(cryodata)
        
    def set_params(self,cparams):
        for obj in self.objs: obj.set_params(cparams)

    def set_data(self,cparams,minibatch):
        Objective.set_data(self,cparams,minibatch)
        for obj in self.objs: obj.set_data(cparams,minibatch)

    def get_preconditioner(self,precond_type):
        precond = 0
        for (w,obj) in zip(self.ws,self.objs):
            if w is None:
                precond = precond + obj.get_preconditioner(precond_type)
            else:
                precond = precond + w*obj.get_preconditioner(precond_type)
        return precond
        
        
    def learn_params(self, params, cparams, M=None, fM=None):
        anyfspace = any([obj.fspace for obj in self.objs])
        anyrspace = any([not obj.fspace for obj in self.objs])
            
        N = None
        if fM is None and anyfspace:
            assert M is not None, 'M or fM must be set!'
            N = M.shape[0]
            fM = density.real_to_fspace(M)
        elif fM is not None:
            N = fM.shape[0]

        if M is None and anyrspace:
            assert fM is not None, 'M or fM must be set!'
            N = fM.shape[0]
            M = density.fspace_to_real(fM)
        elif M is not None:
            assert N is None or N == M.shape[0]
            N = M.shape[0]

        assert N is not None

        for obj in self.objs:
            obj.learn_params(params,cparams,M=M,fM=fM)

    def eval(self, M=None, fM=None, compute_gradient=True, all_grads=False,**kwargs):
        anyfspace = any([obj.fspace for obj in self.objs])
        anyrspace = any([not obj.fspace for obj in self.objs])

        N = None
        if fM is None and anyfspace:
            assert M is not None, 'M or fM must be set!'
            N = M.shape[0]
            fM = density.real_to_fspace(M)
        elif fM is not None:
            N = fM.shape[0]

        if M is None and anyrspace:
            assert fM is not None, 'M or fM must be set!'
            N = fM.shape[0]
            M = density.fspace_to_real(fM)
        elif M is not None:
            assert N is None or N == M.shape[0]
            N = M.shape[0]

        assert N is not None

        logP = 0
        logPs = []
        if compute_gradient:
            if all_grads:
                dlogP = density.zeros_like(fM) if self.fspace else density.zeros_like(M)
                dlogPs = []
            else:
                if (not self.fspace) or anyrspace:
                    dlogPdM = density.zeros_like(M)
                if self.fspace or anyfspace:
                    dlogPdfM = density.zeros_like(fM)
        outputs = {}
        for w,obj in zip(self.ws,self.objs):
            if compute_gradient:
                clogP, cdlogP, coutputs = obj.eval(M = M, fM = fM, 
                                                   compute_gradient = compute_gradient,
                                                   **kwargs)
                if w is not None and w != 1:
                    clogP *= w
                    cdlogP *= w

                if all_grads:
                    if obj.fspace == self.fspace:
                        dlogPs.append(cdlogP)
                    elif self.fspace:
                        dlogPs.append(density.real_to_fspace(cdlogP))
                    else:
                        dlogPs.append(density.fspace_to_real(cdlogP))
                    dlogP += dlogPs[-1]
                else:
                    if obj.fspace:
                        dlogPdfM += cdlogP
                    else:
                        dlogPdM += cdlogP

            else:
                clogP, coutputs = obj.eval(M = M, fM = fM,
                                           compute_gradient = compute_gradient,
                                           **kwargs)
                if w is not None and w != 1:
                    clogP *= w

            logP += clogP
            logPs.append(clogP)
            outputs.update(coutputs)

        if compute_gradient and not all_grads:
            if self.fspace:
                dlogP = dlogPdfM
                if anyrspace:
                    dlogP += density.real_to_fspace(dlogPdM)
            else:
                dlogP = dlogPdM
                if anyfspace:
                    dlogP += density.fspace_to_real(dlogPdfM)
        
        outputs['all_logPs'] = logPs
        if compute_gradient and all_grads:
            outputs['all_dlogPs'] = dlogPs

        if compute_gradient:
            return logP, dlogP, outputs 
        else:
            return logP, outputs 
