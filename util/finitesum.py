import numpy as n

class FiniteRunningSum:
    def __init__(self,second_order=False,alt_sum=False):
        self.N_sum = None
        self.second_order = second_order
        
        # FIXME: this isn't done yet so it's turned off by default
        assert alt_sum == False
        self.alt_sum = alt_sum
        
    def setup(self,val,N_sum,allow_decay=True,keep_history=True):
        self.allow_decay=allow_decay
        if self.N_sum is None or not keep_history:
            self.history_mean = None
            self.history_wsum = None
            self.history_upds = 0
            self.history_updated = None
            if self.second_order:
                self.history_meansq = None
                self.history_w2sum = None
        else:
            self.history_mean = self.get_mean()
            self.history_wsum = self.get_wsum()
            if self.second_order:
                self.history_meansq = self.get_meansq()
                self.history_w2sum = self.get_w2sum()

            self.history_updated = set()
            self.history_upds = N_sum
    
        self.w_sum = 0.0
        self.val_sum = n.zeros_like(val)

        self.inds = {}
        self.curr_ind = 0
        self.prev_vals = {}
        if self.allow_decay:
            self.prev_ws = n.zeros(N_sum)
        self.N_sum = N_sum

        if self.second_order:
            self.w2_sum = 0.0
            self.val2_sum = n.zeros_like(val)
            if self.allow_decay:
                self.prev_w2s = n.zeros(N_sum)
        
        if self.alt_sum:
            self.alt_sum_cnt = 0
            self.w_altsum = None
            self.val_altsum = None
            if self.second_order:
                self.w2_altsum = None
                self.val2_altsum = None

    def get_sum(self):
        """ Return the raw, unnormalized sum for the mean. """
        if self.history_upds == 0:
            ret = 0
        else:
            ret = (self.history_wsum * self.history_upds / self.N_sum) * self.history_mean  

        if self.alt_sum and self.val_altsum is not None:
            ret = ret + self.val_altsum

        return ret + self.val_sum

    def get_sumsq(self):
        """ Return the raw, unnormalized sum of squares. """
        assert self.second_order
        
        if self.history_upds == 0:
            ret = 0
        else:
            ret = (self.history_w2sum * self.history_upds / self.N_sum) * self.history_meansq

        if self.alt_sum and self.val2_altsum is not None:
            ret = ret + self.val2_altsum

        return ret + self.val2_sum

    def get_wsum(self):
        """ Return the current weight sum for the mean. """
        if self.history_upds == 0:
            ret = 0
        else:
            ret = (self.history_wsum * self.history_upds / self.N_sum)

        if self.alt_sum and self.w_altsum is not None:
            ret = ret + self.w_altsum

        return ret + self.w_sum
    
    def get_w2sum(self):
        """ Return the current weight sum for the mean of squares. """
        assert self.second_order
        
        if self.history_upds == 0:
            ret = 0
        else:
            ret = (self.history_w2sum * self.history_upds / self.N_sum)

        if self.alt_sum and self.w2_altsum is not None:
            ret = ret + self.w2_altsum

        return ret + self.w2_sum

    def get_mean(self):
        return self.get_sum() / self.get_wsum()

    def get_meansq(self):
        assert self.second_order

        return self.get_sumsq() / self.get_w2sum()

    def reset_meansq(self):
        assert self.second_order

        self.w2_sum = 0.0
        self.val2_sum[:] = 0
        self.prev_w2s[:] = 0
        self.history_w2sum = 0
        if self.alt_sum:
            self.w2_altsum = None
            self.val2_altsum = None
    
    def set_value(self,key,value,decay=None):
        assert self.allow_decay or decay is None
        
        ind = self.inds.get(key,None)
        if ind is None:
            ind = self.curr_ind
            self.curr_ind += 1
            self.inds[key] = ind
            
        if self.history_upds > 0 and key not in self.history_updated:
            self.history_upds -= 1
            if self.history_upds == 0:
                # This isn't strictly necessary, but it frees memory and ensures consistency
                self.history_mean = None
                self.history_wsum = None
                self.history_updated = None
                if self.second_order:
                    self.history_meansq = None
                    self.history_w2sum = None
            else:
                self.history_updated.add(key)
                    

        prev_val = self.prev_vals.get(ind,None)
        if self.allow_decay:
            prev_w = self.prev_ws[ind]
            if self.second_order:
                prev_w2 = self.prev_w2s[ind]
            else:
                prev_w2 = None
        else:
            prev_w = 1.0
            if self.second_order:
                prev_w2 = 1.0
            else:
                prev_w2 = None
            
        
        # If we've seen this instance before, subtract it from the sum
        if prev_val is not None:
            self.val_sum -= prev_w*prev_val
            self.w_sum -= prev_w
            if self.second_order:
                self.val2_sum -= prev_w2*prev_val**2
                self.w2_sum -= prev_w2

        # If we have an exponential decay, apply it to whats left
        if decay is not None and 0 < decay and decay < 1:
            if self.history_upds > 0: self.history_wsum *= decay
            self.w_sum *= decay
            self.val_sum *= decay
            self.prev_ws *= decay # This technically makes the step O(data set)...
            if self.second_order:
                if self.history_upds > 0: self.history_w2sum *= decay
                self.val2_sum *= decay
                self.w2_sum *= decay
                self.prev_w2s *= decay # This technically makes the step O(data set)...

        # Now add the current value 
        self.val_sum += value
        self.w_sum += 1.0
        if self.second_order:
            self.val2_sum += value**2
            self.w2_sum += 1.0
            if self.allow_decay:
                self.prev_w2s[ind] = 1.0

        self.prev_vals[ind] = value
        if self.allow_decay:
            self.prev_ws[ind] = 1.0
        
        return prev_val, (prev_w,prev_w2) 

