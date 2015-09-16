import numpy as n

class HermiteShiftQuadrature:
    @staticmethod
    def get_degree(N,rad,shiftsigma,shiftextent,usFactor):
        return 1+2*n.round(shiftextent*rad/usFactor)

    @staticmethod
    def get_quad_points(degree,shiftsigma,shiftextent,trunctype):
        assert trunctype in ['none','circ']
        if degree == 0:
            gpoints = []
            gweights = []
        elif degree == 1:
            gpoints = [0]
            gweights = [1]
        else:
            # Assumes a N(0,shiftsigma^2) distribution over shifts
            # truncated to [-shiftextent,shiftextent]
            assert n.isfinite(shiftsigma)
            gpoints, gweights = n.polynomial.hermite.hermgauss(degree)
            gpoints *= n.sqrt(2.0)*shiftsigma
            goodpoints = n.abs(gpoints) < shiftextent
            gpoints = gpoints[goodpoints]
            gweights = gweights[goodpoints]

        K = len(gpoints)**2
        W = n.empty(K, dtype=n.float32)

        i = 0
        pts = n.empty((K,2),dtype=n.float32)
        for sx, wx in zip(gpoints, gweights):
            for sy, wy in zip(gpoints, gweights):
                if trunctype is None or trunctype == 'none' or (trunctype == 'circ' and sx**2 + sy**2 < shiftextent**2):
                    W[i] = wx*wy
                    pts[i,0] = sx
                    pts[i,1] = sy
                    i += 1

        return pts[0:i],W[0:i]/n.sum(W[0:i])


