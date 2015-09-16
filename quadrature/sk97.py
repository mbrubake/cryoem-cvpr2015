import numpy as n
from . import compute_max_angle

def gensk97(N):
    # See http://dx.doi.org/10.1016/j.jsb.2006.06.002 and references therein
    h = -1.0 + (2.0/(N-1))*n.arange(0,N)
    theta = n.arccos(h)
    phi_base = n.zeros_like(theta)
    phi_base[1:(N-1)] = ((3.6/n.sqrt(N))/n.sqrt(1 - h[1:(N-1)]**2))
    phi = n.cumsum(phi_base)
    phi[0] = 0
    phi[N-1] = 0

    stheta = n.sin(theta)
    dirs = n.vstack([n.cos(phi)*stheta, n.sin(phi)*stheta, n.cos(theta)]).T

    return dirs

class SK97Quadrature:
    @staticmethod
    def compute_degree(N,rad,usFactor):
        ang = compute_max_angle(N,rad,usFactor)
        return SK97Quadrature.get_degree(ang)

    @staticmethod
    def get_degree(maxAngle):
        degree = n.ceil((3.6/maxAngle)**2)
        cmaxAng = 3.6/n.sqrt(degree)

        return degree, cmaxAng

    @staticmethod
    def get_quad_points(degree,sym = None):
        verts = gensk97(degree)
        p = n.array(verts)
        w = (4.0*n.pi/len(verts)) * n.ones(len(verts))

        if sym is None:
            return p, w
        else:
            validPs = sym.in_asymunit(p)
            return p[validPs], w[validPs] * (n.sum(w) / n.sum(w[validPs]))

