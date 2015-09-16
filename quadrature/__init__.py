import numpy as n

def compute_max_angle(N,rad,usFactor=1.0):
    r = rad*N/2.0

    # This angle is (roughly) the largest angle step that can be taken
    # and not miss any of the high frequency coefficients at this value
    # of rad.  Using larger steps means rad could be set smaller
    # while smaller steps are likely being wasted.  We can use this to 
    # set the "optimal" values of inplane and degree
    ang = 0.5*n.arccos((r**2 - 1.0)/(r**2 + 1.0))

    return usFactor*ang

from domain import FixedCircleDomain, FixedDirectionalDomain, \
                   FixedPlanarDomain, FixedSphereDomain, FixedSO3Domain

from icosphere import IcosphereQuadrature
from sk97 import SK97Quadrature

from legendre import LegendreShiftQuadrature
from hermite import HermiteShiftQuadrature

quad_schemes = { ('dir','icosphere'):IcosphereQuadrature,
                 ('dir','sk97'):SK97Quadrature,
                 ('shift','legendre'):LegendreShiftQuadrature,
                 ('shift','hermite'):HermiteShiftQuadrature }

