import numpy as n
from util import memoize

def rotmat2D (theta):
    return n.array([[n.cos(theta), -n.sin(theta)],
                   [n.sin(theta),  n.cos(theta)]])

def rotmat3D_to_quat(R):
    qw = n.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2.0
    qw4 = 4.0*qw
    qx = (R[2,1] - R[1,2])/qw4
    qy = (R[0,2] - R[2,0])/qw4
    qz = (R[1,0] - R[0,1])/qw4
    
    qs = n.array([qw,qx,qy,qz])
    return qs

def rotmat3D_quat(q):
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    
    qx2 = qx**2
    qy2 = qy**2
    qz2 = qz**2

    qxqw = qx*qw
    qxqy = qx*qy
    qxqz = qx*qz
    qyqw = qy*qw
    qyqz = qy*qz
    qzqw = qz*qw

    Rs = n.array([[ 1 - 2*qy2 - 2*qz2,      2*qxqy - 2*qzqw,      2*qxqz + 2*qyqw], \
                  [   2*qxqy + 2*qzqw,    1 - 2*qx2 - 2*qz2,      2*qyqz - 2*qxqw], \
                  [   2*qxqz - 2*qyqw,      2*qyqz + 2*qxqw,    1 - 2*qx2 - 2*qy2]])
    
    return Rs

def rotmat3D_EA (phi, theta, psi=None):
    """
    Generates a rotation matrix from Z-Y-Z Euler angles. This rotation matrix
    maps from image coordinates (x,y,0) to view coordinates and should be
    consistent with JLRs code.
    """
    R_z  = n.array([[ n.cos(phi), -n.sin(phi),  0],
                    [ n.sin(phi),  n.cos(phi),  0],
                    [          0,           0,  1]])
    R_y  = n.array([[ n.cos(theta),  0,  n.sin(theta)],
                    [0,              1,             0],
                    [-n.sin(theta),  0,  n.cos(theta)]])
    R = n.dot(R_z, R_y)
    if psi is not None and psi != 0:
        R_in = n.array([[ n.cos(psi), -n.sin(psi),  0],
                        [ n.sin(psi),  n.cos(psi),  0],
                        [          0,           0,  1]])
    
        R = n.dot(R, R_in);

    return R
        

def rotmat3D_dir(projdir, psi=None):
    d = projdir.reshape((3,))/n.linalg.norm(projdir)
    vdir = n.array([0,0,1],dtype=projdir.dtype)
    rotax = n.cross(vdir,d)
    rotaxnorm = n.linalg.norm(rotax)
    if rotaxnorm > 1e-16:
        rang = n.arctan2(rotaxnorm,d[2])
        rotax /= rotaxnorm
        x,y,z = rotax[0],rotax[1],rotax[2]
        c, s = n.cos(rang), n.sin(rang)
        C = 1 - c
        
        R = n.array([[ x*x*C + c  , x*y*C - z*s, x*z*C + y*s ], 
                     [ y*x*C + z*s, y*y*C + c  , y*z*C - x*s ],
                     [ z*x*C - y*s, z*y*C + x*s, z*z*C + c   ]],
                    dtype=projdir.dtype)
    else:
        R = n.identity(3, dtype=projdir.dtype)
        if d[2] < 0:
            R[1,1] = -1
            R[2,2] = -1

    if psi is not None and psi != 0:
        R_in = n.array([[ n.cos(psi), -n.sin(psi),  0],
                        [ n.sin(psi),  n.cos(psi),  0],
                        [          0,           0,  1]])
    
        R = n.dot(R, R_in);

    return R

def rotmat3D_expmap(e):
    theta = n.linalg.norm(e)
    if theta < 1e-16:
        return n.identity(3, dtype=e.dtype)
    k = e/theta
    K = n.array([[    0,-k[2], k[1]],\
                 [ k[2],    0,-k[0]],\
                 [-k[1], k[0],   0]],dtype=e.dtype)
    return n.identity(3, dtype=e.dtype) + n.sin(theta)*K + (1-n.cos(theta))*n.dot(K,K)

def genDir(EAs):
    """
    Generate the projection direction given the euler angles.  Since the image
    is in the x-y plane, the projection direction is given by R(EA)*z where 
    z = (0,0,1)
    """
    
    p = n.array([rotmat3D_EA(*EA)[:,2] for EA in EAs])
    return p

def genEA(p):
    """
    Generates euler angles from a vector direction
    p is a column vector in the direction that the new x-axis should point
    returns tuple (phi, theta, psi) with psi=0
    """
    assert p.shape[-1] == 3
    p = n.asarray(p).reshape((-1,3))
    theta = n.arctan2(n.linalg.norm(p[:,0:2],axis=1),p[:,2]).reshape((-1,1))
    phi = n.arctan2(p[:,1],p[:,0]).reshape((-1,1))
    
    return n.hstack([phi,theta,n.zeros_like(theta)])


@memoize
def gencoords_base (N,d):
    x = n.arange(-N/2,N/2,dtype=n.float32)
    c = x.copy()
    for i in range(1,d):
        c = n.column_stack([n.repeat(c, N, axis=0), n.tile(x, N**i)])

    return c

@memoize
def gencoords (N,d,rad=None,truncmask=False,trunctype='circ'):
    """ generate coordinates of all points in an NxN..xN grid with d dimensions 
    coords in each dimension are [-N/2, N/2) 
    N should be even"""
    
    if not truncmask:
        _,truncc,_ = gencoords(N,d,rad,True)
        return truncc
    
    c = gencoords_base(N,d)

    if rad is not None:
        if trunctype == 'circ':
            r2 = n.sum(c**2,axis=1)
            trunkmask = r2 < (rad*N/2.0)**2
        elif trunctype == 'square':
            r = n.max(n.abs(c),axis=1)
            trunkmask = r < (rad*N/2.0)
            
        truncc = c[trunkmask,:]
    else:
        trunkmask = n.ones((c.shape[0],),dtype=n.bool8)
        truncc = c
 
    
    return c,truncc,trunkmask
 
