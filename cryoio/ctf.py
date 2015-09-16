from util import memoize
import geom
import numpy as n
import exceptions

def envelope_function(freq_radius,bfactor):
    logenv = -(bfactor/4.0)*freq_radius**2
    envelope = n.exp(logenv)
    return envelope


def compute_ctf(freqs,rots,akv,cs,wgh,dfmid1f,dfmid2f,angastf,dscale,bfactor=None):
    """ Evaluate the CTF at a set of frequences, rotated by a certain amount """ 
    av = akv * 1e3 # Convert kilovots to volts
    cs = cs * 1e7 # Convert spherical aberation from mm to A
    
    # wavelength of electrons
    elambda = 12.2643247 / n.sqrt(av + av**2 * 0.978466e-6)
    
    wgh1 = dscale*n.sqrt(1.0 - wgh**2)
    wgh2 = dscale*wgh

    ix = freqs[:,0]
    iy = freqs[:,1]
    freq_radius = n.sqrt(ix**2 + iy**2)

    angle = elambda*freq_radius
    angspt = n.arctan2(iy,ix)
    if rots is not None:
        angspt = n.mod(angspt.reshape((-1,1)) + rots.reshape((1,-1)),2.0*n.pi)
        angle = angle.reshape((-1,1)) 
    c1 = 2.0*n.pi*angle**2/(2.0*elambda)
    c2 = -c1*cs*angle**2/2.0
    angdif = angspt - angastf
    ccos = n.cos(2.0*angdif)
    df = 0.5*(dfmid1f + dfmid2f + ccos*(dfmid1f-dfmid2f))
    chi = c1*df + c2

    ctf = -wgh1*n.sin(chi) - wgh2*n.cos(chi)
    
    if bfactor is not None:
        ctf *= envelope_function(freq_radius, bfactor)

    return n.require(ctf,dtype = freqs.dtype)

@memoize
def compute_full_ctf(rots,N,psize,akv,csf,wgh,dfmid1,dfmid2,angastf,dscale,bfactor):
    freqs = geom.gencoords(N,2)/(N*psize)
    return compute_ctf(freqs,rots,akv,csf,wgh,dfmid1,dfmid2,angastf,dscale,bfactor)


class CTFBase:
    def compute(self,freqs,rots=None):
        raise exceptions.NotImplementedError()
    
class ParametricCTF(CTFBase):
    def __init__(self,params):
        self.params = params
    
    def compute(self,freqs,rots=None):
        p = self.params
        ctfx = compute_ctf(freqs, rots,
                           p['akv'], p['cs'], p['wgh'],
                           p['df1'], p['df2'], p['angast'],
                           p['dscale'],
                           p.get('bfactor',None))
        
        return ctfx

    def dense_ctf(self, N, psize, bfactor=None):
        p = self.params
        bfactor = bfactor if bfactor is not None else p.get('bfactor',None)
        ctfx = compute_full_ctf(None,N,psize,
                           p['akv'], p['cs'], p['wgh'],
                           p['df1'], p['df2'], p['angast'],
                           p['dscale'],
                           bfactor)
        
        return ctfx
        

if __name__ == '__main__':
    import cryoops as coops
    
    fcoords = n.random.randn(10,2)
    rots = n.array([n.pi/3.0])
    R = n.array([[n.cos(rots), -n.sin(rots)],[n.sin(rots), n.cos(rots)]]).reshape((2,2))
    rotfcoords = n.dot(fcoords,R.T)
    akv = 200
    wgh=0.07
    cs=2.0
    df1, df2, angast = 44722,49349,45.0*(n.pi/180.0)
    dscale = 1.0
    v1 = compute_ctf(fcoords,rots,akv,cs,wgh,df1,df2,angast,dscale).reshape((-1,))
    v2 = compute_ctf(rotfcoords,None,akv,cs,wgh,df1,df2,angast,dscale).reshape((-1,))

    # This being small confirms that using the rots parameter is equivalent to rotating the coordinates
    print n.abs(v1-v2).max()
    

    N = 512
    psz = 5.6
    rad = 0.25
    fcoords = geom.gencoords(N, 2, rad) / (N*psz)
    ctf1_rot = compute_full_ctf(rots,N,psz,akv,cs,wgh,df1,df2,angast,dscale,None)
    ctf2_full = compute_full_ctf(None,N,psz,akv,cs,wgh,df1,df2,angast,dscale,None)

    P_rot = coops.compute_inplanerot_matrix(rots,N,'lanczos',10,rad)
    ctf2_rot = P_rot.dot(ctf2_full).reshape((-1,))

    P_null = coops.compute_inplanerot_matrix(n.array([0]),N,'linear',2,rad)
    ctf1_rot = P_null.dot(ctf1_rot).reshape((-1,))
    
    roterr = ctf1_rot - ctf2_rot
    relerr = n.abs(roterr)/n.maximum(n.abs(ctf1_rot),n.abs(ctf2_rot))
    
    # This being small confirms that compute_inplane_rotmatrix and rots use the same rotation convention 
    print relerr.max(), relerr.mean()
    
    
        
