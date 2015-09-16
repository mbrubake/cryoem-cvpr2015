#!/usr/bin/env python

import sys,os,inspect
# This file is run from a subdirectory of the package.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),os.pardir))

import time
from cryoio import mrc
from cryoio.ctfstack import CTFStack, GeneratedCTFStack
import cryoem, density, cryoops, geom
from util import format_timedelta

import cPickle

import numpy as n
import pyximport; pyximport.install(setup_args={"include_dirs":n.get_include()},reload_support=True)
import sincint


def genphantomdata (N_D, phantompath, ctfparfile):
    mscope_params = {'akv':200,'wgh':0.07,'cs':2.0,'psize':2.8,'bfactor':500.0}
    N = 128
    rad = 0.95
    shift_sigma = 3.0
    sigma_noise = 25.0
    M_totalmass = 80000
    kernel = 'lanczos'
    ksize = 6

    premult = cryoops.compute_premultiplier(N, kernel, ksize) 
    
    tic = time.time()

    N_D = int(N_D)
    N = int(N)
    rad = float(rad)
    psize = mscope_params['psize']
    bfactor = mscope_params['bfactor']
    shift_sigma = float(shift_sigma)
    sigma_noise = float(sigma_noise)
    M_totalmass = float(M_totalmass)

    srcctf_stack = CTFStack(ctfparfile,mscope_params)
    genctf_stack = GeneratedCTFStack(mscope_params,parfields=['PHI','THETA','PSI','SHX','SHY'])

    TtoF = sincint.gentrunctofull(rad=rad)
    Cmap = n.sort(n.random.random_integers(0,srcctf_stack.get_num_ctfs()-1,N_D))

    M = mrc.readMRC(phantompath)
    cryoem.window(M, 'circle')
    M[M<0] = 0
    if M_totalmass is not None:
        M *= M_totalmass/M.sum()
        
    V = density.real_to_fspace(premult.reshape((1,1,-1)) * premult.reshape((1,-1,1)) * premult.reshape((-1,1,1)) * M)

    print "Generating data..." ; sys.stdout.flush()
    imgdata = n.empty( (N_D, N, N), dtype=density.real_t )
    
    pardata = {'R':[],'t':[]}

    prevctfI = None
    for i,srcctfI in enumerate(Cmap):
        ellapse_time = time.time() - tic
        remain_time = float(N_D - i)*ellapse_time/max(i,1)
        print "\r%.2f Percent.. (Elapsed: %s, Remaining: %s)      " % (i/float(N_D)*100.0,format_timedelta(ellapse_time),format_timedelta(remain_time)),
        sys.stdout.flush()
        
        # Get the CTF for this image
        cCTF = srcctf_stack.get_ctf(srcctfI)
        if prevctfI != srcctfI:
            genctfI = genctf_stack.add_ctf(cCTF)
            C = cCTF.dense_ctf(N,psize,bfactor).reshape((N**2,))
            prevctfI = srcctfI 
        
        # Randomly generate the viewing direction/shift
        pt = n.random.randn(3)
        pt /= n.linalg.norm(pt)
        psi = 2*n.pi*n.random.rand()
        EA = geom.genEA(pt)[0]
        EA[2] = psi
        shift = n.random.randn(2) * shift_sigma

        R = geom.rotmat3D_EA(*EA)[:,0:2]
        slop = cryoops.compute_projection_matrix([R], N, kernel, ksize, rad, 'rots')
        S = cryoops.compute_shift_phases(shift.reshape((1,2)), N, rad)[0]

        D = slop.dot( V.reshape((-1,)) )
        D *= S

        imgdata[i] = density.fspace_to_real((C*TtoF.dot(D)).reshape((N,N))) + n.require(n.random.randn(N, N)*sigma_noise,dtype=density.real_t)

        genctf_stack.add_img(genctfI,
                             PHI=EA[0]*180.0/n.pi,THETA=EA[1]*180.0/n.pi,PSI=EA[2]*180.0/n.pi,
                             SHX=shift[0],SHY=shift[1])

        pardata['R'].append(R)
        pardata['t'].append(shift)
        
    print "\rDone in ", time.time()-tic, " seconds."
    return imgdata, genctf_stack, pardata, mscope_params


if __name__=='__main__':
    if len(sys.argv) != 5:
        print """Wrong Number of Arguments. Usage:
%run genphantomdata num inputmrc inputpar output
%run genphantomdata 5000 finalphantom.mrc Data/thermus/thermus_Nature2012_128x128.par Data/phantom_5000
""" 
        sys.exit()

    imgdata, ctfstack, pardata, mscope_params = genphantomdata(sys.argv[1], sys.argv[2], sys.argv[3])
    outpath = sys.argv[4]
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    tic = time.time()    
    print "Dumping data...", ; sys.stdout.flush()
    def_path = os.path.join(outpath,'defocus.txt')
    ctfstack.write_defocus_txt(def_path)

    par_path = os.path.join(outpath,'ctf_gt.par')
    ctfstack.write_pardata(par_path)

    mrc_path = os.path.join(outpath,'imgdata.mrc')
    mrc.writeMRC(mrc_path, n.transpose(imgdata,(1,2,0)), mscope_params['psize'])

    pard_path = os.path.join(outpath,'pardata.pkl')
    with open(pard_path,'wb') as fi:
        cPickle.dump(pardata, fi, protocol=2)
    print "Done in ", time.time()-tic, " seconds."

    sys.exit()
    

