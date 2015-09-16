import numpy as n
from copy import copy
from . import compute_max_angle

def get_icosphere():
    t = (1.0 + n.sqrt(5.0))/2.0
    tnorm = 1.0/n.sqrt(1 + t*t)

    vertices = []
    vertices.append(tnorm*n.array((-1, t, 0)))
    vertices.append(tnorm*n.array(( 1, t, 0)))
    vertices.append(tnorm*n.array((-1,-t, 0)))
    vertices.append(tnorm*n.array(( 1,-t, 0)))

    vertices.append(tnorm*n.array(( 0,-1, t)))
    vertices.append(tnorm*n.array(( 0, 1, t)))
    vertices.append(tnorm*n.array(( 0,-1,-t)))
    vertices.append(tnorm*n.array(( 0, 1,-t)))

    vertices.append(tnorm*n.array(( t, 0,-1)))
    vertices.append(tnorm*n.array(( t, 0, 1)))
    vertices.append(tnorm*n.array((-t, 0,-1)))
    vertices.append(tnorm*n.array((-t, 0, 1)))

    triangles = []
    triangles.append((0,11,5))
    triangles.append((0,5,1))
    triangles.append((0,1,7))
    triangles.append((0,7,10))
    triangles.append((0,10,11))

    triangles.append((1,5,9))
    triangles.append((5,11,4))
    triangles.append((11,10,2))
    triangles.append((10,7,6))
    triangles.append((7,1,8))

    triangles.append((3,9,4))
    triangles.append((3,4,2))
    triangles.append((3,2,6))
    triangles.append((3,6,8))
    triangles.append((3,8,9))

    triangles.append((4,9,5))
    triangles.append((2,4,11))
    triangles.append((6,2,10))
    triangles.append((8,6,7))
    triangles.append((9,8,1))

    return vertices, triangles

def get_split_vertex_ind(splitcache,vertices,v1,v2):
    key = (min(v1,v2),max(v1,v2))
    if key not in splitcache:
        splitcache[key] = len(vertices)
        newpt = 0.5*(vertices[v1]+vertices[v2])
        newpt /= n.linalg.norm(newpt)
        vertices.append(newpt)
    return splitcache[key]

def refine_icosphere(vertices,triangles):
    splitcache = {}
    newvertices = copy(vertices)
    newtriangles = []

    for (v1,v2,v3) in triangles:
        a = get_split_vertex_ind(splitcache,newvertices,v1,v2)
        b = get_split_vertex_ind(splitcache,newvertices,v2,v3)
        c = get_split_vertex_ind(splitcache,newvertices,v3,v1)

        newtriangles.append((v1, a, c))
        newtriangles.append((v2, b, a))
        newtriangles.append((v3, c, b))
        newtriangles.append(( a, b, c))

    return newvertices, newtriangles

class IcosphereQuadrature:
    @staticmethod
    def get_quad_scheme(depth = None, degree = None):
        # only one of depth or degree should be specified 
        assert bool(depth == None) != bool(degree == None)

        if not hasattr(IcosphereQuadrature,'data'):
            verts,tris = get_icosphere()
            cosang = verts[tris[0][0]].dot(verts[tris[0][1]])
            angle = n.arccos(n.clip(cosang,-1,1))
            IcosphereQuadrature.data = [ (verts,tris,angle) ]
            IcosphereQuadrature.degrees = { len(verts):IcosphereQuadrature.data[-1] }

        while (depth != None and depth >= len(IcosphereQuadrature.data)) or \
              (degree != None and degree > len(IcosphereQuadrature.data[-1][0])):
            verts,tris = refine_icosphere(IcosphereQuadrature.data[-1][0],
                                          IcosphereQuadrature.data[-1][1])
            cosang = verts[tris[0][0]].dot(verts[tris[0][1]])
            angle = n.arccos(n.clip(cosang,-1,1))
            IcosphereQuadrature.data.append( (verts,tris,angle) )
            IcosphereQuadrature.degrees[len(verts)] = IcosphereQuadrature.data[-1]

        if degree != None:
            return IcosphereQuadrature.degrees[degree]
        else:
            return IcosphereQuadrature.data[depth]

    @staticmethod
    def compute_degree(N,rad,usFactor):
        ang = compute_max_angle(N,rad,usFactor)
        return IcosphereQuadrature.get_degree(ang)

    @staticmethod
    def get_degree(maxAngle):
        depth = 0
        while True:
            verts,_,cmaxAng = IcosphereQuadrature.get_quad_scheme(depth)
            if cmaxAng < maxAngle:
                break
            depth += 1

        degree = len(verts)
        return degree, cmaxAng

    @staticmethod
    def get_quad_points(degree,sym = None):
        verts,_,_ = IcosphereQuadrature.get_quad_scheme(degree=degree)
        p = n.array(verts)
        w = (4.0*n.pi/len(verts)) * n.ones(len(verts))

        if sym is None:
            return p, w
        else:
            validPs = sym.in_asymunit(p)
            return p[validPs], w[validPs] * (n.sum(w) / n.sum(w[validPs]))

