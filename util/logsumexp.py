import numpy as n

def logsumexp(a, axis=None, b=None, dtype=None):
    a = n.asarray(a)
    if (axis is None and a.size == 1) or (axis is not None and a.shape[axis] == 1):
        if b is not None:
            return a + n.log(b)
        else:
            return a
    if axis is None:
        a = a.ravel()
    else:
        a = n.rollaxis(a, axis)
    a_max = a.max(axis=0)
    if b is not None:
        b = n.asarray(b)
        if axis is None:
            b = b.ravel()
        else:
            b = n.rollaxis(b, axis)
        out = n.log(n.sum(b * n.exp(a - a_max), axis=0, dtype=dtype))
    else:
        out = n.log(n.sum(n.exp(a - a_max), axis=0, dtype=dtype))
    out += a_max
    return out
