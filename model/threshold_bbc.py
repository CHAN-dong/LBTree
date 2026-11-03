"""Exact Eq.(4) sweep within a small family of frozen embedding orderings."""
import ctypes
from pathlib import Path
import numpy as np
import parameter as p
_lib=None

def library():
    global _lib
    if _lib is None:
        path=Path(__file__).resolve().parents[1]/'experiment/training_methods/threshold_sweep.so'
        _lib=ctypes.CDLL(str(path));ip=ctypes.POINTER(ctypes.c_int64)
        _lib.best_cut.argtypes=[ctypes.c_int64]*3+[ip]*5+[ctypes.c_double]*6+[ctypes.POINTER(ctypes.c_double)]
        _lib.best_cut.restype=ctypes.c_int64
    return _lib

def sweep(objects,queries,order):
    if objects.shape[0]<2:raise ValueError('A threshold sweep needs at least two objects')
    active=np.union1d(objects.indices,queries.indices);a=objects[:,active];q=queries[:,active]
    arrays=[np.ascontiguousarray(x,dtype=np.int64) for x in (a.indptr,a.indices,q.indptr,q.indices,order)]
    ptr=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    cost=ctypes.c_double()
    cut=library().best_cut(a.shape[0],a.shape[1],q.shape[0],*(ptr(x) for x in arrays),
        p.w_s*p.W_S,p.w_q*p.W_Q,p.EPSILON,p.ALFA,p.BETA,p.GAMA,ctypes.byref(cost))
    return int(cut),cost.value

def search(objects,queries,embeddings,logits):
    if objects.shape[0]<2:return np.zeros(objects.shape[0],dtype=np.int64)
    scores=[np.asarray(logits).reshape(-1)]
    dims=np.argsort(np.var(embeddings,axis=0))[-3:]
    scores.extend(embeddings[:,i] for i in dims)
    best=float('inf');labels=None
    for score in scores:
        order=np.argsort(score,kind='stable');cut,cost=sweep(objects,queries,order)
        if cost<best:
            best=cost;labels=np.ones(objects.shape[0],dtype=np.int64);labels[order[:cut]]=0
    return labels
