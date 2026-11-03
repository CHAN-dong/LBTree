"""Optional native S2 backend. Build via CMake; Python remains the fallback."""
import ctypes
import os
from pathlib import Path
import numpy as np

_library=None
_attempted=False
calls=0

def _load():
    global _library,_attempted
    if os.environ.get('LBTREE_DISABLE_NATIVE')=='1': return None
    if _attempted: return _library
    _attempted=True
    root=Path(__file__).resolve().parents[1]/'build-local'
    candidates=[root/'liblbtree_stream.so',root/'liblbtree_stream.dylib',
                root/'lbtree_stream.dll',root/'Release/lbtree_stream.dll']
    if os.environ.get('LBTREE_NATIVE_LIBRARY'):
        candidates.insert(0,Path(os.environ['LBTREE_NATIVE_LIBRARY']))
    for path in candidates:
        if path.exists():
            try:
                lib=ctypes.CDLL(str(path))
                ptr=ctypes.POINTER(ctypes.c_int64)
                lib.lbtree_stream.argtypes=[ctypes.c_int64]*3+[ptr]*7+[ctypes.c_int64,ptr,ctypes.c_int64,
                    ptr,ctypes.c_int64,ptr,ctypes.c_int64]+[ctypes.c_double]*7+[ctypes.c_void_p]*3
                lib.lbtree_stream.restype=ctypes.c_int
                _library=lib
                break
            except (OSError,AttributeError): pass
    return _library

def assign(objects,queries,hits,sl,sr,ql,qr,epsilon,alpha,beta,gamma,ws,wq,balance):
    global calls
    lib=_load()
    if lib is None or hits is None or not len(sl) or not len(sr): return None
    n,v=objects.shape;m=queries.shape[0]
    if hits.shape != (n,m): return None
    object_indices, query_indices = objects.indices, queries.indices
    # Preserve keyword order and all query-only terms. Small nodes must not
    # allocate/initialize 604k inverted-list headers for a few hundred terms.
    if v > 4096 and objects.nnz + queries.nnz < v // 2:
        keywords, inverse = np.unique(np.concatenate((object_indices, query_indices)),
                                      return_inverse=True)
        object_indices = inverse[:objects.nnz]
        query_indices = inverse[objects.nnz:]
        v = keywords.size
    # Dense keyword flags/inverted-list headers are inappropriate for enormous
    # vocabularies. Keep the original sparse Python path in that case.
    if 32*v+64*m+16*queries.nnz > 128*1024*1024: return None
    seeds=np.concatenate((sl,sr))
    if len(np.unique(seeds))!=len(seeds) or np.any(seeds<0) or np.any(seeds>=n): return None
    if np.any(ql<0) or np.any(ql>=m) or np.any(qr<0) or np.any(qr>=m): return None
    arrays=[np.ascontiguousarray(x,dtype=np.int64) for x in
            [objects.indptr,object_indices,queries.indptr,query_indices,hits.indptr,hits.indices,sl,sr,ql,qr]]
    ptr=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    op,oi,qp,qi,hp,hi,l,r,lq,rq=arrays
    labels=np.empty(n,dtype=np.int8);al=np.empty(m,dtype=np.uint8);ar=np.empty(m,dtype=np.uint8)
    code=lib.lbtree_stream(n,v,m,ptr(op),ptr(oi),ptr(qp),ptr(qi),ptr(hp),ptr(hi),
        ptr(l),len(l),ptr(r),len(r),ptr(lq),len(lq),ptr(rq),len(rq),
        epsilon,alpha,beta,gamma,ws,wq,balance,labels.ctypes.data,al.ctypes.data,ar.ctypes.data)
    if code: return None
    calls += 1
    remaining=np.ones(n,dtype=bool);remaining[seeds]=False
    return (np.concatenate((sl,np.flatnonzero((labels==0)&remaining))),
            np.concatenate((sr,np.flatnonzero((labels==1)&remaining))),np.flatnonzero(al),np.flatnonzero(ar))

def available(): return _load() is not None
