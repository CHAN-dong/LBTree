"""Exact binary graph scoring fused natively; original NumPy top-K/ties."""
import ctypes
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import parameter as p
from model.cooccurrence_graph import _build_hit_matrix_H,build_cooccurrence_graph
_libraries={}

def build(a,q,k,hits=None,engine='sort'):
    if engine not in ('sort','bitset'):raise ValueError('Unknown exact graph engine')
    if k<=0:raise ValueError('K must be positive')
    a=a.tocsr();h=hits if hits is not None else _build_hit_matrix_H(a,q);original_h=h
    if h is None:h=csr_matrix((a.shape[0],0),dtype=np.float32)
    else:h=h.tocsr()
    if (not a.has_canonical_format or not h.has_canonical_format or np.any(a.data!=1) or np.any(h.data!=1)
        or a.dtype not in (np.float32,np.float64) or h.dtype not in (np.float32,np.float64,np.int8)):
        return build_cooccurrence_graph(a,q,k,object_query_hits=original_h)
    ip=ctypes.POINTER(ctypes.c_int64);fp=ctypes.POINTER(ctypes.c_float)
    if engine not in _libraries:
        library=Path(__file__).resolve().parents[1]/'experiment/light_graph'/('fused_bitset.so' if engine=='bitset' else 'fused.so')
        try:_libraries[engine]=ctypes.CDLL(str(library))
        except OSError:return build_cooccurrence_graph(a,q,k,object_query_hits=original_h)
        _libraries[engine].score_block.argtypes=[ctypes.c_int64]*3+[ip]*8+[ctypes.c_int,ctypes.c_float,ctypes.c_float,ip,ip,fp]
        _libraries[engine].score_block.restype=ctypes.c_int64
    _lib=_libraries[engine]
    # Empty global columns do not affect intersections or row lengths. Avoid
    # allocating a 604k-column posting header for a two-object sample.
    compact=a[:,np.unique(a.indices)] if a.shape[1]>4*max(1,a.nnz) else a
    compact_h=h[:,np.unique(h.indices)] if h.shape[1]>4*max(1,h.nnz) else h
    ac=compact.tocsc();hc=compact_h.tocsc();n=a.shape[0]
    arrays=[np.ascontiguousarray(x,dtype=np.int64) for x in (compact.indptr,compact.indices,ac.indptr,ac.indices,compact_h.indptr,compact_h.indices,hc.indptr,hc.indices)]
    ptr=lambda x:x.ctypes.data_as(ip)
    block=32;cp=np.empty(block+1,dtype=np.int64);ci=np.empty(max(1,block*n),dtype=np.int64);cv=np.empty(ci.size,dtype=np.float32)
    edges=[];weights=[]
    for start in range(0,n,block):
        stop=min(start+block,n)
        _lib.score_block(n,start,stop,*(ptr(x) for x in arrays),int(h.dtype==np.int8),p.w_s,p.w_q,ptr(cp),ptr(ci),cv.ctypes.data_as(fp))
        for local,row in enumerate(range(start,stop)):
            cols=ci[cp[local]:cp[local+1]];values=cv[cp[local]:cp[local+1]]
            if len(cols)>k:
                chosen=np.argpartition(values,-k)[-k:];cols=cols[chosen];values=values[chosen]
                order=np.argsort(-values);cols=cols[order];values=values[order]
            if len(cols):
                edges.append(np.vstack((np.full(len(cols),row,dtype=np.int64),cols)));weights.append(values.copy())
    return dict(X=a,H=original_h,edge_index=np.concatenate(edges,axis=1) if edges else np.empty((2,0),dtype=np.int64),
        edge_weight=np.concatenate(weights) if weights else np.empty(0,dtype=np.float32))
