"""Experimental bounded-token candidate graphs; full features/hits unchanged."""
import ctypes,math
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import parameter as p
from model.cooccurrence_graph import _build_hit_matrix_H,build_cooccurrence_graph
_lib=None

def rerank(a,h,indptr,indices,k):
    global _lib
    ip=ctypes.POINTER(ctypes.c_int64)
    if _lib is None:
        _lib=ctypes.CDLL(str(Path(__file__).resolve().parents[1]/'experiment/light_graph/rerank.so'))
        _lib.rank_edges.argtypes=[ctypes.c_int64,ctypes.c_int64]+[ip]*6+[ctypes.c_float]*2+[ip,ip,ctypes.POINTER(ctypes.c_float)]
        _lib.rank_edges.restype=ctypes.c_int64
    arrays=[np.ascontiguousarray(x,dtype=np.int64) for x in (a.indptr,a.indices,h.indptr,h.indices,indptr,indices)]
    src=np.empty(a.shape[0]*k,dtype=np.int64);dst=np.empty_like(src);weights=np.empty(src.size,dtype=np.float32)
    ptr=lambda x:x.ctypes.data_as(ip)
    count=_lib.rank_edges(a.shape[0],k,*(ptr(x) for x in arrays),p.w_s,p.w_q,ptr(src),ptr(dst),weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return np.vstack((src[:count],dst[:count])),weights[:count]

def build(a,q,k,hits=None,policy='bounded8'):
    a=a.tocsr();n,d=a.shape;h=hits if hits is not None else _build_hit_matrix_H(a,q)
    if h is None:h=csr_matrix((n,0),dtype=np.float32)
    cutoff=4096 if policy.startswith('adaptive') else 32
    if n<cutoff:return build_cooccurrence_graph(a,q,k,object_query_hits=h),{'fallback_small':1}
    policy={'adaptive16':'bounded16','adaptive8_half':'bounded8_half','adaptive_near':'near'}.get(policy,policy)
    near=policy=='near';conservative=policy=='bounded16' or near
    tokens=32 if near else 16 if conservative else 8
    cap=2048 if near else 256 if conservative else 128;window=32 if near else 8 if conservative else 4
    pool_factor=32 if near else 8 if conservative else 4
    df=np.bincount(a.indices,minlength=d)
    parts=[];ptr=[0]
    for i in range(n):
        cols=a.indices[a.indptr[i]:a.indptr[i+1]];cols=cols[df[cols]<=cap]
        if len(cols)>tokens:cols=cols[np.lexsort((cols,df[cols]))[:tokens]]
        parts.append(cols);ptr.append(ptr[-1]+len(cols))
    ix=np.concatenate(parts) if parts else np.empty(0,dtype=np.int32)
    x=csr_matrix((np.ones(ix.size,dtype=np.float32),ix,np.asarray(ptr)),shape=(n,d));x.sort_indices()
    def posting_ring(csc,columns,width):
        rs=[];cs=[]
        for j in columns:
            ids=csc.indices[csc.indptr[j]:csc.indptr[j+1]]
            if len(ids)<2:continue
            if len(ids)<=2*width+1:
                rr=np.repeat(ids,len(ids));cc=np.tile(ids,len(ids));valid=rr!=cc
                rs.append(rr[valid]);cs.append(cc[valid])
            else:
                for offset in range(1,width+1):
                    rs.append(ids);cs.append(np.roll(ids,offset));rs.append(ids);cs.append(np.roll(ids,-offset))
        if not rs:return csr_matrix((n,n),dtype=np.float32)
        ri=np.concatenate(rs);ci=np.concatenate(cs)
        result=csr_matrix((np.ones(ri.size,dtype=np.float32),(ri,ci)),shape=(n,n));result.sum_duplicates()
        return result
    # Long keyword postings and queries get bounded ring neighborhoods rather
    # than all pairs. Keep keyword/query channels separate until ranking.
    hc=h.tocsc()
    if near:
        # Preserve every workload-sharing candidate. This channel is often much
        # sparser than the keyword graph and is central to BBC's workload bias.
        query_transpose=h.astype(np.float32).T.tocsr();ring=None
    else:ring=posting_ring(hc,np.flatnonzero(np.diff(hc.indptr)>1),window)
    keyword_ring=posting_ring(a.tocsc(),np.flatnonzero(df>cap),16 if near else 2 if not conservative else 4)
    xt=x.T.tocsr();counts=np.diff(a.indptr).astype(np.float32);hcounts=np.diff(h.indptr).astype(np.float32)
    candidates=[];cp=[0];raw=0;empty=0
    for start in range(0,n,128):
        stop=min(start+128,n);overlap=(x[start:stop]@xt+keyword_ring[start:stop]).tocsr();overlap.sort_indices()
        query_overlap=(h[start:stop].astype(np.float32)@query_transpose).tocsr() if near else None
        if near:query_overlap.sort_indices()
        for local,i in enumerate(range(start,stop)):
            lo,hi=overlap.indptr[local:local+2];cols=overlap.indices[lo:hi];inter=overlap.data[lo:hi]
            query_source=query_overlap if near else ring;query_row=local if near else i
            rlo,rhi=query_source.indptr[query_row:query_row+2];rc=query_source.indices[rlo:rhi];rv=query_source.data[rlo:rhi]
            allcols=np.union1d(cols,rc);score=np.zeros(len(allcols),dtype=np.float32)
            if len(cols):score[np.searchsorted(allcols,cols)]=p.w_s*inter/np.maximum(1,counts[i]+counts[cols]-inter)
            if len(rc):
                # Ring multiplicity is a ranking hint, not an exact overlap.
                denominator=hcounts[i]+hcounts[rc]-rv if near else np.maximum(hcounts[i],hcounts[rc])
                score[np.searchsorted(allcols,rc)]+=p.w_q*np.minimum(rv,np.minimum(hcounts[i],hcounts[rc]))/np.maximum(1,denominator)
            valid=allcols!=i;allcols=allcols[valid];score=score[valid];raw+=len(allcols)
            limit=pool_factor*k
            if len(allcols)>limit:
                # Same deterministic score/ID cutoff as a complete lexsort,
                # without sorting thousands of discarded candidates per row.
                chosen=np.argpartition(score,-limit)[-limit:];threshold=score[chosen].min()
                above=np.flatnonzero(score>threshold);ties=np.flatnonzero(score==threshold)
                chosen=np.r_[above,ties[:limit-len(above)]];allcols=allcols[chosen]
            allcols=np.sort(allcols);empty+=len(allcols)==0;candidates.append(allcols);cp.append(cp[-1]+len(allcols))
    ci=np.concatenate(candidates) if candidates else np.empty(0,dtype=np.int64)
    target_k=max(1,k//2) if policy=='bounded8_half' else k
    edges,weights=rerank(a,h,np.asarray(cp),ci,target_k)
    return dict(X=a,H=h,edge_index=edges,edge_weight=weights),dict(tokens=tokens,posting_cap=cap,query_window=window,
        raw_candidates=raw,verified_pairs=len(ci),empty_candidate_rows=empty,edges=len(weights),target_k=target_k,fallback_small=0)
