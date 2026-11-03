"""Experimental weighted coarse initialization and exact Eq.(4) refinement.

This is an alternative to neural BBC, not a mathematically equivalent encoder.
All acceptance decisions use the original binary objects and complete workload.
"""
import numpy as np
from scipy.sparse import csr_matrix
import parameter as p


def coarse_partition(objects, seed=42, bits=7):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    n,d=objects.shape
    active=np.unique(objects.indices)
    if n<2 or not active.size:return np.zeros(n,dtype=np.int64),{'supernodes':1}
    x=objects[:,active].astype(np.float32)
    df=np.asarray((x>0).sum(axis=0)).ravel()
    features=normalize(x.multiply(np.log1p(n/(1+df))).tocsr())
    rng=np.random.default_rng(seed)
    projections=rng.standard_normal((active.size,bits),dtype=np.float32)
    signatures=((features@projections>0).astype(np.int64)*(1<<np.arange(bits))).sum(axis=1)
    _,groups=np.unique(signatures,return_inverse=True)
    sizes=np.bincount(groups);k=sizes.size
    if k<2:return np.zeros(n,dtype=np.int64),{'supernodes':k}
    membership=csr_matrix((np.ones(n,dtype=np.float32),(groups,np.arange(n))),shape=(k,n))
    # Weighted centroids preserve total feature sums and object multiplicities.
    prototypes=(membership@features).multiply((1/sizes)[:,None]).tocsr()
    km=KMeans(n_clusters=2,n_init=3,max_iter=40,random_state=seed)
    labels=km.fit_predict(prototypes,sample_weight=sizes)[groups]
    return labels.astype(np.int64),{'supernodes':int(k),'objects':n,'groups':groups}


class ExactPartitionCost:
    """Incremental leaf+two-branch-parent cost, including query rerouting.

    The gamma*true-results term is partition invariant: each true result belongs
    to exactly one child, and keyword routing always reaches that child.
    """
    def __init__(self,objects,queries,labels,total_hits):
        if not objects.has_canonical_format or not queries.has_canonical_format or np.any(objects.data!=1) or np.any(queries.data!=1):
            raise ValueError('Exact refinement requires canonical binary object/query CSR matrices')
        active=np.union1d(objects.indices,queries.indices)
        if active.size<objects.shape[1]:
            objects=objects[:,active];queries=queries[:,active]
        self.a=objects;self.q=queries;self.labels=np.asarray(labels,dtype=np.int64).copy()
        self.n=np.bincount(self.labels,minlength=2).astype(np.int64)
        self.count=np.stack([np.asarray(objects[self.labels==s].sum(axis=0)).ravel().astype(np.int64) for s in (0,1)])
        self.omega=(self.count>0).sum(axis=1).astype(np.int64)
        self.length=np.diff(queries.indptr);self.nonempty=self.length>0
        self.missing=np.stack([np.asarray(queries@(self.count[s]==0).astype(np.int64)).ravel() for s in (0,1)])
        self.routes=(self.missing==0)&self.nonempty
        self.wn=self.routes.sum(axis=1).astype(np.int64)
        self.lens=(self.routes*np.maximum(self.length-1,0)).sum(axis=1).astype(np.int64)
        self.post=queries.tocsc();self.hits=int(total_hits)
        self.parent_omega=np.unique(objects.indices).size

    def cost(self):
        storage=np.dot(self.omega,self.n)+32*self.omega.sum()+32*self.n.sum()
        storage+=34*self.parent_omega+64
        query=p.EPSILON*self.wn.sum()+p.ALFA*np.dot(self.n,self.lens)+p.BETA*np.dot(self.n,self.wn)+p.GAMA*self.hits
        query+=p.EPSILON*self.q.shape[0]+2*p.ALFA*np.maximum(self.length-1,0).sum()+2*p.BETA*self.q.shape[0]+p.GAMA*self.wn.sum()
        return float(p.weighted_cost(storage,query))

    def move(self,i,accept_only=True):
        return self.move_group(np.array([i]),accept_only=accept_only)

    def move_group(self,rows,accept_only=True):
        src=int(self.labels[rows[0]]);dst=1-src;amount=len(rows)
        if self.n[src]<=amount or np.any(self.labels[rows]!=src):return False
        if amount==1:
            i=int(rows[0]);cols=self.a.indices[self.a.indptr[i]:self.a.indptr[i+1]];counts=np.ones(cols.size,dtype=np.int64)
        else:cols,counts=np.unique(self.a[rows].indices,return_counts=True)
        lost=cols[self.count[src,cols]==counts];gained=cols[self.count[dst,cols]==0]
        changed=[]
        before=self.cost()
        for side,terms,delta in ((src,lost,1),(dst,gained,-1)):
            parts=[self.post.indices[self.post.indptr[t]:self.post.indptr[t+1]] for t in terms]
            ids=np.concatenate(parts) if parts else np.empty(0,dtype=np.int64)
            affected=np.unique(ids);old=self.routes[side,affected].copy()
            np.add.at(self.missing[side],ids,delta)
            new=(self.missing[side,affected]==0)&self.nonempty[affected]
            diff=new.astype(np.int64)-old.astype(np.int64)
            self.wn[side]+=diff.sum();self.lens[side]+=np.dot(diff,np.maximum(self.length[affected]-1,0))
            self.routes[side,affected]=new;changed.append((side,ids,affected,old,diff,delta))
        self.n[src]-=amount;self.n[dst]+=amount;self.omega[src]-=len(lost);self.omega[dst]+=len(gained)
        after=self.cost()
        if not accept_only or after<before-1e-14:
            self.count[src,cols]-=counts;self.count[dst,cols]+=counts;self.labels[rows]=dst
            return True
        self.n[src]+=amount;self.n[dst]-=amount;self.omega[src]+=len(lost);self.omega[dst]-=len(gained)
        for side,ids,affected,old,diff,delta in changed:
            np.add.at(self.missing[side],ids,-delta);self.routes[side,affected]=old
            self.wn[side]-=diff.sum();self.lens[side]-=np.dot(diff,np.maximum(self.length[affected]-1,0))
        return False


def refine_partition(objects,queries,labels,total_hits,seed=42,rounds=2,budget=512,groups=None):
    if np.unique(labels).size<2:return np.asarray(labels),{'moves':0}
    state=ExactPartitionCost(objects,queries,labels,total_hits);initial=state.cost();moves=0
    rng=np.random.default_rng(seed)
    if groups is not None:
        members=[np.flatnonzero(groups==g) for g in np.unique(groups)]
        for _ in range(rounds):
            accepted=sum(state.move_group(members[g]) for g in rng.permutation(len(members)))
            moves+=accepted
            if not accepted:break
    for _ in range(rounds):
        accepted=0
        for i in rng.permutation(objects.shape[0])[:budget]:accepted+=state.move(int(i))
        moves+=accepted
        if not accepted:break
    assert state.cost()<=initial+1e-12
    return state.labels,{'moves':moves,'cost_before':initial,'cost_after':state.cost()}


def refinement_groups(objects,labels,seed=42):
    """MinHash cohorts intersected with the current side; no tree fabrication."""
    hashes=(objects.indices.astype(np.uint64)*np.uint64(2654435761)+np.uint64(seed*2246822519))&np.uint64(0xffffffff)
    nonempty=np.diff(objects.indptr)>0;minimum=np.zeros(objects.shape[0],dtype=np.uint64)
    minimum[nonempty]=np.minimum.reduceat(hashes,objects.indptr[:-1][nonempty])
    # A shared minimum keyword yields a shared cohort; collisions are harmless
    # because an entire-cohort move still needs an exact cost improvement.
    return (minimum%128).astype(np.int64)*2+np.asarray(labels,dtype=np.int64)


def result_from_labels(objects,queries,obj_idx,query_idx,labels,parent_cost,context,all_objects,all_queries):
    from model.model_split import SplitResult,node_cost
    from model.query_split import build_kw_lr_bitset,split_queries_by_kw_lr_bitset
    if np.unique(labels).size<2:
        return SplitResult(obj_idx,np.empty(0,dtype=np.int64),query_idx,np.empty(0,dtype=np.int64),parent_cost,float('inf'))
    left=obj_idx[labels==0];right=obj_idx[labels==1]
    lk=np.unique(objects[labels==0].indices);rk=np.unique(objects[labels==1].indices)
    kw,bits=build_kw_lr_bitset(lk,rk)
    ql,qr=split_queries_by_kw_lr_bitset(queries,kw,bits,query_idx_global=query_idx)
    cost=node_cost(all_objects,all_queries,left,ql,context=context,keyword_count=len(lk))
    cost+=node_cost(all_objects,all_queries,right,qr,context=context,keyword_count=len(rk))
    pcq=p.EPSILON*len(query_idx)+2*p.ALFA*np.maximum(np.diff(queries.indptr)-1,0).sum()+2*p.BETA*len(query_idx)+p.GAMA*(len(ql)+len(qr))
    cost+=p.weighted_cost(34*len(kw)+64,pcq)
    return SplitResult(left,right,ql,qr,parent_cost,float(cost),lk,rk,kw,bits)
