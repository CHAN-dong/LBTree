"""Opt-in factorized BBC and inductive partitioning; legacy BBC is untouched.

Q = rowmean(A_qw K0); K = ReLU(K0 + Wq rowmean(A_qw.T Q));
O = rowmean(A_ow K). No object pairs, Jaccard, or object-object graph.
"""
import copy,time
import numpy as np
import torch
from torch import nn
from scipy.sparse import csr_matrix
from model.gcn_encoder import csr_to_torch_sparse
from model.loss import clustering_loss,prepare_loss_context
from model.cluster_mlp import BinaryClusterMLP
from model.split_cache import BestValidSplit


def rowmean(a,power=1):
    a=a.tocsr();counts=np.asarray(a.sum(axis=1)).ravel()
    return a.multiply((1/np.maximum(counts,1)**power)[:,None]).tocsr()


class FactorizedEncoder(nn.Module):
    def __init__(self,vocab,width=64):
        super().__init__();self.W0=nn.Parameter(torch.empty(vocab,width));nn.init.xavier_uniform_(self.W0)
        self.query_projection=nn.Linear(width,width,bias=False)

    def keywords(self,qnorm,kfromq,keys=None):
        base=self.W0 if keys is None else self.W0.index_select(0,keys)
        queries=torch.sparse.mm(qnorm,base)
        # Move the bias-free linear map before the sparse aggregation: exactly
        # the same relation operator, O(m*r^2) rather than O(keywords*r^2).
        context=torch.sparse.mm(kfromq,self.query_projection(queries))
        return torch.relu(base+context)


class InductiveHead(nn.Module):
    def __init__(self,width=64,rank=0):
        super().__init__();self.linear=nn.Linear(width,1)
        self.factors=nn.Linear(width,rank,bias=False) if rank else None

    def forward(self,keywords,objects,objects_squared=None):
        scores=torch.sparse.mm(objects,self.linear(keywords)-self.linear.bias).squeeze(1)+self.linear.bias[0]
        if self.factors is not None:
            v=self.factors(keywords);summed=torch.sparse.mm(objects,v)
            correction=torch.sparse.mm(objects_squared,v.square())
            scores=scores+.5*(summed.square()-correction).sum(dim=1)
        return scores

    def lookup(self,keywords):
        scores=self.linear(keywords)-self.linear.bias
        if self.factors is None:return scores
        v=self.factors(keywords)
        return torch.cat((scores,v,v.square()),dim=1)


def predict_scan(a,lookup,bias,rank=0,batch=65536,aggregation='mean'):
    """One fused CSR/dense product per batch, each incidence used once.

    Linear/factor sums and squared-factor sums share the same sparse scan.
    No per-object delta cost. All original sample objects are predicted too.
    """
    labels=np.empty(a.shape[0],dtype=np.int64)
    for start in range(0,a.shape[0],batch):
        block=a[start:start+batch];raw=block@lookup
        # Input is a canonical binary incidence matrix. Row lengths come from
        # CSR pointers, avoiding a second pass over object-keyword entries.
        count=np.maximum(np.diff(block.indptr),1).astype(raw.dtype,copy=False) if aggregation=='mean' else np.ones(len(raw),dtype=raw.dtype)
        logits=raw[:,0]/count+bias
        if rank:
            logits=logits+.5*((raw[:,1:1+rank]/count[:,None])**2-raw[:,1+rank:]/count[:,None]**2).sum(axis=1)
        labels[start:start+len(count)]=(logits>=0).astype(np.int64)
    return labels


def split(a_node,q_node,obj_idx,query_idx,sample_local,parent_cost,context,node_hits,
          a_all,q_all,cache,mode,epochs,lr,patience,min_delta,is_root,width,device,node_keywords=None):
    from model.model_split import _record_stage,_complete_external_partition,get_tau
    from model.multilevel_bbc import result_from_labels
    from model.cooccurrence_graph import _build_hit_matrix_H
    if cache is None:
        from model.split_cache import SplitModelCache
        cache=SplitModelCache()
    start=time.perf_counter();sample=a_node[sample_local]
    aggregation=getattr(cache,'fi_aggregation','mean')
    if aggregation not in ('mean','sum'):raise ValueError('Unknown object aggregation')
    active=np.union1d(sample.indices,q_node.indices)
    aq=q_node[:,active];ao=sample[:,active]
    qnorm=csr_to_torch_sparse(rowmean(aq)).to(device)
    kfromq=csr_to_torch_sparse(rowmean(aq.T)).to(device)
    objectkeys=np.unique(ao.indices)
    xnorm=csr_to_torch_sparse(rowmean(ao[:,objectkeys]) if aggregation=='mean' else ao[:,objectkeys]).to(device)
    xsquared=csr_to_torch_sparse(rowmean(ao[:,objectkeys],2) if aggregation=='mean' else ao[:,objectkeys]).to(device) if mode=='fi_fm' else None
    objectkey_tensor=torch.tensor(objectkeys,dtype=torch.int64,device=device)
    _record_stage(cache,'graph_seconds',time.perf_counter()-start)
    start=time.perf_counter()
    loss_objects=csr_to_torch_sparse(ao).to(device);loss_queries=csr_to_torch_sparse(aq).to(device)
    loss_context=prepare_loss_context(loss_objects,loss_queries)
    hits=node_hits[sample_local] if node_hits is not None else _build_hit_matrix_H(sample,q_node)
    r=torch.tensor(np.asarray(hits.sum(axis=0)).ravel().astype(np.float32),device=device)
    _record_stage(cache,'tensor_preparation_seconds',time.perf_counter()-start)
    start=time.perf_counter()
    encoder=getattr(cache,'fi_encoder',None)
    if encoder is None:encoder=FactorizedEncoder(a_all.shape[1],width).to(device)
    head=BinaryClusterMLP(in_dim=width,hidden_dim=2*width,num_layers=2,dropout=0.) if mode=='fi_stream' else InductiveHead(width,8 if mode=='fi_fm' else 0)
    head=head.to(device)
    previous=getattr(cache,'fi_head_state',None)
    if previous is not None:head.load_state_dict(previous)
    keys=torch.tensor(active,dtype=torch.int64,device=device)
    full_weight=None
    if is_root:
        full_weight=encoder.W0
        encoder.W0=nn.Parameter(full_weight.detach().index_select(0,keys));training_keys=None
        optimizer=torch.optim.Adam(list(encoder.parameters())+list(head.parameters()),lr=lr)
        fixed=None
    else:
        for weight in encoder.parameters():weight.requires_grad_(False)
        encoder.eval();training_keys=keys
        fixed_start=time.perf_counter()
        with torch.no_grad():fixed=encoder.keywords(qnorm,kfromq,keys).detach()
        cache.fi_inference_seconds=getattr(cache,'fi_inference_seconds',0.)+time.perf_counter()-fixed_start
        optimizer=torch.optim.Adam(head.parameters(),lr=lr)
    def logits(kw):
        # Query-only terms still participate in the encoder and cost loss;
        # they have zero object incidence, so the head need not evaluate them.
        used=kw.index_select(0,objectkey_tensor)
        if mode=='fi_stream':return head(torch.sparse.mm(xnorm,used))
        return head(used,xnorm,xsquared)
    best=float('inf');bad=0;checkpoint=BestValidSplit();ep=-1
    for ep in range(epochs):
        encoder.train(is_root);head.train();optimizer.zero_grad()
        kw=encoder.keywords(qnorm,kfromq,training_keys) if is_root else fixed
        pi=torch.sigmoid(logits(kw)/get_tau(ep))
        # Deliberately call the exact same baseline function/default kwargs.
        # No FI-specific balance, entropy, contrastive or auxiliary objective.
        loss,_=clustering_loss(loss_objects,loss_queries,pi,r,context=loss_context)
        if not is_root:checkpoint.consider(head,loss,pi)
        loss.backward();optimizer.step();score=float(loss.detach())
        cache.sample_object_epochs+=len(sample_local)
        if not is_root:
            if score<best-min_delta:best=score;bad=0
            else:bad+=1
            if bad>=patience:break
    if not is_root:checkpoint.restore(head)
    if is_root:
        with torch.no_grad():full_weight.index_copy_(0,keys,encoder.W0.detach())
        encoder.W0=full_weight;cache.root_epochs+=ep+1
    else:cache.nonroot_epochs+=ep+1
    cache.training_calls+=1;cache.fi_encoder=encoder;cache.fi_head_state=copy.deepcopy(head.state_dict())
    _record_stage(cache,'training_and_inference_seconds',time.perf_counter()-start)
    infer_start=time.perf_counter();encoder.eval();head.eval()
    with torch.no_grad():
        keyword_vectors=encoder.keywords(qnorm,kfromq,keys)
        sample_labels=(logits(keyword_vectors)>=0).cpu().numpy().astype(np.int64)
    cache.fi_inference_seconds=getattr(cache,'fi_inference_seconds',0.)+time.perf_counter()-infer_start
    # Include final sample inference in the same aggregate definition used by
    # legacy training_and_inference_seconds.
    _record_stage(cache,'training_and_inference_seconds',time.perf_counter()-infer_start)
    assign_start=time.perf_counter()
    if mode=='fi_stream':
        result=_complete_external_partition(a_node,q_node,obj_idx,query_idx,sample_local,sample_labels,
            parent_cost,context,node_hits,a_all,q_all)
    else:
        with torch.no_grad():
            # Tree construction already knows this exact sorted vocabulary.
            # Reuse it rather than sorting all object incidences before scan.
            nodekeys=node_keywords if node_keywords is not None else np.unique(a_node.indices)
            vectors=torch.relu(encoder.W0.index_select(0,torch.tensor(nodekeys,dtype=torch.int64,device=device)))
            pos=np.searchsorted(nodekeys,active);valid=pos<len(nodekeys)
            valid[valid]=nodekeys[pos[valid]]==active[valid]
            vectors[torch.tensor(pos[valid],device=device)]=keyword_vectors[torch.tensor(np.flatnonzero(valid),device=device)]
            rank=8 if mode=='fi_fm' else 0;columns=1+2*rank
            if not hasattr(cache,'fi_lookup'):cache.fi_lookup=np.zeros((a_all.shape[1],columns),dtype=np.float32)
            cache.fi_lookup[nodekeys]=head.lookup(vectors).cpu().numpy()
            labels=predict_scan(a_node,cache.fi_lookup,float(head.linear.bias[0]),rank,aggregation=aggregation)
        _record_stage(cache,'inductive_assignment_seconds',time.perf_counter()-assign_start)
        route_start=time.perf_counter()
        result=result_from_labels(a_node,q_node,obj_idx,query_idx,labels,parent_cost,context,a_all,q_all)
        _record_stage(cache,'split_cost_validation_seconds',time.perf_counter()-route_start)
    if mode=='fi_stream':_record_stage(cache,'streaming_and_query_routing_seconds',time.perf_counter()-assign_start)
    if not hasattr(cache,'fi_audit'):cache.fi_audit=[]
    cache.fi_audit.append(dict(objects=len(obj_idx),sample=len(sample_local),queries=len(query_idx),
        graph_incidence_edges=ao.nnz+aq.nnz,object_pair_edges=0,epochs=ep+1,aggregation=aggregation,
        sample_left=int((sample_labels==0).sum()),sample_right=int((sample_labels==1).sum()),
        left=len(result.obj_left),right=len(result.obj_right),parent_cost=float(parent_cost),split_cost=float(result.cost_children)))
    return result
