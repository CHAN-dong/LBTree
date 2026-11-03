"""Experimental candidate-directed sample search replacing an MLP call."""
import heapq
import numpy as np
import parameter as p
from model.multilevel_bbc import ExactPartitionCost

def search(objects,queries,embeddings,initial,total_hits,budget=32):
    labels=np.asarray(initial,dtype=np.int64).copy();n=len(labels)
    if np.unique(labels).size<2:
        # A parent classifier often collapses on one of its own children.
        # Use two distant frozen-embedding anchors; do not retrain the encoder.
        x=np.asarray(embeddings);first=int(np.argmax(np.sum((x-x[0])**2,axis=1)))
        second=int(np.argmax(np.sum((x-x[first])**2,axis=1)))
        labels=(np.sum((x-x[second])**2,axis=1)<np.sum((x-x[first])**2,axis=1)).astype(np.int64)
    if np.unique(labels).size<2:return labels,0
    state=ExactPartitionCost(objects,queries,labels,total_hits)
    # Storage-only move estimates rank candidates. Every accepted move then
    # recalculates the true query-routing delta; estimates never accept moves.
    def priorities():
        result=np.zeros(n)
        for side in (0,1):
            rows=np.flatnonzero(state.labels==side);other=1-side
            a=state.a[rows]
            lost=np.asarray(a@(state.count[side]==1).astype(np.int64)).ravel()
            added=np.asarray(a@(state.count[other]==0).astype(np.int64)).ravel()
            result[rows]=p.w_s*p.W_S*(state.omega[other]-state.omega[side]-(state.n[side]+31)*lost+(state.n[other]+33)*added)
            result[rows]+=p.w_q*p.W_Q*(p.ALFA*(state.lens[other]-state.lens[side])+p.BETA*(state.wn[other]-state.wn[side]))
        return result
    priority=priorities();heap=list(zip(priority,np.arange(n)));heapq.heapify(heap);moves=0
    # Rebuild priorities once after the first candidate batch, avoiding a
    # repeated complete-object scan after every single move.
    for phase in range(2):
        for _ in range(min(budget//2,len(heap))):
            _,i=heapq.heappop(heap);moves+=state.move(int(i))
        if phase==0:
            priority=priorities();heap=list(zip(priority,np.arange(n)));heapq.heapify(heap)
    return state.labels,moves
