"""LBTree-G sampled clustering: k-means followed by exact Eq.(5) moves.

No graph, embeddings, surrogate loss, or post-stream refinement. The shared
ExactPartitionCost maintains Eq.(1)-(4) statistics, including parent bitmap
and keyword-feasibility query routing (not only queries with true results).
"""
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from model.multilevel_bbc import ExactPartitionCost


def cluster(objects, queries, total_hits, seed=42, max_rounds=20):
    if objects.shape[0]<2:raise ValueError('Two nonempty clusters require >=2 objects')
    if max_rounds<1:raise ValueError('max_rounds must be positive')
    active=np.unique(objects.indices)
    repaired=False
    if active.size:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            labels=KMeans(n_clusters=2,n_init=10,max_iter=300,random_state=seed).fit_predict(objects[:,active])
    else:labels=np.zeros(objects.shape[0],dtype=np.int64)
    if np.unique(labels).size<2:
        # Identical/zero feature rows cannot give two k-means clusters. Keep
        # every real object and deterministically seed a singleton second side.
        labels=np.zeros(objects.shape[0],dtype=np.int64);labels[-1]=1;repaired=True
    state=ExactPartitionCost(objects,queries,labels,total_hits)
    history=[state.cost()];rng=np.random.default_rng(seed);moves=0;converged=False
    for round_id in range(max_rounds):
        accepted=0
        for i in rng.permutation(objects.shape[0]):
            before=state.cost()
            if not state.move(int(i),accept_only=False):continue
            after=state.cost()
            if after<before:
                history.append(after);accepted+=1;moves+=1
            else:
                assert state.move(int(i),accept_only=False), 'Rollback must be legal'
                assert abs(state.cost()-before)<=1e-12*max(1.,abs(before))
        if accepted==0:converged=True;break
    assert np.all(state.n>0)
    assert all(b<a for a,b in zip(history,history[1:]))
    return state.labels.copy(),dict(cost_before=history[0],cost_after=state.cost(),
        accepted_costs=history,moves=moves,rounds=round_id+1,converged=converged,
        degenerate_kmeans_repaired=repaired,cluster_sizes=state.n.tolist())
