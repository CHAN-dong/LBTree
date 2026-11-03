"""Bounded, per-build exact object/query hits shared by all recursive nodes."""
import numpy as np
from model.cooccurrence_graph import _build_hit_matrix_H


class BuildContext:
    def __init__(self, objects, queries, max_hits=8_000_000):
        # The old cost casts column sums to int64. Values >= 1 guarantee that
        # each present column survives that cast (including non-binary counts).
        self.positive_objects = bool(np.all(objects.data >= 1))
        self.hits = (_build_hit_matrix_H(objects, queries, max_hits=max_hits)
                     if objects.shape[0]*8 <= 128*1024*1024 else None)
        self.query_lengths = np.diff(queries.indptr).astype(np.int64)

    def slice_hits(self, objects, queries):
        if self.hits is None:
            return None
        return self.hits[objects][:, queries]

    @property
    def bytes(self):
        if self.hits is None:
            return 0
        return self.hits.data.nbytes + self.hits.indices.nbytes + self.hits.indptr.nbytes
