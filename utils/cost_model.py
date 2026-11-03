# utils/cost_model.py
from typing import List, Optional, Tuple
import parameter as P

def storage_cost(num_objects: int, num_keywords: int) -> float:
    """
    Storage(B) = |惟O|*|O| + 32*|惟O| + 32*|O|
    (bit-level model in the paper)
    """
    return (num_keywords * num_objects) + (32 * num_keywords) + (32 * num_objects)

def query_cost_single(
    num_objects: int,
    q_len: int,
    r_size: int,
) -> float:
    """
    Query(B,q) = 蔚 + 伪*(|惟q|-1)*|O| + 尾*|O| + 纬*|Rq|
    """
    if q_len <= 0:
        return 0.0
    return float(P.predict_query(num_objects, q_len, r_size))


def _get_row_term_ids(A_qw, q_idx: int) -> List[int]:
    """Return the column indices where A_qw[q_idx, j] == 1."""
    # scipy sparse
    if hasattr(A_qw, "getrow"):
        row = A_qw.getrow(q_idx)
        return row.indices.tolist()  # nonzero column ids
    # dense list
    return [j for j, v in enumerate(A_qw[q_idx]) if v == 1]


def _result_size_for_query(A_ow, term_ids: List[int]) -> int:
    """
    Given query term column ids, compute |Rq|:
    objects that contain ALL query terms.
    """
    if not term_ids:
        return 0

    # scipy sparse
    if hasattr(A_ow, "multiply"):
        v = A_ow[:, term_ids[0]]
        for t in term_ids[1:]:
            v = v.multiply(A_ow[:, t])
        # v is a column vector; count non-zeros => number of satisfying objects
        return int(v.count_nonzero())

    # dense list
    cnt = 0
    for i in range(len(A_ow)):
        ok = True
        row = A_ow[i]
        for t in term_ids:
            if row[t] != 1:
                ok = False
                break
        if ok:
            cnt += 1
    return cnt


def get_root_costs(
    A_ow,
    A_qw,
    object_query_hits=None,
) -> Tuple[float, float]:

    if hasattr(A_ow, "shape"):
        num_objects, num_keywords = A_ow.shape
    else:
        num_objects = len(A_ow)
        num_keywords = len(A_ow[0]) if num_objects > 0 else 0

    if hasattr(A_ow, 'indices'):
        import numpy as np
        num_keywords = np.unique(A_ow.indices).size
    W_S = storage_cost(num_objects, num_keywords)

    W_Q = 0.0
    if A_qw is None:
        return W_S, W_Q

    if hasattr(A_qw, "shape"):
        num_queries = A_qw.shape[0]
    else:
        num_queries = len(A_qw)

    # Build postings once for the global reference. Repeated CSR column slicing
    # otherwise scans the full object matrix for every term of every query.
    hit_counts = None
    if object_query_hits is not None:
        import numpy as np
        hit_counts = np.asarray(object_query_hits.sum(axis=0)).reshape(-1)
    postings = A_ow.tocsc() if hit_counts is None and hasattr(A_ow, 'tocsc') else None
    for qi in range(num_queries):
        term_ids = _get_row_term_ids(A_qw, qi)
        q_len = len(term_ids)
        if hit_counts is not None:
            r_size = int(hit_counts[qi])
        elif postings is not None and term_ids:
            import numpy as np
            lists = [postings.indices[postings.indptr[t]:postings.indptr[t+1]] for t in term_ids]
            lists.sort(key=len)
            hits = lists[0]
            for other in lists[1:]:
                if not hits.size:
                    break
                hits = np.intersect1d(hits, other)
            r_size = hits.size
        else:
            r_size = _result_size_for_query(A_ow, term_ids)
        W_Q += query_cost_single(num_objects, q_len, r_size)

    return W_S, W_Q
