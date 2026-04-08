# utils/cost_model.py
from typing import List, Optional, Tuple
import parameter as P

def storage_cost(num_objects: int, num_keywords: int) -> float:
    """
    Storage(B) = |ΩO|*|O| + 32*|ΩO| + 32*|O|
    (bit-level model in the paper)
    """
    return (num_keywords * num_objects) + (32 * num_keywords) + (32 * num_objects)

def query_cost_single(
    num_objects: int,
    q_len: int,
    r_size: int,
) -> float:
    """
    Query(B,q) = ε + α*(|Ωq|-1)*|O| + β*|O| + γ*|Rq|
    """
    if q_len <= 0:
        return 0.0
    return P.EPSILON + P.ALFA * (q_len - 1) * num_objects + P.BETA * num_objects + P.GAMA * r_size


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
) -> Tuple[float, float]:

    if hasattr(A_ow, "shape"):
        num_objects, num_keywords = A_ow.shape
    else:
        num_objects = len(A_ow)
        num_keywords = len(A_ow[0]) if num_objects > 0 else 0

    W_S = storage_cost(num_objects, num_keywords)

    W_Q = 0.0
    if A_qw is None:
        return W_S, W_Q

    if hasattr(A_qw, "shape"):
        num_queries = A_qw.shape[0]
    else:
        num_queries = len(A_qw)

    for qi in range(num_queries):
        term_ids = _get_row_term_ids(A_qw, qi)
        q_len = len(term_ids)
        r_size = _result_size_for_query(A_ow, term_ids)
        W_Q += query_cost_single(num_objects, q_len, r_size)

    return P.w_s * W_S + P.w_q * W_Q