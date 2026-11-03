# Eq. (4): lambda * Query/Q_ref + (1-lambda) * Storage/S_ref.
LAMBDA = 0.5
w_s = 1.0 - LAMBDA
w_q = LAMBDA
S_REF = Q_REF = 1.0
W_S = W_Q = 1.0  # reciprocal global references, never recomputed per node
_INITIALIZED = False

CKPT_PATH = None  # legacy utilities only; use LBTREE_QUERY_MODEL for the paper path

# cost model parameters

EPSILON = 380.17
ALFA = 0.106
BETA  = 0
GAMA = 1.749

def set_root_costs(Ws: float, Wq: float):
    """Accept raw flat-index storage/query costs, not their reciprocals."""
    import math
    global S_REF, Q_REF, W_S, W_Q, _INITIALIZED
    if not all(math.isfinite(v) and v >= 0 for v in (Ws, Wq)):
        raise ValueError('Reference costs must be finite and nonnegative')
    S_REF, Q_REF = float(Ws), float(Wq)
    W_S = 1.0 / S_REF if S_REF else 0.0
    W_Q = 1.0 / Q_REF if Q_REF else 0.0
    _INITIALIZED = True

def set_tradeoff(value: float = 0.5):
    global LAMBDA, w_s, w_q
    if not 0 <= value <= 1:
        raise ValueError('lambda must be in [0, 1]')
    LAMBDA = float(value)
    w_s, w_q = 1.0 - LAMBDA, LAMBDA

def weighted_cost(storage, query):
    return w_s * W_S * storage + w_q * W_Q * query



# The paper uses a calibrated MLP. The old affine model is an explicit diagnostic.
import os as _os
COST_MODE = _os.environ.get('LBTREE_COST_MODE', 'mlp')
_QUERY_MODEL = None

def predict_query(n, u, results):
    global _QUERY_MODEL
    if COST_MODE == 'analytic':
        return EPSILON + ALFA * (u - 1) * n + BETA * n + GAMA * results
    if COST_MODE != 'mlp':
        raise ValueError('LBTREE_COST_MODE must be mlp or analytic')
    from model.query_latency import load_query_model, predict
    if _QUERY_MODEL is None:
        path = _os.environ.get('LBTREE_QUERY_MODEL')
        if not path:
            raise RuntimeError('Paper mode requires LBTREE_QUERY_MODEL: calibrate and train Mq first; '
                               'LBTREE_COST_MODE=analytic is a diagnostic, not paper reproduction')
        _QUERY_MODEL = load_query_model(path)
    return predict(_QUERY_MODEL, n, u, results)
