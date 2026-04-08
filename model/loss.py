# model/loss.py
import torch
import parameter as param
def surrogate_stats(
    Aow_sparse: torch.Tensor,
    Aqw_sparse: torch.Tensor,
    pi: torch.Tensor,
    r: torch.Tensor,
    eps_stable: float = 1e-9,
    tau: float = 1.0,
    hard: bool = True,          
    hard_thr: float = 0.5,    
):
    # 1) make "differentiable hard" assignment
    if hard:
        # 1) hard 0/1 assignment
        z = (pi >= hard_thr).to(pi.dtype)

        # 3) Straight-Through Estimator (STE):
        #    forward uses hard z, backward uses soft pi
        pi_used = z + (pi - pi.detach())
    else:
        z = None
        pi_used = pi

    # (d,) expected "keyword counts" on each side
    sL = torch.sparse.mm(Aow_sparse.transpose(0, 1), pi_used.unsqueeze(1)).squeeze(1)
    sR = torch.sparse.mm(Aow_sparse.transpose(0, 1), (1.0 - pi_used).unsqueeze(1)).squeeze(1)

    # keyword existence gate
    # --- soft gate (for backward) ---
    kL_soft = 1.0 - torch.exp(-sL / tau)
    kR_soft = 1.0 - torch.exp(-sR / tau)
    kL_soft = torch.clamp(kL_soft, 0.0, 1.0 - 1e-6)
    kR_soft = torch.clamp(kR_soft, 0.0, 1.0 - 1e-6)

    if hard:
        # --- hard gate (match "nonzero") ---
        kw_eps = 0.0  # 或者 1e-12，避免数值噪声
        kL_hard = (sL > kw_eps).to(sL.dtype)
        kR_hard = (sR > kw_eps).to(sR.dtype)

        # --- STE: forward hard, backward soft ---
        kL = kL_hard + (kL_soft - kL_soft.detach())
        kR = kR_hard + (kR_soft - kR_soft.detach())
    else:
        kL = kL_soft
        kR = kR_soft

    # soft |O| and |Ω|
    oL = pi_used.sum()
    oR = (1.0 - pi_used).sum()
    omegaL = kL.sum()
    omegaR = kR.sum()

    # query membership surrogate
    logkL = torch.log(kL + eps_stable).unsqueeze(1)
    logkR = torch.log(kR + eps_stable).unsqueeze(1)

    qL = torch.exp(torch.sparse.mm(Aqw_sparse, logkL).squeeze(1))
    qR = torch.exp(torch.sparse.mm(Aqw_sparse, logkR).squeeze(1))

    q_len = torch.sparse.sum(Aqw_sparse, dim=1).to_dense()
    non_empty = (q_len > 0).to(qL.dtype)
    qL = qL * non_empty
    qR = qR * non_empty

    wL = qL.sum()
    wR = qR.sum()
    sum_qL = (qL * q_len).sum()
    sum_qR = (qR * q_len).sum()
    sum_rL = (qL * r).sum()
    sum_rR = (qR * r).sum()

    left = dict(o=oL, omega=omegaL, w=wL, sum_q=sum_qL, sum_r=sum_rL)
    right = dict(o=oR, omega=omegaR, w=wR, sum_q=sum_qR, sum_r=sum_rR)

    return left, right, kL, kR, z

def bitmap_cost(o, omega, w, sum_q, sum_r):
    # storage term
    storage = (omega * o + 32.0 * omega + 32.0 * o)

    # query term
    query = (w * param.EPSILON
             + param.ALFA * sum_q * o
             + param.BETA * w * o
             + param.GAMA * sum_r
             )

    # print(f"insize: ---- sum_q: {sum_q}, sum_r: {sum_r}, o: {o}")

    return param.w_s * param.W_S * storage + param.w_q * param.W_Q * query


def cost_element_expand(d, r_o, r_omega):
    return {
        "o":     d["o"] * r_o,
        "omega": d["omega"] * r_omega,
        "w":     d["w"],
        "sum_q": d["sum_q"],
        "sum_r": d["sum_r"],
    }

def clustering_loss(
    Aow_sparse: torch.Tensor,
    Aqw_sparse: torch.Tensor,
    pi: torch.Tensor,
    r: torch.Tensor,
    r_o: float = 0, 
    r_omega: float = 0,
    balance_lambda: float = 1e-2,
    tau: float = 1.0,
    entropy_lambda: float = 1e-3,
    return_debug: bool = True,
):

    """
    total = Cost(left) + Cost(right) + regularizers

    Regularizers (optional but helpful to avoid collapse):
      - balance: encourage pi.mean ~ 0.5
      - entropy: encourage pi not all 0/1 too early
    """
    left, right, kL, kR, _= surrogate_stats(Aow_sparse, Aqw_sparse, pi, r, tau=tau)

    # left = cost_element_expand(left, r_o = r_o, r_omega = r_omega)
    # right = cost_element_expand(right, r_o = r_o, r_omega = r_omega)

    loss_main = bitmap_cost(**left) + bitmap_cost(**right)

    loss = loss_main

    if balance_lambda > 0:
        loss = loss + balance_lambda * (pi.mean() - 0.5) ** 2

    if entropy_lambda > 0:
        # maximize entropy => minimize -H
        p = torch.clamp(pi, 1e-6, 1 - 1e-6)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
        loss = loss - entropy_lambda * entropy


    debug = {
        "loss_main": loss_main.detach(),
        "oL": left["o"], "oR": right["o"],
        "omegaL": left["omega"], "omegaR": right["omega"],
        "wL": left["w"], "wR": right["w"],
    }
    return loss, debug
