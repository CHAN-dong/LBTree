W_S = 1
W_Q = 1

w_s = 1
w_q = 0

CKPT_PATH = "/data/gpfs/projects/punim2898/MLKQI1.29/dataset/datasets_for_costmodel/Mc_mlp_controlled_accuracy_plots_data.xlsx"

# cost model parameters

EPSILON = 380.17
ALFA = 0.106
BETA  = 0
GAMA = 1.749

def set_root_costs(Ws: float, Wq: float):
    global W_S, W_Q, _INITIALIZED
    W_S = float(Ws)
    W_Q = float(Wq)
    _INITIALIZED = True

