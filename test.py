# test.py
import numpy as np
import torch
from utils.text_io import build_Aow_from_objects, build_Aqw_from_workload

from model.tree_builder import TreeBuilder
from model.tree_node_break import break_redundant_nodes
from model.tree_cost import tree_cost_from_builder_meta, tree_cost_from_treejson
from utils.cost_model import get_root_costs
import parameter as param
import time

def main():

    # A_ow_all, vocab, _ = build_Aow_from_objects("/root/dong_11.02/MLKQI1.29/dataset/dblp/dblp_10w.csv", save_vocab_path = "./tree_out/vocab.json")
    # A_qw_all, _ = build_Aqw_from_workload("dataset/dblp/dblp_workload_1k.csv", vocab)
    # binary_out_dir = "./tree_out/"
    # dptree_out_dir = "./tree_out_break/"

    # A_ow_all, vocab, _ = build_Aow_from_objects("/root/dong_11.02/dataset/dblp/dblp_title.csv", save_vocab_path = "./tree_out/vocab.json")
    # A_qw_all, _ = build_Aqw_from_workload("/root/dong_11.02/dataset/dblp/dblp_10venues/workloads/test_all.csv", vocab)
    # binary_out_dir = "./tree_out/dblp_top10/"
    # dptree_out_dir = "./tree_out_break/dblp_top10/"

    # A_ow_all, vocab, _ = build_Aow_from_objects("/root/dong_11.02/dataset/dblp/dblp_title.csv", save_vocab_path = "./tree_out/vocab.json")
    # A_qw_all, _ = build_Aqw_from_workload("/root/dong_11.02/dataset/dblp/dblp_top3/workloads/test_all.csv", vocab)
    # binary_out_dir = "./tree_out/dblp_top3/"
    # dptree_out_dir = "./tree_out_break/dblp_top3/"

    # A_ow_all, vocab, _ = build_Aow_from_objects("/root/dong_11.02/dataset/msmarco/msmarco.csv", save_vocab_path = "./tree_out/vocab.json")
    # A_qw_all, _ = build_Aqw_from_workload("/root/dong_11.02/dataset/msmarco/workload.csv", vocab)
    # binary_out_dir = "./tree_out/msmarco/"
    # dptree_out_dir = "./tree_out_break/msmarco/"

    A_ow_all, vocab, _ = build_Aow_from_objects("/root/dong_11.02/dataset/ir_datasets/ir_datasets.csv", save_vocab_path = "./tree_out/vocab.json")
    A_qw_all, _ = build_Aqw_from_workload("/root/dong_11.02/dataset/ir_datasets/workload.csv", vocab)
    binary_out_dir = "./tree_out/ir_datasets/"
    dptree_out_dir = "./tree_out_break/ir_datasets/"

    # A_ow_all, vocab, _ = build_Aow_from_objects("/root/dong_11.02/dataset/synthetic/synthetic_dataset.csv", save_vocab_path = "./tree_out/vocab.json")
    # A_qw_all, _ = build_Aqw_from_workload("/root/dong_11.02/dataset/synthetic/synthetic_query.csv", vocab)
    # binary_out_dir = "./tree_out/synthetic/"
    # dptree_out_dir = "./tree_out_break/synthetic/"

    print("A_ow shape:", A_ow_all.shape)
    print("A_qw_all shape:", A_qw_all.shape)

    # improverate = 1e-4
    # cst = get_root_costs(A_ow=A_ow_all, A_qw=A_qw_all)

    cost_improve_eps = 10000
    # cost_improve_eps = 100
    # param.set_root_costs(1/Ws, 1/Wq)
    # print("Init W_S =", param.W_S, "Init W_Q =", param.W_Q)

    n_all = A_ow_all.shape[0]
    m_all = A_qw_all.shape[0]

    root_obj_idx = np.arange(n_all, dtype=np.int64)
    root_query_idx = np.arange(m_all, dtype=np.int64)
    
    learn_rate = 1e-3

    builder = TreeBuilder(
        A_ow_all=A_ow_all,
        A_qw_all=A_qw_all,
        out_dir=binary_out_dir,
        N_sample=1000,          # sampling size
        K_graph=20,
        epochs=300,
        lr=learn_rate,
        balance_lambda=1e-3,
        min_objects=0,         # minimal objects size of stop splitting
        max_depth=100,
        cost_improve_eps = cost_improve_eps,   # cost must be decrease
    )

    t_binary_before = time.perf_counter()
    root_id = builder.build_tree(root_obj_idx, root_query_idx)
    t_binary_after = time.perf_counter()
    print(f"binary construct time: {float(t_binary_after - t_binary_before)} s")

    t_fanout_before = time.perf_counter()
    info = break_redundant_nodes(builder, root_id=root_id, out_dir_break=dptree_out_dir, copy_leaves=False)
    t_fanout_after = time.perf_counter()
    print(f"fanout-tree construct time: {float(t_fanout_after - t_fanout_before)} s")
    print("Break done:", info)

    # print("Original bitmap cost: ", cst)

    # cost_before_tree = tree_cost_from_builder_meta(builder, root_id, A_ow_all=A_ow_all, A_qw_all=A_qw_all)
    # print("Original tree cost (tree-sum):", cost_before_tree)
    # 3) break 后树 cost（从 break tree.json）
    # cost_after_tree = tree_cost_from_treejson("./tree_out_break", A_ow_all=A_ow_all, A_qw_all=A_qw_all)
    # print("Break tree cost (tree-sum):", cost_after_tree)

    # print("DP cost_before:", info["cost_before_dp"], "DP cost_after:", info["cost_after_dp"], "best_k:", info["best_k_root"])


if __name__ == "__main__":
    main()
