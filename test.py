"""Build the static LBTree-B and LBTree paths described in Sections IV-V."""
import argparse
import os
from pathlib import Path
import random
import numpy as np
import torch
import parameter as param
from utils.text_io import build_Aow_from_objects, build_Aqw_from_workload
from model.tree_builder import TreeBuilder
from model.tree_node_break import break_redundant_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', required=True)
    parser.add_argument('--workload', required=True, help='construction queries, separate from evaluation queries')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--query-model', help='offline-trained three-input Mq checkpoint')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=.0005)
    parser.add_argument('--sample', type=int, default=None, help='default ceil(node objects / 100), minimum 2')
    parser.add_argument('--tradeoff', type=float, default=.5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--max-depth', type=int, default=100, help='explicit construction safety limit')
    args = parser.parse_args()
    if args.query_model: os.environ['LBTREE_QUERY_MODEL'] = args.query_model
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)
    param.set_tradeoff(args.tradeoff)
    args.out.mkdir(parents=True, exist_ok=True)
    a, vocab, _ = build_Aow_from_objects(args.objects, save_vocab_path=str(args.out/'vocab.json'))
    q, _ = build_Aqw_from_workload(args.workload, vocab)
    builder = TreeBuilder(a, q, out_dir=str(args.out/'binary'), N_sample=args.sample,
                          epochs=args.epochs, lr=args.lr, balance_lambda=0,
                          max_depth=args.max_depth, device=args.device)
    root = builder.build_tree(np.arange(a.shape[0]), np.arange(q.shape[0]))
    print(break_redundant_nodes(builder, root, str(args.out/'fanout'), copy_leaves=False))


if __name__ == '__main__': main()
