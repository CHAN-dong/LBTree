"""Small CPU training and export smoke run, without machine-specific paths."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
from scipy.sparse import csr_matrix
from model.tree_builder import TreeBuilder
from model.tree_node_break import break_redundant_nodes

torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)
rng = np.random.default_rng(42)
out = Path('benchmarks/trained')
out.mkdir(parents=True, exist_ok=True)
objects = []
for i in range(256):
    group = i % 4
    objects.append(sorted(set([group * 8 + int(k) for k in rng.integers(0, 8, 8)])))
queries = [[g * 8 + k] for g in range(4) for k in range(8)]
def matrix(rows):
    return csr_matrix((np.ones(sum(map(len,rows)), dtype=np.float32),
                       np.concatenate(rows), np.r_[0, np.cumsum(list(map(len, rows)))]),
                      shape=(len(rows),32))
for name, rows in [('objects.csv',objects),('queries.csv',queries)]:
    (out/name).write_text('Title\n'+'\n'.join(' '.join('k'+str(k) for k in row) for row in rows)+'\n')
(out/'vocab.json').write_text(json.dumps({'k'+str(k):k for k in range(32)}))
builder=TreeBuilder(matrix(objects),matrix(queries),out_dir=str(out/'binary'),
                    N_sample=64,K_graph=5,epochs=3,min_objects=32,max_depth=2,device='cpu')
root=builder.build_tree(np.arange(len(objects)),np.arange(len(queries)))
info=break_redundant_nodes(builder, root_id=root, out_dir_break=str(out/'fanout'),copy_leaves=False)
print('Training and binary/fanout export succeeded:', info)
