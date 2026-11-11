"""Build the static LBTree-B and LBTree paths described in Sections IV-V."""
import argparse
import os
from pathlib import Path
import random
import json
import platform
import subprocess
import time
from importlib.metadata import version
import numpy as np
import torch
import parameter as param
from utils.text_io import build_Aow_from_objects, build_Aqw_from_workload
from model.tree_builder import TreeBuilder
from model.tree_node_break import break_redundant_nodes
from benchmarks.reproducibility import inspect, sha256, DEFAULT_CONFIG


def main():
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--protocol', type=Path, default=DEFAULT_CONFIG)
    selected, _ = preliminary.parse_known_args()
    protocol = json.loads(selected.protocol.read_text(encoding='utf-8'))
    defaults = protocol['training']
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', required=True)
    parser.add_argument('--workload', required=True, help='construction queries, separate from evaluation queries')
    parser.add_argument('--eval-workload', required=True, help='evaluation queries; checked before construction')
    parser.add_argument('--validation-workload', help='separate workload for parameter selection')
    parser.add_argument('--profile', choices=['paper_disjoint', 'historical_shared'], default=protocol['default_profile'])
    parser.add_argument('--dataset', choices=['DBLP','MARCO','SciDocs','SYN','AOL'])
    parser.add_argument('--protocol', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--enforce-paper-counts', action='store_true', help='reject mismatched counts for --dataset')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--query-model', help='offline-trained three-input Mq checkpoint')
    parser.add_argument('--epochs', type=int, default=defaults['epochs'])
    parser.add_argument('--lr', type=float, default=defaults['learning_rate'])
    parser.add_argument('--sample', type=int, default=None, help='default ceil(node objects / 100), minimum 2')
    parser.add_argument('--tradeoff', type=float, default=defaults['lambda_value'])
    parser.add_argument('--device', default=defaults['default_device'])
    parser.add_argument('--seed', type=int, default=protocol['seeds']['model_default'])
    parser.add_argument('--threads', type=int, default=defaults['cpu_threads'])
    parser.add_argument('--max-depth', type=int, default=defaults['max_depth_safeguard'], help='explicit construction safety limit')
    args = parser.parse_args()
    if args.enforce_paper_counts and (not args.dataset or args.profile != 'paper_disjoint'):
        parser.error('--enforce-paper-counts requires --dataset and --profile paper_disjoint')
    if args.epochs < 1 or args.lr <= 0 or args.threads < 1 or args.max_depth < 1:
        parser.error('epochs, lr, threads and max-depth must be positive')
    if args.sample is not None and args.sample < 2:
        parser.error('--sample must be at least two')
    preflight = inspect(args.objects, args.workload, args.eval_workload,
                        validation=args.validation_workload, dataset=args.dataset,
                        profile=args.profile, config=args.protocol, enforce_counts=args.enforce_paper_counts)
    if args.query_model: os.environ['LBTREE_QUERY_MODEL'] = args.query_model
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)
    param.set_tradeoff(args.tradeoff)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'run_manifest.json').exists() or any((args.out/k/'tree.bin').exists() for k in ('binary','fanout')):
        raise FileExistsError('Use a fresh output directory; previous manifests/indexes are preserved')
    checkpoint = args.query_model or os.environ.get('LBTREE_QUERY_MODEL')
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).parent, text=True).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'], cwd=Path(__file__).parent, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    source_root = Path(__file__).resolve().parent
    source_paths = [source_root/'test.py', source_root/'parameter.py', source_root/'CMakeLists.txt',
                    source_root/'requirements.txt']
    for folder in ('model', 'utils', 'index', 'benchmarks'):
        source_paths.extend(p for p in (source_root/folder).rglob('*')
                            if p.is_file() and p.suffix in ('.py','.cpp','.h','.hpp'))
    source_hashes = {str(p.relative_to(source_root)):sha256(p) for p in sorted(set(source_paths))}
    manifest = dict(preflight, status='building', config_sha256=sha256(args.protocol),
                    source_commit=commit, source_dirty=dirty,
                    source_file_sha256=source_hashes,
                    environment=dict(python=platform.python_version(), platform=platform.platform(),
                                     packages={k:version(k) for k in ('torch','numpy','scipy','scikit-learn')}),
                    training=dict(epochs=args.epochs, lr=args.lr, seed=args.seed, threads=args.threads,
                                  sample=args.sample, default_sampling='min(n,max(2,ceil(n/100)))',
                                  max_depth=args.max_depth, device=args.device, lambda_value=args.tradeoff,
                                  cost_mode=param.COST_MODE, partition='learned', early_stopping=False),
                    query_model=None if not checkpoint else dict(path=str(Path(checkpoint).resolve()), sha256=sha256(checkpoint)))
    def save_manifest():
        (args.out/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    save_manifest()
    a, vocab, _ = build_Aow_from_objects(args.objects, save_vocab_path=str(args.out/'vocab.json'))
    q, _ = build_Aqw_from_workload(args.workload, vocab)
    started = time.perf_counter()
    try:
        builder = TreeBuilder(a, q, out_dir=str(args.out/'binary'), N_sample=args.sample,
                              epochs=args.epochs, lr=args.lr, balance_lambda=0,
                              max_depth=args.max_depth, device=args.device)
        root = builder.build_tree(np.arange(a.shape[0]), np.arange(q.shape[0]))
        binary_seconds = time.perf_counter() - started
        fanout = break_redundant_nodes(builder, root, str(args.out/'fanout'), copy_leaves=False)
        manifest.update(status='complete', binary_build_seconds=binary_seconds,
                        construction_seconds=time.perf_counter()-started, fanout_info=fanout,
                        observed_objects=a.shape[0], observed_keywords=a.shape[1], observed_pairs=a.nnz)
        manifest['indexes'] = {k:dict(bytes=(args.out/k/'tree.bin').stat().st_size,
                                    sha256=sha256(args.out/k/'tree.bin')) for k in ('binary','fanout')}
        print(fanout)
    except Exception as error:
        manifest.update(status='failed', error=repr(error), failed_attempt_seconds=time.perf_counter()-started)
        raise
    finally:
        save_manifest()


if __name__ == '__main__': main()
