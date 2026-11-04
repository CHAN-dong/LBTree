"""Validate benchmark inputs before training; never infer a held-out split."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/reproducibility.json'
TOKEN = re.compile(r'[a-z0-9]+')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_sets(path, *, objects=False):
    """Strict public input contract, including normalized query duplicates."""
    with Path(path).open(encoding='utf-8-sig') as f:
        if f.readline().rstrip('\r\n') != 'Title':
            raise ValueError(f'{path}: expected one Title header')
        for line_number, line in enumerate(f, 2):
            text = line.rstrip('\r\n')
            tokens = text.split()
            if not tokens or any(TOKEN.fullmatch(t) is None for t in tokens):
                raise ValueError(f'{path}:{line_number}: empty or non-normalized keyword set')
            if len(tokens) != len(set(tokens)):
                raise ValueError(f'{path}:{line_number}: duplicate terms')
            yield tuple(sorted(tokens))


def workload_stats(train, evaluation, policy, validation=None):
    train = list(train)
    evaluation = list(evaluation)
    valid = list(validation) if validation is not None else None
    t, e = set(train), set(evaluation)
    if policy == 'shared' and train != evaluation:
        raise ValueError('shared profile requires the identical ordered train/evaluation workload')
    if policy == 'disjoint' and t & e:
        raise ValueError(f'normalized train/test overlap: {len(t & e)} keyword sets')
    if policy not in ('shared', 'disjoint'):
        raise ValueError(f'Unknown workload policy: {policy}')
    if valid is not None and set(valid) & (t | e):
        raise ValueError('validation must be combination-disjoint from construction and evaluation')
    return dict(construction_queries=len(train), evaluation_queries=len(evaluation),
                construction_unique_queries=len(t), evaluation_unique_queries=len(e),
                normalized_overlap=len(t & e), validation_queries=None if valid is None else len(valid),
                parameter_selection='fixed_defaults_no_selection' if valid is None else 'validation_only',
                construction_query_length_histogram=dict(Counter(map(len, train))),
                evaluation_query_length_histogram=dict(Counter(map(len, evaluation))))


def inspect(objects, train, evaluation, *, dataset=None, profile='paper_disjoint',
            validation=None, config=DEFAULT_CONFIG, enforce_counts=False):
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    if profile not in cfg['profiles']:
        raise ValueError(f'Unknown profile: {profile}')
    policy = cfg['profiles'][profile]['workload_policy']
    train_rows = list(read_sets(train))
    eval_rows = list(read_sets(evaluation))
    val_rows = list(read_sets(validation)) if validation else None
    report = workload_stats(train_rows, eval_rows, policy, val_rows)
    counts = Counter()
    vocabulary = set()
    for row in read_sets(objects, objects=True):
        counts['objects'] += 1
        counts['keyword_object_pairs'] += len(row)
        vocabulary.update(row)
    counts['active_keywords'] = len(vocabulary)
    oov_train = {w for row in train_rows for w in row if w not in vocabulary}
    if oov_train:
        raise ValueError(f'{len(oov_train)} out-of-vocabulary construction terms')
    report.update(counts)
    report['evaluation_oov_query_count'] = sum(any(w not in vocabulary for w in row) for row in eval_rows)
    report.update(profile=profile, workload_policy=policy, dataset=dataset,
                  generalization_evaluation=policy == 'disjoint',
                  inputs={name: dict(path=str(Path(path).resolve()), sha256=sha256(path))
                          for name, path in [('objects', objects), ('construction', train), ('evaluation', evaluation)]})
    if validation:
        report['inputs']['validation'] = dict(path=str(Path(validation).resolve()), sha256=sha256(validation))
    report['count_mismatches'] = []
    if dataset:
        spec = cfg['datasets'][dataset]
        target = spec['paper_claim'] if profile == 'paper_disjoint' else spec['historical_shared']
        expected = {k: target[k] for k in ('objects', 'keyword_object_pairs', 'construction_queries', 'evaluation_queries')}
        expected['active_keywords'] = target['active_keywords']
        for field, value in expected.items():
            if report[field] != value:
                report['count_mismatches'].append(dict(field=field, expected=value, actual=report[field]))
    report['selected_profile_counts_matched'] = not report['count_mismatches'] if dataset else None
    report['paper_counts_matched'] = report['selected_profile_counts_matched'] if profile == 'paper_disjoint' else None
    if enforce_counts and report['count_mismatches']:
        raise ValueError(f'Dataset differs from selected protocol: {report["count_mismatches"]}')
    report['full_performance_reproduction'] = False
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objects', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--validation')
    p.add_argument('--dataset', choices=['DBLP', 'MARCO', 'SciDocs', 'SYN'])
    p.add_argument('--profile', choices=['paper_disjoint', 'historical_shared'], default='paper_disjoint')
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--enforce-counts', action='store_true')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    report = inspect(a.objects, a.train, a.evaluation, dataset=a.dataset, profile=a.profile,
                     validation=a.validation, config=a.config, enforce_counts=a.enforce_counts)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
