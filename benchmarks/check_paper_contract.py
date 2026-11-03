"""Small, equation-level regressions; these are not paper performance results."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.sparse import csr_matrix
import parameter as param
from model.paper_bbc import prepare_paper_loss, paper_loss, RelationEncoder, row_normalize, sparse_tensor
from model.query_latency import QueryLatencyMLP
from model.streaming_assign import streaming_assign
from model.model_split import node_cost
from model.tree_builder import TreeBuilder, TreeNodeMeta
from model.tree_node_break import TreeNodeBreaker
from model.build_context import BuildContext
from model.query_split import build_kw_lr_bitset
from utils.packed_bitmap import make_leaf_keyword_object_bitmaps
from utils.text_io import build_Aqw_from_workload


def matrix(rows, d):
    rows = [sorted(set(row)) for row in rows]
    return csr_matrix((np.ones(sum(map(len, rows)), np.float32),
                       np.array([k for row in rows for k in row]),
                       np.r_[0, np.cumsum([len(row) for row in rows])]), shape=(len(rows), d))


class Contract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        param.COST_MODE = 'mlp'
        param._QUERY_MODEL = QueryLatencyMLP().double().eval()
        for p in param._QUERY_MODEL.parameters(): p.requires_grad_(False)
        param.set_tradeoff(.5); param.set_root_costs(1000., 20.)

    def test_missing_mlp_is_not_an_implicit_affine_fallback(self):
        previous = param._QUERY_MODEL
        path = os.environ.pop('LBTREE_QUERY_MODEL', None)
        try:
            param._QUERY_MODEL = None
            with self.assertRaisesRegex(RuntimeError, 'LBTREE_QUERY_MODEL'):
                param.predict_query(10, 2, 1)
        finally:
            param._QUERY_MODEL = previous
            if path is not None: os.environ['LBTREE_QUERY_MODEL'] = path

    def test_soft_or_and_partition_specific_results(self):
        a = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], float)
        q = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], float)
        ctx = prepare_paper_loss(csr_matrix(a), csr_matrix(q), dtype=torch.float64)
        for values in [[.2, .7, .4], [1., 0., 0.]]:
            pi = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            loss, stats = paper_loss(pi, ctx)
            hit = ((a[:, None, :] >= q[None, :, :]).all(2)).astype(float)
            expected = 0.
            gates = []
            for assign, actual in zip([np.array(values), 1-np.array(values)], stats['children']):
                presence = 1-np.prod(1-a*assign[:, None], axis=0)
                gate = np.prod(np.where(q, presence, 1), axis=1)
                result = hit.T @ assign
                np.testing.assert_allclose(actual['presence'].detach(), presence)
                np.testing.assert_allclose(actual['results'].detach(), result)
                np.testing.assert_allclose(actual['gates'].detach(), gate)
                count = assign.sum(); terms = presence.sum()
                query = (gate * param.predict_query(count, q.sum(1), result)).sum()
                expected += param.weighted_cost(terms*count+32*terms+32*count, query)
                gates.append(gate)
            parent = param.predict_query(2, q.sum(1), gates[0]+gates[1]).sum()
            expected += param.weighted_cost(2*3+32*3+64, parent)
            self.assertAlmostEqual(loss.item(), expected, places=6)
            loss.backward(); self.assertTrue(torch.isfinite(pi.grad).all())
        # Eq. (14) cannot reuse the unpartitioned hit count for both children.
        self.assertFalse(np.allclose(stats['children'][0]['results'].detach(), hit.sum(0)))

    def test_gradcheck(self):
        a = matrix([[0, 1], [1, 2], [0, 2]], 3); q = matrix([[0], [0, 1], [2]], 3)
        ctx = prepare_paper_loss(a, q, dtype=torch.float64)
        pi = torch.tensor([.23, .61, .47], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda v: paper_loss(v, ctx)[0], pi))

    def test_relation_encoding_dense_equation(self):
        aq = matrix([[0, 1], [1], [1]], 4)
        x = torch.arange(4*18, dtype=torch.float32).reshape(4, 18)/72
        encoder = RelationEncoder(8)
        qn = row_normalize(aq); kq = row_normalize(aq.T.tocsr())
        actual = encoder(x, sparse_tensor(qn), sparse_tensor(kq))
        base = encoder.features(x)
        message = torch.tensor(kq.toarray(), dtype=torch.float32) @ (torch.tensor(qn.toarray(), dtype=torch.float32) @ base)
        expected = torch.relu(encoder.self_projection(base)+encoder.query_projection(message)+encoder.bias)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(float(message[3].detach().abs().sum()), 0.)

    def test_streaming_against_full_recomputation(self):
        rng = np.random.default_rng(4)
        for trial in range(8):
            objects = [set(rng.choice(6, size=3, replace=False)) for _ in range(14)]
            queries = [{0}, {1}, {2, 3}, {0, 4}, {4, 5}, {1, 5}]
            a = matrix(objects, 6); q = matrix(queries, 6)
            left, right = [0, 1], [2, 3]
            def routed(ids):
                union = set().union(*(objects[i] for i in ids))
                return np.array([z.issubset(union) for z in queries])
            def total(l, r):
                value = 0.; accesses = routed(l).astype(float)+routed(r).astype(float)
                value += param.weighted_cost(2*6+32*6+64, param.predict_query(2, np.array(list(map(len, queries)), float), accesses).sum())
                for ids in [l, r]:
                    union = set().union(*(objects[i] for i in ids)); active = routed(ids)
                    results = np.array([sum(z.issubset(objects[i]) for i in ids) for z in queries], float)
                    cost = param.predict_query(len(ids), np.array(list(map(len, queries)), float)[active], results[active]).sum()
                    value += param.weighted_cost(len(union)*len(ids)+32*len(union)+32*len(ids), cost)
                return value
            for i in range(4, len(objects)):
                if total(left+[i], right) <= total(left, right+[i]): left.append(i)
                else: right.append(i)
            result = streaming_assign(a, q, [0, 1], [2, 3], [], [],
                                      epsilon=param.EPSILON, alpha=param.ALFA, beta=param.BETA, gamma=param.GAMA)
            self.assertEqual(set(result.obj_left), set(left))
            self.assertEqual(set(result.obj_right), set(right))
            np.testing.assert_array_equal(result.q_left, np.flatnonzero(routed(left)))

    def test_unknown_training_term_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'queries.csv'; path.write_text('Title\nknown unknown\n')
            with self.assertRaisesRegex(ValueError, 'unknown'):
                build_Aqw_from_workload(str(path), {'known': 0})

    def test_dp_records_and_multinode_export(self):
        a = matrix([[0], [0, 1], [1], [1, 2], [2], [2, 3], [3], [0, 3]], 4)
        q = matrix([[0], [1], [2], [3], [0, 1], [1, 3]], 4)
        out = Path('benchmarks/trained/paper-fixture'); out.mkdir(parents=True, exist_ok=True)
        b = TreeBuilder(a, q, out_dir=str(out/'binary'), device='cpu')
        b._node_queries = {}; b.build_context = BuildContext(a, q)
        b._reset_mem_tree_cache()
        leaves = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5]), np.array([6, 7])]
        def build(ids):
            nid = b._new_id(); objects = np.concatenate([leaves[i] for i in ids])
            kw = np.unique(a[objects].indices); qs = np.array([i for i in range(q.shape[0]) if set(q[i].indices).issubset(set(kw))])
            b._node_queries[nid] = qs
            data = dict(omega=len(kw), WN=len(qs), sum_q_len_minus1=int((np.diff(q[qs].indptr)-1).sum()))
            if len(ids)==1:
                obj, keys, bits = make_leaf_keyword_object_bitmaps(a, objects)
                b._cache_leaf_payload(nid, obj, keys, bits)
                cost = node_cost(a, q, objects, qs, context=b.build_context)
                b._meta[nid] = TreeNodeMeta(nid, 0, True, cost, None, None, 'leaf', data)
            else:
                l = build(ids[:len(ids)//2]); r = build(ids[len(ids)//2:])
                keys, bits = build_kw_lr_bitset(b.get_node_keywords_mem(l), b.get_node_keywords_mem(r))
                b._cache_internal_payload(nid, np.array([l, r]), keys, bits[:, None])
                lengths = np.diff(q[qs].indptr).astype(float)
                access = np.isin(qs, b._node_queries[l]).astype(float)+np.isin(qs, b._node_queries[r]).astype(float)
                cost = param.weighted_cost(2*len(kw)+32*len(kw)+64, param.predict_query(2, lengths, access).sum())
                b._meta[nid] = TreeNodeMeta(nid, 0, False, float(cost), None, None, 'internal', data)
            return nid
        root = build([0, 1, 2, 3]); b._root_id = root; b.save_tree()
        breaker = TreeNodeBreaker(b, str(out/'fanout')); info = breaker.run(root)
        self.assertLessEqual(info['cost_after_dp'], info['cost_before_dp']+1e-6)
        for nid, state in breaker._dp.items():
            self.assertEqual(state.RV[1], {int(i): 1 for i in b._node_queries[nid]})
            self.assertTrue(all(state.WS[k] == sum(state.RV[k].values()) for k in range(1, len(state.CT)) if np.isfinite(state.CT[k])))
        import json
        (out/'vocab.json').write_text(json.dumps({'k'+str(i): i for i in range(4)}))
        for name, mat in [('objects.csv', a), ('queries.csv', q)]:
            lines = [' '.join('k'+str(k) for k in mat[i].indices) for i in range(mat.shape[0])]
            if name == 'queries.csv': lines += ['k0 unknown', 'k0 k0']
            (out/name).write_text('Title\n'+'\n'.join(lines)+'\n')


if __name__ == '__main__': unittest.main()
