"""Meaningful protocol and batching checks; no production training needed."""
import unittest

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from data_pipeline import load_data, make_splits, prepare_fold, make_graph, query_neighbors, inductive_predict
from model import GraphSAGE
from select_model import knee_candidate, select_one_se, summarize
import pandas as pd


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.raw, cls.frame, cls.y, cls.numeric = load_data()

    def test_holdout_and_folds_disjoint(self):
        dev, holdout, folds = make_splits(self.y, 42, 5)
        self.assertEqual(len(holdout), 294)
        self.assertFalse(set(dev) & set(holdout))
        self.assertEqual(set(np.concatenate([va for _, va in folds])), set(dev))
        for tr, va in folds:
            self.assertFalse(set(tr) & set(va))
            self.assertFalse(set(tr) & set(holdout))
            self.assertEqual(set(tr) | set(va), set(dev))

    def test_scaler_is_fit_on_train_only(self):
        _, _, folds = make_splits(self.y, 42, 2)
        tr, va = folds[0]
        pre, train, val = prepare_fold(self.frame, self.numeric, tr, va)
        np.testing.assert_allclose(pre.named_transformers_["num"].mean_, self.frame.iloc[tr][self.numeric].mean().to_numpy())
        self.assertEqual(len(train), len(tr))
        self.assertEqual(len(val), len(va))
        self.assertNotIn("EmployeeNumber", pre.get_feature_names_out().tolist())

    def test_disjoint_batch_matches_explicit_app_attachment(self):
        rng = np.random.RandomState(17)
        train, query = rng.normal(size=(24, 6)).astype("float32"), rng.normal(size=(5, 6)).astype("float32")
        edges = make_graph(train, 4)
        torch.manual_seed(42)
        model = GraphSAGE(6, 12, 8, 0.3).eval()
        actual = inductive_predict(model, train, edges, query, 4, batch_size=3)
        expected = []
        for q in query:
            js = np.argsort(cosine_similarity(q[None], train)[0])[::-1][:4]
            attachment = torch.tensor([[24] * 4 + js.tolist(), js.tolist() + [24] * 4])
            ei = torch.cat((edges, attachment), dim=1)
            with torch.no_grad():
                expected.append(torch.sigmoid(model(torch.tensor(np.vstack((train, q))), ei)[24]).item())
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        alone = inductive_predict(model, train, edges, query[:1], 4, batch_size=1)
        np.testing.assert_allclose(alone[0], actual[0], atol=1e-6)

    def test_graph_is_train_only_symmetric_no_self(self):
        x = np.random.RandomState(5).normal(size=(20, 4)).astype("float32")
        ei = make_graph(x, 3).numpy()
        self.assertLess(ei.max(), len(x))
        self.assertTrue(np.all(ei[0] != ei[1]))
        pairs = set(zip(ei[0], ei[1]))
        self.assertTrue(all((b, a) in pairs for a, b in pairs))
        neighbors = query_neighbors(x, x[:2], 3)
        self.assertEqual(neighbors.shape, (2, 3))

    def test_knee_does_not_force_flat_linear_or_noisy(self):
        self.assertIsNone(knee_candidate([1, 2, 3], [0.5, 0.5, 0.5]))
        self.assertIsNone(knee_candidate([1, 2, 3], [0.1, 0.2, 0.3]))
        self.assertIsNone(knee_candidate([1, 2, 3], [0.1, 0.3, 0.2]))
        self.assertEqual(knee_candidate([1, 2, 3, 4], [0.1, 0.8, 0.9, 0.95])["x"], 2)

    def test_seed_averaging_precedes_fold_se(self):
        rows = []
        for fold in range(2):
            for seed in range(3):
                rows.append(dict(stage="k", candidate="a", fold=fold, val_ap=0.5 + fold * 0.1,
                                 val_auc=0.7, k=5, h1=16, h2=8, dropout=0.3, lr=0.01,
                                 weight_decay=0.0, parameters=100, edges=50, seconds=1,
                                 run_id=f"{fold}-{seed}", best_epoch=2))
        summary = summarize(pd.DataFrame(rows))
        self.assertAlmostEqual(summary.iloc[0].se_ap, 0.05)
        chosen, _, _ = select_one_se(summary)
        self.assertEqual(chosen.candidate, "a")


if __name__ == "__main__":
    unittest.main()
