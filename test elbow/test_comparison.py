"""Validate nested selection, paired reporting and fixed-epoch refit isolation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_pipeline import load_data, make_splits
from comparison_core import (nested_splits, choose_methods, choose_threshold,
                             refit_predict, outer_summary, evaluate_predictions, METHODS)
from comparison_report import complete_outer_folds
from compare_methods import completed_job


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.raw, cls.frame, cls.y, cls.numeric = load_data()

    def test_inner_folds_never_see_outer_validation_or_holdout(self):
        dev, holdout, _ = make_splits(self.y, 42, 3)
        splits = nested_splits(dev, self.y, 3, 2, 42)
        self.assertEqual(set(np.concatenate([va for _, va, _ in splits])), set(dev))
        for tr, va, inner in splits:
            self.assertFalse(set(tr) & set(va))
            self.assertFalse(set(va) & set(holdout))
            self.assertEqual(set(np.concatenate([b for _, b in inner])), set(tr))
            for a, b in inner:
                self.assertEqual(set(a) | set(b), set(tr))
                self.assertFalse(set(a) & set(b))
                self.assertFalse((set(a) | set(b)) & (set(va) | set(holdout)))

    def test_grid_maximum_vs_one_se_smaller_model(self):
        summary = pd.DataFrame([
            dict(candidate="baseline", n_folds=3, mean_ap=.60, se_ap=.02, mean_edges=100, parameters=1000, k=10),
            dict(candidate="big", n_folds=3, mean_ap=.61, se_ap=.02, mean_edges=200, parameters=2000, k=20),
        ])
        chosen, notes = choose_methods(summary, "baseline")
        self.assertEqual(chosen["grid_ap"], "big")
        self.assertEqual(chosen["elbow_one_se"], "baseline")
        self.assertIsNone(notes["knee_k"])
        self.assertAlmostEqual(notes["cutoff"], .59)

    def test_threshold_averages_seed_predictions_before_f1(self):
        oof = pd.DataFrame(dict(source_index=[1, 2, 3, 4] * 2, actual=[0, 0, 1, 1] * 2,
                                probability=[.1, .2, .8, .9, .1, .4, .6, .9]))
        threshold = choose_threshold(oof)
        self.assertGreater(threshold, .30)
        self.assertLessEqual(threshold, .70)
        conflict = oof.copy()
        conflict.loc[4, "actual"] = 1
        with self.assertRaises(ValueError):
            choose_threshold(conflict)

    def test_refit_parameters_unchanged_when_only_eval_features_change(self):
        tr = np.arange(80)
        va = np.arange(80, 85)
        cfg = dict(k=3, h1=8, h2=4, dropout=.3, lr=.01, weight_decay=5e-5)
        options = dict(inference_batch_size=2)
        labels = self.y[tr]
        _, _, first, pre = refit_predict(cfg, options, self.frame, self.numeric, tr, va, labels, 42, 2)
        changed = self.frame.copy()
        changed.loc[va, "MonthlyIncome"] = 999999
        _, _, second, _ = refit_predict(cfg, options, changed, self.numeric, tr, va, labels, 42, 2)
        self.assertEqual(first["epochs"], 2)
        for name, tensor in first["state_dict"].items():
            torch.testing.assert_close(tensor, second["state_dict"][name], atol=0, rtol=0)
        np.testing.assert_allclose(pre.named_transformers_["num"].mean_, self.frame.iloc[tr][self.numeric].mean())
        with self.assertRaises(ValueError):
            refit_predict(cfg, options, self.frame, self.numeric, tr, va, self.y, 42, 2)

    def test_only_complete_paired_folds_reported(self):
        records = []
        for fold in range(2):
            for method in METHODS:
                for seed in [42, 123]:
                    if fold == 1 and method == "grid_ap" and seed == 123:
                        continue
                    records.append(dict(result_id=f"{fold}-{method}-{seed}", outer_fold=fold, method=method, seed=seed))
        complete = complete_outer_folds(pd.DataFrame(records), {"evaluation_seeds": [42, 123]})
        self.assertEqual(set(complete.outer_fold), {0})
        self.assertEqual(len(complete), 6)

    def test_outer_metrics_and_completion_marker(self):
        result = evaluate_predictions(np.array([0, 0, 1, 1]), np.array([.1, .8, .2, .9]), .5)
        self.assertEqual([result[k] for k in ("tn", "fp", "fn", "tp")], [1, 1, 1, 1])
        with tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()) / "opencode") as tmp:
            path = Path(tmp)
            self.assertIsNone(completed_job(path))
            (path / "result.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                completed_job(path)

    def test_outer_summary_averages_seed_repeats(self):
        records = []
        for fold in [0, 1]:
            for seed in [42, 123, 2026]:
                row = dict(method="grid_ap", outer_fold=fold, ap=.5 + .1 * fold)
                row.update({key: 1 for key in ("auc", "f1", "precision", "recall", "accuracy", "parameters", "edges", "train_seconds", "inference_ms_per_employee", "search_seconds")})
                records.append(row)
        summary, folds = outer_summary(pd.DataFrame(records))
        self.assertEqual(len(folds), 2)
        self.assertEqual(summary.iloc[0].n_outer_folds, 2)
        self.assertAlmostEqual(summary.iloc[0].mean_ap, .55)
        self.assertAlmostEqual(summary.iloc[0].sd_ap, np.std([.5, .6], ddof=1))


if __name__ == "__main__":
    unittest.main()
