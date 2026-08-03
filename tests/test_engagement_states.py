from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behavior_analyses.engagement_states import (  # noqa: E402
    build_design_matrix,
    classify_trials,
    session_folds,
    session_log_likelihood,
    session_sequences,
    summarize_feasibility,
)


class EngagementStateContractTests(unittest.TestCase):
    def test_exclusions_are_exclusive_and_history_resets_by_session(self):
        trials = pd.DataFrame(
            {
                "subject_name": ["GRB006"] * 6,
                "session_name": ["s1", "s1", "s1", "s1", "s2", "s2"],
                "dataset_name": ["d1", "d1", "d1", "d1", "d2", "d2"],
                "trial_num": np.arange(6),
                "response": [1, -1, -1, 0, -1, 1],
                "with_choice": [1, 1, 1, 0, 1, 1],
                "early_withdrawal": [0, 1, 0, 0, 0, 0],
                "rewarded": [1, 0, 0, 0, 1, 0],
                "stim_rate_vision": [12, 8, 16, 12, 12, np.nan],
                "category_boundary": [12] * 6,
            }
        )

        classified = classify_trials(trials)
        design = build_design_matrix(classified)
        session, subject = summarize_feasibility(classified)

        self.assertEqual(
            classified["exclusion_reason"].tolist(),
            [
                "valid",
                "early_withdrawal",
                "valid",
                "no_choice",
                "valid",
                "missing_stimulus",
            ],
        )
        self.assertEqual(int(classified["is_boundary"].sum()), 2)
        np.testing.assert_array_equal(design["previous_choice"], [0, 1, 0])
        np.testing.assert_array_equal(design["previous_reward"], [0, 1, 0])
        self.assertEqual(session["n_trials"].sum(), 6)
        self.assertEqual(subject["n_valid"].item(), 3)
        sequences = session_sequences(design)
        self.assertEqual([len(sequence[1]) for sequence in sequences], [2, 1])
        folds = session_folds(sequences, n_splits=2, seed=445)
        self.assertEqual(
            sorted(index for _, test in folds for index in test),
            [0, 1],
        )

    def test_block_likelihood_matches_separate_session_likelihoods(self):
        import jax.random as jr
        from dynamax.hidden_markov_model import CategoricalRegressionHMM

        sequences = [
            (
                "s1",
                np.array([0, 1], dtype=np.int32),
                np.zeros((2, 3), dtype=np.float32),
            ),
            (
                "s2",
                np.array([1, 0, 1], dtype=np.int32),
                np.ones((3, 3), dtype=np.float32),
            ),
        ]
        model = CategoricalRegressionHMM(2, 2, 3)
        params, _ = model.initialize(key=jr.key(445))
        separate = sum(
            float(model.marginal_log_prob(params, choices, inputs))
            for _, choices, inputs in sequences
        )

        blocked = session_log_likelihood(model, params, sequences)

        self.assertAlmostEqual(blocked, separate, places=5)


if __name__ == "__main__":
    unittest.main()
