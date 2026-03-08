"""Tests for hungarian_match()."""

import numpy as np
import pytest

from covid_ndd_extraction.utils.matching import hungarian_match


class TestHungarianMatch:
    def test_2x2_optimal_assignment(self):
        # Pred 0 best matches Gold 1 (sim=0.9)
        # Pred 1 best matches Gold 0 (sim=0.8)
        sim = np.array([[0.5, 0.9], [0.8, 0.3]])
        row_ind, col_ind = hungarian_match(sim)
        pairs = dict(zip(row_ind.tolist(), col_ind.tolist()))
        assert pairs[0] == 1
        assert pairs[1] == 0

    def test_identity_matrix(self):
        """Perfect diagonal matching: pred_i should match gold_i."""
        sim = np.eye(3)
        row_ind, col_ind = hungarian_match(sim)
        for r, c in zip(row_ind, col_ind):
            assert r == c

    def test_returns_numpy_arrays(self):
        sim = np.array([[0.7, 0.3], [0.4, 0.9]])
        row_ind, col_ind = hungarian_match(sim)
        assert isinstance(row_ind, np.ndarray)
        assert isinstance(col_ind, np.ndarray)

    def test_output_lengths_match_min_dim(self):
        """For a non-square matrix the assignment length = min(rows, cols)."""
        sim = np.array([[0.5, 0.9, 0.1], [0.8, 0.3, 0.6]])
        row_ind, col_ind = hungarian_match(sim)
        assert len(row_ind) == len(col_ind)
        assert len(row_ind) == 2  # min(2, 3) rows

    def test_1x1_matrix(self):
        sim = np.array([[0.85]])
        row_ind, col_ind = hungarian_match(sim)
        assert row_ind[0] == 0
        assert col_ind[0] == 0

    def test_all_equal_similarities(self):
        """All sims equal — assignment is still valid (any 1-to-1 mapping works)."""
        sim = np.full((3, 3), 0.7)
        row_ind, col_ind = hungarian_match(sim)
        # Each row index appears at most once; each col index appears at most once
        assert len(set(row_ind)) == len(row_ind)
        assert len(set(col_ind)) == len(col_ind)
