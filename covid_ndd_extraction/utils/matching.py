"""Hungarian algorithm wrapper for optimal triple matching."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_match(sim_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row and column indices of the optimal one-to-one assignment.

    Parameters
    ----------
    sim_matrix:
        2-D similarity matrix (higher = better).  Shape (n_pred, n_gold).

    Returns
    -------
    row_ind, col_ind:
        Index arrays such that sim_matrix[row_ind[k], col_ind[k]] is the
        k-th matched pair.
    """
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind
