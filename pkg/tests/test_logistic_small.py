
import numpy

import vtreat.stats_utils


def test_logistic_small():
    preds = vtreat.stats_utils.sklearn_solve_logistic(x=[1, 2, 3], y=[1, 0, 1])
    assert numpy.max(numpy.abs(preds - 2/3)) < 1e-3
