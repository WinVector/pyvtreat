
import scipy.stats

import vtreat.util

def test_perm_cor():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    c1 = scipy.stats.pearsonr(x, y)
    c2 = vtreat.util.perm_est_correlation(x, y)
    assert abs(c1[0] - c2[0]) < 1e-9
    assert abs(c1[1] - c2[1]) < 0.1
