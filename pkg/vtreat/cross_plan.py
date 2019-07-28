

import vtreat.util


class CrossValidationPlan:
    """Data splitting plan"""

    def __init__(self):
        pass

    def split_plan(self, *, n_rows=None, k_folds=None, data=None, y=None):
        raise Exception("base class called")


class KWayCrossPlan(CrossValidationPlan):
    """K-way cross validation plan"""

    def __init__(self):
        CrossValidationPlan.__init__(self)

    def split_plan(self, *, n_rows=None, k_folds=None, data=None, y=None):
        return vtreat.util.k_way_cross_plan(n_rows=n_rows, k_folds=k_folds)
