
import numpy


def k_way_cross_plan(n_rows, k_folds):
    """randomly split range(n_rows) into k_folds disjoint groups"""
    n2 = int(numpy.floor(n_rows / 2))
    if k_folds > n2:
        k_folds = n2
    if n_rows <= 1 or k_folds <= 1:
        # degenerate overlap cases
        plan = [
            {"train": [i for i in range(n_rows)], "app": [i for i in range(n_rows)]}
        ]
        return plan
    # first assign groups modulo k (ensuring at least one in each group)
    grp = [i % k_folds for i in range(n_rows)]
    # now shuffle
    numpy.random.shuffle(grp)
    plan = [
        {
            "train": [i for i in range(n_rows) if grp[i] != j],
            "app": [i for i in range(n_rows) if grp[i] == j],
        }
        for j in range(k_folds)
    ]
    return plan


def support_indicator(n_rows, cross_plan):
    """return a vector indicating which rows had app assignments"""
    support = numpy.full(n_rows, False, dtype=bool)
    for ci in cross_plan:
        support[ci["app"]] = True
    return support


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
        return k_way_cross_plan(n_rows=n_rows, k_folds=k_folds)
