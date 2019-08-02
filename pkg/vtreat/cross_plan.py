
import numpy
import numpy.random
import pandas


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


class KWayCrossPlan(CrossValidationPlan):
    """K-way cross validation plan"""

    def __init__(self):
        CrossValidationPlan.__init__(self)

    def split_plan(self, *, n_rows=None, k_folds=None, data=None, y=None):
        return k_way_cross_plan(n_rows=n_rows, k_folds=k_folds)


def k_way_cross_plan_y_stratified(n_rows, k_folds, y):
    """randomly split range(n_rows) into k_folds disjoint groups, attempting an even y-distribution"""
    n2 = int(numpy.floor(n_rows / 2))
    if k_folds > n2:
        k_folds = n2
    if n_rows <= 1 or k_folds <= 1:
        # degenerate overlap cases
        plan = [
            {"train": [i for i in range(n_rows)], "app": [i for i in range(n_rows)]}
        ]
        return plan
    # first sort by y plus a random key
    d = pandas.DataFrame({'y':y,
                          'i':[i for i in range(n_rows)],
                          'r':numpy.random.uniform(size=n_rows)})
    d.sort_values(by = ['y', 'r'], inplace=True)
    d.reset_index(inplace=True, drop=True)
    # assign y-blocks to lose fine details of y
    fold_size = n_rows/k_folds
    d['block'] = [numpy.floor(i/fold_size) for i in range(n_rows)]
    d.sort_values(by=['block', 'r'], inplace=True)
    d.reset_index(inplace=True, drop=True)
    # now assign groups modulo k (ensuring at least one in each group)
    d['grp'] = [i % k_folds for i in range(n_rows)]
    d.sort_values(by=['i'], inplace=True)
    d.reset_index(inplace=True, drop=True)
    grp = numpy.asarray(d['grp'])
    plan = [
        {
            "train": [i for i in range(n_rows) if grp[i] != j],
            "app": [i for i in range(n_rows) if grp[i] == j],
        }
        for j in range(k_folds)
    ]
    return plan


class KWayCrossPlanYStratified(CrossValidationPlan):
    """K-way cross validation plan, attempting an even y-distribution"""

    def __init__(self):
        CrossValidationPlan.__init__(self)

    def split_plan(self, *, n_rows=None, k_folds=None, data=None, y=None):
        return k_way_cross_plan_y_stratified(n_rows=n_rows, k_folds=k_folds, y=y)


class OrderedCrosPlan(CrossValidationPlan):
    """ordered cross-validation plan"""

    def __init__(self, order_column_name):
        CrossValidationPlan.__init__(self)
        self.order_column_name_ = order_column_name

    def split_plan(self, *, n_rows=None, k_folds=None, data=None, y=None):
        n_rows = data.shape[0]
        order_series = data[self.order_column_name_]
        order_values = numpy.sort(numpy.unique(order_series))
        mid = len(order_values)/2
        ov_left = [order_values[i] for i in range(len(order_values)) if i<mid]
        ov_right = [order_values[i] for i in range(len(order_values)) if i>=mid]
        if len(ov_left)<1 or len(ov_right)<1:
            # degenerate case, fall back to simple method
            return k_way_cross_plan(n_rows=n_rows, k_folds=5)
        plan = [
            {
                "train": [i for i in range(n_rows) if order_series[i]>ov],
                "app": [i for i in range(n_rows) if order_series[i]==ov],
            }
            for ov in ov_left
        ] + [
            {
                "train": [i for i in range(n_rows) if order_series[i]<ov],
                "app": [i for i in range(n_rows) if order_series[i]==ov],
            }
            for ov in ov_right
        ]
        return plan

