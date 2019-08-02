
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


def order_cross_plan(k_folds, order_vector):
    """Build a k_folds cross validation plan based on the ordered series"""

    n_rows = len(order_vector)
    # see if we can build about k intervals of the order values
    order_vector = numpy.asarray(order_vector)
    order_values = numpy.asarray(numpy.sort(numpy.unique(order_vector)))
    nv = len(order_values)
    if k_folds>nv:
        k_folds = nv
    group_size = n_rows/k_folds
    group_frame = pandas.DataFrame({
        'v':order_values,
        'g':[numpy.floor(i/group_size) for i in range(nv)]
    })
    groups = numpy.asarray(numpy.sort(numpy.unique(group_frame['g'])))
    n_groups = len(groups)
    if n_groups <= 1:
        # degenerate case, fall back to simple method
        return k_way_cross_plan(n_rows=n_rows, k_folds=k_folds)
    left_sets = [set(group_frame['v'][group_frame['g'] < g]) for g in groups]
    match_sets = [set(group_frame['v'][group_frame['g'] == g]) for g in groups]
    right_sets = [set(group_frame['v'][group_frame['g'] > g]) for g in groups]
    mid = (n_groups - 1) / 2
    idx_left = [i for i in range(n_groups) if i < mid]
    idx_right = [i for i in range(n_groups) if i >= mid]
    plan = [
               { # train using future data
                   "train": [i for i in range(n_rows) if order_vector[i] in right_sets[idx]],
                   "app": [i for i in range(n_rows) if order_vector[i] in match_sets[idx]],
               }
               for idx in idx_left
           ] + [
               { # train using past data
                   "train": [i for i in range(n_rows) if order_vector[i] in left_sets[idx]],
                   "app": [i for i in range(n_rows) if order_vector[i] in match_sets[idx]],
               }
               for idx in idx_right
           ]
    return plan


class OrderedCrossPlan(CrossValidationPlan):
    """ordered cross-validation plan"""

    def __init__(self, order_column_name):
        CrossValidationPlan.__init__(self)
        self.order_column_name_ = order_column_name

    def split_plan(self, *, n_rows=None, k_folds=None, data=None, y=None):
        order_vector = data[self.order_column_name_]
        return order_cross_plan(k_folds=k_folds, order_vector=order_vector)