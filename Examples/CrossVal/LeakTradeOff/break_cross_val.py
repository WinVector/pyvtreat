from itertools import compress

import numpy
import pandas

from sklearn.base import BaseEstimator, TransformerMixin

from data_algebra.data_ops import *


def mk_data(*, nrow, n_noise_var=0, n_signal_var=0, n_noise_level=1000):
    # combination of high-complexity useless variables
    # and low-complexity useful variables
    y = numpy.random.normal(size=nrow)
    d = pandas.DataFrame({"const_col": ["a"] * nrow})
    noise_levels = ["nl_" + str(j) for j in range(n_noise_level)]
    for i in range(n_noise_var):
        d["noise_" + str(i)] = numpy.random.choice(
            noise_levels, replace=True, size=nrow
        )
    signal_levels = {"a": 1, "b": -1}
    for i in range(n_signal_var):
        v = "signal_" + str(i)
        d[v] = numpy.random.choice(
            [k for k in signal_levels.keys()], replace=True, size=nrow
        )
        vn = d[v].map(signal_levels)
        y = y + vn
    return d, y


# noinspection PyPep8Naming,PyUnusedLocal
class TransformerAdapter(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        return self.transform(X)

    def fit_predict(self, X, y=None, **fit_params):
        return self.fit_transform(X, y)


# https://github.com/WinVector/data_algebra/blob/master/Examples/cdata/ranking_pivot_example.md
class Container:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return self.value.__repr__()

    def __str__(self):
        return self.value.__repr__()

    def update(self, other):
        if not isinstance(other, Container):
            return self
        return Container(sorted([vi for vi in set(self.value).union(other.value)]))


def solve_for_partition(d_original, d_coded):
    def sorted_concat(vals):
        return Container(sorted([vi for vi in set(vals)]))

    def combine_containers(lcv, rcv):
        return [lft.update(rgt) for lft, rgt in zip(lcv, rcv)]

    nrow = d_original.shape[0]
    ncol = d_original.shape[1]
    pairs = pandas.DataFrame({"idx": range(nrow), "complement": [Container([])] * nrow})
    for j in range(ncol):
        dj = pandas.DataFrame(
            {
                "orig": d_original.iloc[:, j],
                "coded": d_coded.iloc[:, j],
                "idx": range(nrow),
            }
        )
        ops_collect = (
            describe_table(dj, table_name="dj")
            .rename_columns({"coded_left": "coded", "idx_left": "idx"})
            .natural_join(
                b=describe_table(dj, table_name="dj"), jointype="full", by=["orig"]
            )
            .select_rows("(coded_left - coded).abs() > 1.0e-5")
            .project(
                {"complement": user_fn(sorted_concat, "idx")}, group_by=["idx_left"]
            )
            .rename_columns({"idx": "idx_left"})
        )
        pairsj = ops_collect.transform(dj)
        ops_join = (
            describe_table(pairs, table_name="pairs")
            .natural_join(
                b=describe_table(pairsj, table_name="pairsj").rename_columns(
                    {"c_right": "complement"}
                ),
                jointype="left",
                by=["idx"],
            )
            .extend(
                {"complement": user_fn(combine_containers, ["complement", "c_right"])}
            )
            .drop_columns("c_right")
        )
        pairs = ops_join.eval({"pairs": pairs, "pairsj": pairsj})
    return pairs


def collect_relations(*, d_original, d_coded, d_partition, est_fn, y_check=None):
    nrow = d_original.shape[0]
    ncol = d_original.shape[1]
    relns_x = []
    relns_y = []
    for j in range(ncol):
        col_j = d_original.iloc[:, j]
        values_j = [v for v in set(col_j)]
        for v in values_j:
            positions = set([i for i in compress(range(nrow), col_j == v)])
            for p in positions:
                partition_indexes = d_partition["complement"][p].value
                value_indexes = [i for i in positions.intersection(partition_indexes)]
                wts = est_fn(
                    nrow=nrow,
                    partition_indexes=partition_indexes,
                    value_indexes=value_indexes,
                )
                if wts is not None and numpy.sum(numpy.abs(wts)) > 1.0e-7:
                    # should have d_coded.iloc[p, j] == numpy.dot(wts, y_check)
                    value = d_coded.iloc[p, j]
                    relns_x.append(wts)
                    relns_y.append([value, p, j, v])
                    if y_check is not None:
                        check = numpy.dot(wts, y_check)
                        if numpy.abs(value - check) > 1.0e-3:
                            raise ValueError(
                                "check failed j: " + str(j) + ", p: " + str(p)
                            )
    relns_y = pandas.DataFrame(relns_y)
    relns_y.columns = ["code", "i", "j", "level"]
    return pandas.DataFrame(relns_x), relns_y

