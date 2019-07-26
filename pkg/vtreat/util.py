# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:40:41 2019

@author: johnmount
"""


import numpy
import warnings

import pandas
import scipy.stats
import statistics


def k_way_cross_plan(n_rows, k_folds):
    """randomly split range(n_rows) into k_folds disjoint groups"""
    if k_folds >= n_rows:
        k_folds = n_rows - 1
    if n_rows <= 1 or k_folds <= 1 or k_folds>=n_rows/2:
        # degenerate overlap cases
        plan = [
            {
                "train": [i for i in range(n_rows)],
                "app": [i for i in range(n_rows)],
            }
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


def grouped_by_x_statistics(x, y):
    """compute some grouped by x vector summaries of numeric y vector (no missing values in y)"""
    eps = 1.0e-3
    sf = pandas.DataFrame({"x": x, "y": y})
    sf.reset_index(inplace=True, drop=True)
    na_posns = sf["x"].isnull()
    sf.loc[na_posns, "x"] = "_NA_"
    global_mean = sf["y"].mean()
    sf["_group_mean"] = sf.groupby("x")["y"].transform("mean")
    sf["_var"] = (sf["y"] - sf["_group_mean"]) ** 2
    sf["_ni"] = 1
    sf = sf.groupby("x").sum()
    sf.reset_index(inplace=True, drop=False)
    sf["y"] = sf["y"] / sf["_ni"]
    sf["_group_mean"] = sf["_group_mean"] / sf["_ni"]
    sf["_var"] = sf["_var"] / (sf["_ni"] - 1) + eps
    avg_var = 0
    if sum(sf["_var"].isnull())<len(sf["_var"]):
        avg_var = numpy.nanmean(sf["_var"])
    sf.loc[sf["_var"].isnull(), "_var"] = avg_var
    sf["_vb"] = statistics.variance(sf["_group_mean"]) + eps
    sf["_gm"] = global_mean
    # heirarchical model is in:
    # http://www.win-vector.com/blog/2017/09/partial-pooling-for-lower-variance-variable-encoding/
    # using naive empirical estimates of variances
    # adjusted from ni to ni-1 and +eps variance to make
    # rare levels look like new levels.
    sf["_hest"] = (
        (sf["_ni"] - 1) * sf["_group_mean"] / sf["_var"] + sf["_gm"] / sf["_vb"]
    ) / ((sf["_ni"] - 1) / sf["_var"] + 1 / sf["_vb"])
    return sf


def score_variables(cross_frame, variables, outcome):
    """score the linear relation of varaibles to outcomename"""

    def f(v):
        col = cross_frame[v]
        if numpy.max(col) > numpy.min(col):
            with warnings.catch_warnings():
                est = scipy.stats.pearsonr(cross_frame[v], outcome)
                sfi = pandas.DataFrame(
                    {"variable": [v], "has_range": True, "PearsonR": est[0], "significance": est[1]}
                )
        else:
            sfi = pandas.DataFrame(
                {"variable": [v], "has_range": False, "PearsonR": numpy.NaN, "significance": 1}
            )
        return sfi

    sf = [f(v) for v in variables]
    sf = pandas.concat(sf, axis=0)
    sf.reset_index(inplace=True, drop=True)
    return sf
