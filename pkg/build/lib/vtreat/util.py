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


def can_convert_v_to_numeric(x):
    """check if non-empty vector can convert to numeric"""
    try:
        numpy.asarray(x + 0, dtype=float)
        return True
    except TypeError:
        return False


def is_bad(x):
    """ for numeric vector x, return logical vector of positions that are null, NaN, infinite"""
    if can_convert_v_to_numeric(x):
        x = numpy.asarray(x + 0, dtype=float)
        return numpy.logical_or(
            pandas.isnull(x), numpy.logical_or(numpy.isnan(x), numpy.isinf(x))
        )
    return pandas.isnull(x)


def has_range(x):
    x = numpy.asarray(pandas.Series(x)).astype(float)
    return numpy.max(x) > numpy.min(x)


def characterize_numeric(x):
    """compute na count, min,max,mean of a numeric vector"""
    x = numpy.asarray(pandas.Series(x)).astype(float)
    not_bad = numpy.logical_not(is_bad(x))
    n_not_bad = sum(not_bad)
    n = len(x)
    if n_not_bad <= 0:
        return {
            "n": n,
            "n_not_bad": n_not_bad,
            "min": None,
            "mean": None,
            "max": None,
            "varies": False,
            "has_range": False,
        }
    x = x[not_bad]
    mn = numpy.min(x)
    mx = numpy.max(x)
    return {
        "n": n,
        "n_not_bad": n_not_bad,
        "min": mn,
        "mean": numpy.mean(x),
        "max": mx,
        "varies": (mx > mn) or ((n_not_bad > 0) and (n_not_bad < n)),
        "has_range": (mx > mn),
    }


def grouped_by_x_statistics(x, y):
    """compute some grouped by x vector summaries of numeric y vector (no missing values in y)"""
    n = len(x)
    if n <= 0:
        raise ValueError("no rows")
    if n != len(y):
        raise ValueError("len(y)!=len(x)")
    eps = 1.0e-3
    sf = pandas.DataFrame({"x": x, "y": y})
    sf.reset_index(inplace=True, drop=True)
    bad_posns = pandas.isnull(sf["x"])
    sf.loc[bad_posns, "x"] = "_NA_"
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
    bad_vars = is_bad(sf["_var"])
    if sum(bad_vars) < len(sf["_var"]):
        avg_var = numpy.nanmean(sf["_var"])
    sf.loc[bad_vars, "_var"] = avg_var
    if sf.shape[0] > 1:
        sf["_vb"] = statistics.variance(sf["_group_mean"]) + eps
    else:
        sf["_vb"] = eps
    sf["_gm"] = global_mean
    # hierarchical model is in:
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

    if len(variables) <= 0:
        return None
    n = cross_frame.shape[0]
    if n != len(outcome):
        raise ValueError("len(n) must equal cross_frame.shape[0]")

    def f(v):
        col = cross_frame[v]
        col = numpy.asarray(col)
        if n > 0 and numpy.max(col) > numpy.min(col):
            with warnings.catch_warnings():
                est = scipy.stats.pearsonr(cross_frame[v], outcome)
                sfi = pandas.DataFrame(
                    {
                        "variable": [v],
                        "has_range": True,
                        "PearsonR": est[0],
                        "significance": est[1],
                    }
                )
        else:
            sfi = pandas.DataFrame(
                {
                    "variable": [v],
                    "has_range": False,
                    "PearsonR": numpy.NaN,
                    "significance": 1,
                }
            )
        return sfi

    sf = [f(v) for v in variables]
    if len(sf) <= 0:
        return None
    sf = pandas.concat(sf, axis=0, sort=False)
    sf.reset_index(inplace=True, drop=True)
    return sf
