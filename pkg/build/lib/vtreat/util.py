# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:40:41 2019

@author: johnmount
"""

import warnings
import math
import statistics

import numpy
import pandas


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import statsmodels.api
    import statsmodels.formula.api


def safe_to_numeric_array(x):
    # work around https://github.com/WinVector/pyvtreat/issues/7
    # noinspection PyTypeChecker
    return numpy.asarray(pandas.Series(x) + 0.0, dtype=float)


def can_convert_v_to_numeric(x):
    """check if non-empty vector can convert to numeric"""
    try:
        numpy.asarray(x + 0.0, dtype=float)
        return True
    except TypeError:
        return False


def is_bad(x):
    """ for numeric vector x, return logical vector of positions that are null, NaN, infinite"""
    if can_convert_v_to_numeric(x):
        x = safe_to_numeric_array(x)
        return numpy.logical_or(
            pandas.isnull(x), numpy.logical_or(numpy.isnan(x), numpy.isinf(x))
        )
    return pandas.isnull(x)


def has_range(x):
    x = safe_to_numeric_array(x)
    not_bad = numpy.logical_not(is_bad(x))
    n_not_bad = sum(not_bad)
    if n_not_bad < 2:
        return False
    x = x[not_bad]
    return numpy.max(x) > numpy.min(x)


def summarize_column(x, *, fn=numpy.mean):
    """
    Summarize column to a non-missing scalar.

    :param x: a vector/Series or column of numbers
    :param fn: summarize function (such as numpy.mean), only passed non-bad positions
    :return: scalar float summary of the non-None positions of x (otherwise 0)
    """

    x = safe_to_numeric_array(x)
    not_bad = numpy.logical_not(is_bad(x))
    n_not_bad = sum(not_bad)
    if n_not_bad < 1:
        return 0.0
    x = x[not_bad]
    v = 0.0 + fn(x)
    if pandas.isnull(v) or math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def characterize_numeric(x):
    """compute na count, min,max,mean of a numeric vector"""

    x = safe_to_numeric_array(x)
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
    y = safe_to_numeric_array(y)
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


def our_corr_score(*, y_true, y_pred):
    # compute Pearson correlation
    y_true = numpy.asarray(y_true)
    y_pred = numpy.asarray(y_pred)
    y_pred = y_pred - numpy.mean(y_pred)
    y_pred = y_pred / math.sqrt(numpy.dot(y_pred, y_pred))
    y_true = y_true - numpy.mean(y_true)
    y_true = y_true / math.sqrt(numpy.dot(y_true, y_true))
    return numpy.dot(y_pred, y_true)


# noinspection PyPep8Naming
def linear_R2_with_sig(*,  y_true, y_pred):
    y_true = numpy.asarray(y_true)
    y_pred = numpy.asarray(y_pred)
    if len(y_true) < 2:
        return 0, 1
    if numpy.min(y_true) >= numpy.max(y_true):
        return 0, 1
    if numpy.min(y_pred) >= numpy.max(y_pred):
        return 0, 1
    fit = statsmodels.api.OLS(
        y_true,
        pandas.DataFrame({'x': y_pred, 'c': 1})).fit(disp=0)
    return fit.rsquared, fit.f_pvalue  # use the f-test sig


# noinspection PyPep8Naming
def pseudo_R2_with_sig(*,  y_true, y_pred):
    y_true = numpy.asarray(y_true)
    y_pred = numpy.asarray(y_pred)
    if len(y_true) < 2:
        return 0, 1
    if numpy.min(y_true) >= numpy.max(y_true):
        return 0, 1
    if numpy.min(y_pred) >= numpy.max(y_pred):
        return 0, 1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # noinspection PyBroadException
        try:
            fit = statsmodels.api.Logit(
                y_true,
                pandas.DataFrame({'x': y_pred, 'c': 1})).fit(disp=0)
            return fit.prsquared, fit.llr_pvalue  # use the log-likelihood sig
        except Exception:
            pass
    # separated use linear model as a stand-in
    return linear_R2_with_sig(y_true=y_true, y_pred=y_pred)
    # # smoothing approach, very downward biased on small examples
    # p = numpy.mean(y_true)
    # if (p <= 0) or (p >= 1):
    #     return 1, 1
    # eps = min(p/2, (1-p)/2)
    # y_true = numpy.append(
    #     y_true,
    #     [p - eps, p + eps, p - eps, p + eps])  # smoothing/regularization
    #
    # y_pred = numpy.append(
    #     y_pred,
    #     [0, 0, 1, 1])  # smoothing/regularization
    # fit = statsmodels.api.Logit(
    #      y_true,
    #      pandas.DataFrame({'x': y_pred, 'c': 1})).fit(disp=0)
    # return fit.prsquared, fit.llr_pvalue  # use the log-likelihood sig


def score_variables(cross_frame, variables, outcome,
                    *,
                    is_classification=False):
    """score the linear relation of variables to outcome"""

    if len(variables) <= 0:
        return None
    n = cross_frame.shape[0]
    if n != len(outcome):
        raise ValueError("len(n) must equal cross_frame.shape[0]")
    outcome = safe_to_numeric_array(outcome)

    def f(v):
        col = cross_frame[v]
        col = safe_to_numeric_array(col)
        if (n > 2) and \
                (numpy.max(col) > numpy.min(col)) and \
                (numpy.max(outcome) > numpy.min(outcome)):
            cor = our_corr_score(
                y_true=outcome,
                y_pred=col)
            if is_classification:
                est = pseudo_R2_with_sig(y_true=outcome, y_pred=col)
            else:
                est = linear_R2_with_sig(y_true=outcome, y_pred=col)
            sfi = pandas.DataFrame(
                {
                    "variable": [v],
                    "has_range": [True],
                    "PearsonR": cor,
                    "R2": [est[0]],
                    "significance": [est[1]],
                }
            )
        else:
            sfi = pandas.DataFrame(
                {
                    "variable": [v],
                    "has_range": [False],
                    "PearsonR": [numpy.NaN],
                    "R2": [numpy.NaN],
                    "significance": [1.0],
                }
            )
        return sfi

    sf = [f(v) for v in variables]
    if len(sf) <= 0:
        return None
    sf = pandas.concat(sf, axis=0, sort=False)
    sf.reset_index(inplace=True, drop=True)
    return sf


def check_matching_numeric_frames(*, res, expect, tol=1.0e-4):
    """
    Check if two numeric pandas.DataFrame s are identical.  assert if not
    :param res:
    :param expect:
    :param tol: numeric tolerance.
    :return: None
    """
    assert isinstance(expect, pandas.DataFrame)
    assert isinstance(res, pandas.DataFrame)
    assert res.shape == expect.shape
    for c in expect.columns:
        ec = expect[c]
        rc = res[c]
        assert numpy.max(numpy.abs(ec - rc)) <= tol


def unique_itmes_in_order(lst):
    ret = []
    if lst is not None:
        seen = set()
        for item in lst:
            if item not in seen:
                ret.append(item)
                seen.add(item)
    return ret


def clean_string(strng):
    mp = {'<': '_lt_',
          '>': '_gt_',
          '[': '_osq_',
          ']': '_csq_',
          '(': '_op_',
          ')': '_cp_',
          '.': '_',
          }
    for (k, v) in mp.items():
        strng = strng.replace(k, v)
    return strng


def build_level_codes(incoming_column_name, levels):
    levels = [str(lev) for lev in levels]
    levels = [incoming_column_name + "_lev_" + clean_string(lev) for lev in levels]
    if len(set(levels)) != len(levels):
        levels = [levels[i] + "_" + str(i) for i in range(len(levels))]
    return levels
