"""
Utility functions for vtreat
"""

import math
import re
import statistics
from typing import Iterable, List, Optional

import hashlib

import numpy
import pandas

import vtreat.stats_utils


def safe_to_numeric_array(x):
    """
    Convert array to numeric. Note, will parse strings (due to numpy)!

    :param x: array to process
    :return: numeric array
    """
    # note will parse strings!
    return numpy.asarray(x, dtype=float)


def can_convert_v_to_numeric(x) -> bool:
    """
    check if non-empty vector can convert to numeric

    :param x:
    :return: True if can convert to numeric, false otherwise (no string parsing).
    """
    x = numpy.asarray(x)
    not_bad = numpy.logical_not(pandas.isnull(x))
    n_not_bad = numpy.sum(not_bad)
    if n_not_bad < 1:
        return True   # All nan/None, can convert to numeric
    try:
        numpy.asarray(
            x[not_bad] + 0, dtype=float
        )  # +0 prevents string parse, leaving strings as strings
        return True
    except TypeError:
        return False


def is_bad(x):
    """
    For numeric vector x, return logical vector of positions that are null, NaN, infinite.

    :param x:
    :return:
    """
    if can_convert_v_to_numeric(x):
        x = safe_to_numeric_array(x)
        return numpy.logical_or(
            pandas.isnull(x), numpy.logical_or(numpy.isnan(x), numpy.isinf(x))
        )
    return pandas.isnull(x)


def numeric_has_range(x) -> bool:
    """
    Check if a numeric vector has numeric range.

    :param x: vector to check
    :return: True if max > min values in vector, else False.
    """
    x = safe_to_numeric_array(x)
    not_bad = numpy.logical_not(is_bad(x))
    n_not_bad = numpy.sum(not_bad)
    if n_not_bad < 2:
        return False
    x = x[not_bad]
    return numpy.max(x) > numpy.min(x)


def summarize_column(x, *, fn=numpy.mean) -> float:
    """
    Summarize column to a non-missing scalar.

    :param x: a vector/Series or column of numbers
    :param fn: summarize function (such as numpy.mean), only passed non-bad positions
    :return: scalar float summary of the non-None positions of x (otherwise 0)
    """

    x = safe_to_numeric_array(x)
    not_bad = numpy.logical_not(is_bad(x))
    n_not_bad = numpy.sum(not_bad)
    if n_not_bad < 1:
        return 0.0
    x = x[not_bad]
    v = 0.0 + fn(x)
    if pandas.isnull(v) or math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def get_unique_value_count(x):
    """compute how many unique values in list-x"""
    if len(x) <= 1:
        return len(x)
    p = pandas.DataFrame({"x": x, "o": 1})
    s = p.groupby("x").sum()  # drops None
    return numpy.max(pandas.isnull(x)) + s.shape[0]


def grouped_by_x_statistics(x, y):
    """compute some grouped by x vector summaries of numeric y vector (no missing values in y)"""
    n = len(x)
    assert n > 0
    assert n == len(y)
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
    if numpy.sum(bad_vars) < len(sf["_var"]):
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


def score_variables(
        cross_frame: pandas.DataFrame,
        *,
        variables: Optional[Iterable[str]],
        outcome,
        is_classification: bool = False) -> Optional[pandas.DataFrame]:
    """
    Score the linear relation of variables to outcome.

    :param cross_frame: data frame to score
    :param variables: names of variables to score
    :param outcome: numeric outcome
    :param is_classification: logical, True if a classification problem (0/1)
    :return: data frame of variable evaluations, None if there are no non-constant variables.
    """

    assert isinstance(cross_frame, pandas.DataFrame)
    if variables is not None:
        assert not isinstance(variables, str)  # prevents str masquerading as iterable
        variables = list(variables)
    else:
        variables = list(cross_frame.columns)
    if len(variables) <= 0:
        return None
    outcome = safe_to_numeric_array(outcome)
    n = cross_frame.shape[0]
    assert n == len(outcome)
    if numpy.max(outcome) <= numpy.min(outcome):
        return None  # y must have range

    def f(v):
        """summarizing fn"""
        col = cross_frame[v]
        if (
            (n > 2)
            and (numpy.max(col) > numpy.min(col))
            and (numpy.max(outcome) > numpy.min(outcome))
        ):
            cor, sig = vtreat.stats_utils.our_corr_score(y_true=outcome, y_pred=col)
            r2 = cor ** 2
            if is_classification:
                r2, sig = vtreat.stats_utils.our_pseudo_R2(y_true=outcome, y_pred=col)
            sfi = pandas.DataFrame(
                {
                    "variable": [v],
                    "has_range": [True],
                    "PearsonR": [cor],
                    "R2": [r2],
                    "significance": [sig],
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

    sf_list = [f(v) for v in variables]
    sf = pandas.concat(sf_list, axis=0, sort=False)
    sf = sf.reset_index(inplace=False, drop=True)
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


def unique_items_in_order(x) -> list:
    """
    Return de-duplicated list of items in order they are in supplied array.

    :param x: vector to inspect
    :return: list
    """
    ret = []
    if x is not None:
        seen = set()
        for item in x:
            if item not in seen:
                ret.append(item)
                seen.add(item)
    return ret


def clean_string(s: str) -> str:
    """
    Replace common symbols with column-name safe alternatives.

    :param s: incoming string
    :return: string
    """
    mp = {
        "<": "_lt_",
        ">": "_gt_",
        "[": "_osq_",
        "]": "_csq_",
        "(": "_op_",
        ")": "_cp_",
        ".": "_",
        " ": "_",
    }
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    for (k, v) in mp.items():
        s = s.replace(k, v)
    return s


def build_level_codes(incoming_column_name: str, levels: Iterable) -> List[str]:
    """
    Pick level names for a set of levels.

    :param incoming_column_name:
    :param levels:
    :return:
    """
    levels = [str(lev) for lev in levels]
    levels = [incoming_column_name + "_lev_" + clean_string(lev) for lev in levels]
    if len(set(levels)) != len(levels):
        levels = [levels[i] + "_" + str(i) for i in range(len(levels))]
    return levels


def hash_data_frame(d: pandas.DataFrame) -> str:
    """
    Get a hash code representing a data frame.

    :param d: data frame
    :return: hash code as a string
    """
    return hashlib.sha256(pandas.util.hash_pandas_object(d).values).hexdigest()



