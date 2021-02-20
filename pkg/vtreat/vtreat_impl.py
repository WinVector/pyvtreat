# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:07:57 2019

@author: johnmount
"""

from abc import ABC
import math
import pprint
import warnings

import numpy
import pandas

import vtreat.util
import vtreat.transform


def ready_data_frame(d):
    orig_type = type(d)
    if orig_type == numpy.ndarray:
        d = pandas.DataFrame(d)
        d.columns = [str(c) for c in d.columns]
    if not isinstance(d, pandas.DataFrame):
        raise TypeError("not prepared to process type " + str(orig_type))
    return d, orig_type


def back_to_orig_type_data_frame(d, orig_type):
    if not isinstance(d, pandas.DataFrame):
        raise TypeError("Expected result to be a pandas.DataFrame, found: " + str(type(d)))
    columns = [c for c in d.columns]
    if orig_type == numpy.ndarray:
        d = numpy.asarray(d)
    return d, columns


class VarTransform:
    def __init__(self, incoming_column_name, derived_column_names, treatment):
        self.incoming_column_name_ = incoming_column_name
        self.derived_column_names_ = derived_column_names.copy()
        self.treatment_ = treatment
        self.need_cross_treatment_ = False
        self.refitter_ = None

    def transform(self, data_frame):
        raise NotImplementedError("base method called")


class MappedCodeTransform(VarTransform):
    def __init__(self, incoming_column_name, derived_column_name, treatment, code_book):
        VarTransform.__init__(
            self, incoming_column_name, [derived_column_name], treatment
        )
        self.code_book_ = code_book

    def transform(self, data_frame):
        incoming_column_name = self.incoming_column_name_
        derived_column_name = self.derived_column_names_[0]
        sf = pandas.DataFrame({incoming_column_name: data_frame[incoming_column_name]})
        bad_posns = vtreat.util.is_bad(sf[incoming_column_name])
        sf.loc[bad_posns, incoming_column_name] = "_NA_"
        res = pandas.merge(
            sf, self.code_book_, on=[self.incoming_column_name_], how="left", sort=False
        )  # ordered by left table rows
        # could also try pandas .map()
        res = res[[derived_column_name]].copy()
        res.loc[vtreat.util.is_bad(res[derived_column_name]), derived_column_name] = 0
        return res


class YAwareMappedCodeTransform(MappedCodeTransform):
    def __init__(
        self,
        incoming_column_name,
        derived_column_name,
        treatment,
        code_book,
        refitter,
        extra_args,
        params,
    ):
        MappedCodeTransform.__init__(
            self,
            incoming_column_name=incoming_column_name,
            derived_column_name=derived_column_name,
            treatment=treatment,
            code_book=code_book,
        )
        self.need_cross_treatment_ = True
        self.refitter_ = refitter
        self.extra_args_ = extra_args
        self.params_ = params


class CleanNumericTransform(VarTransform):
    def __init__(self, incoming_column_name, replacement_value):
        VarTransform.__init__(
            self, incoming_column_name, [incoming_column_name], "clean_copy"
        )
        self.replacement_value_ = replacement_value

    def transform(self, data_frame):
        col = vtreat.util.safe_to_numeric_array(data_frame[self.incoming_column_name_])
        bad_posns = vtreat.util.is_bad(col)
        col[bad_posns] = self.replacement_value_
        res = pandas.DataFrame({self.derived_column_names_[0]: col})
        return res


class IndicateMissingTransform(VarTransform):
    def __init__(self, incoming_column_name, derived_column_name):
        VarTransform.__init__(
            self, incoming_column_name, [derived_column_name], "missing_indicator"
        )

    def transform(self, data_frame):
        col = vtreat.util.is_bad(data_frame[self.incoming_column_name_])
        res = pandas.DataFrame({self.derived_column_names_[0]: col})
        return res.astype(float)


def fit_clean_code(*, incoming_column_name, x, params, imputation_map):
    if not vtreat.util.numeric_has_range(x):
        return None
    replacement = params['missingness_imputation']
    try:
        replacement = imputation_map[incoming_column_name]
    except KeyError:
        pass
    if vtreat.util.can_convert_v_to_numeric(replacement):
        replacement_value = 0.0 + replacement
    elif callable(replacement):
        replacement_value = vtreat.util.summarize_column(x, fn=replacement)
    else:
        raise TypeError("unexpected imputation type " + str(type(replacement)) + " (" + incoming_column_name + ")")
    if pandas.isnull(replacement_value) or math.isnan(replacement_value) or math.isinf(replacement_value):
        raise ValueError("replacement was bad " + incoming_column_name + ": " + str(replacement_value))
    return CleanNumericTransform(
            incoming_column_name=incoming_column_name, replacement_value=replacement_value
        )


def fit_regression_impact_code(*, incoming_column_name, x, y, extra_args, params):
    sf = vtreat.util.grouped_by_x_statistics(x, y)
    if sf.shape[0] <= 1:
        return None
    if params["use_hierarchical_estimate"]:
        sf["_impact_code"] = sf["_hest"] - sf["_gm"]
    else:
        sf["_impact_code"] = sf["_group_mean"] - sf["_gm"]
    sf = sf.loc[:, ["x", "_impact_code"]].copy()
    newcol = incoming_column_name + "_impact_code"
    sf.columns = [incoming_column_name, newcol]
    return YAwareMappedCodeTransform(
        incoming_column_name=incoming_column_name,
        derived_column_name=newcol,
        treatment="impact_code",
        code_book=sf,
        refitter=fit_regression_impact_code,
        extra_args=extra_args,
        params=params,
    )


def fit_regression_deviation_code(*, incoming_column_name, x, y, extra_args, params):
    sf = vtreat.util.grouped_by_x_statistics(x, y)
    if sf.shape[0] <= 1:
        return None
    sf["_deviation_code"] = numpy.sqrt(sf["_var"])
    sf = sf.loc[:, ["x", "_deviation_code"]].copy()
    newcol = incoming_column_name + "_deviation_code"
    sf.columns = [incoming_column_name, newcol]
    return YAwareMappedCodeTransform(
        incoming_column_name=incoming_column_name,
        derived_column_name=newcol,
        treatment="deviation_code",
        code_book=sf,
        refitter=fit_regression_deviation_code,
        extra_args=extra_args,
        params=params,
    )


def fit_binomial_impact_code(*, incoming_column_name, x, y, extra_args, params):
    outcome_target = (extra_args["outcome_target"],)
    var_suffix = extra_args["var_suffix"]
    y = numpy.asarray(numpy.asarray(y) == outcome_target, dtype=float)
    sf = vtreat.util.grouped_by_x_statistics(x, y)
    if sf.shape[0] <= 1:
        return None
    eps = 1.0e-3
    if params["use_hierarchical_estimate"]:
        sf["_logit_code"] = numpy.log((sf["_hest"] + eps) / (sf["_gm"] + eps))
    else:
        sf["_logit_code"] = numpy.log((sf["_group_mean"] + eps) / (sf["_gm"] + eps))
    sf = sf.loc[:, ["x", "_logit_code"]].copy()
    newcol = incoming_column_name + "_logit_code" + var_suffix
    sf.columns = [incoming_column_name, newcol]
    return YAwareMappedCodeTransform(
        incoming_column_name=incoming_column_name,
        derived_column_name=newcol,
        treatment="logit_code",
        code_book=sf,
        refitter=fit_binomial_impact_code,
        extra_args=extra_args,
        params=params,
    )


class IndicatorCodeTransform(VarTransform):
    def __init__(
        self,
        incoming_column_name,
        derived_column_names,
        levels,
        *,
        sparse_indicators=False
    ):
        VarTransform.__init__(
            self, incoming_column_name, derived_column_names, "indicator_code"
        )
        self.levels_ = levels
        self.sparse_indicators_ = sparse_indicators

    def transform(self, data_frame):
        incoming_column_name = self.incoming_column_name_
        sf = pandas.DataFrame({incoming_column_name: data_frame[incoming_column_name]})
        bad_posns = vtreat.util.is_bad(sf[incoming_column_name])
        sf.loc[bad_posns, incoming_column_name] = "_NA_"
        col = sf[self.incoming_column_name_]

        def f(i):
            v = numpy.asarray(col == self.levels_[i]) + 0.0
            if self.sparse_indicators_:
                v = pandas.arrays.SparseArray(v, fill_value=0.0)
            return v

        res = [
            pandas.DataFrame({self.derived_column_names_[i]: f(i)})
            for i in range(len(self.levels_))
        ]
        res = pandas.concat(res, axis=1, sort=False)
        res.reset_index(inplace=True, drop=True)
        return res


def fit_indicator_code(
    *, incoming_column_name, x, min_fraction, sparse_indicators=False
):
    sf = pandas.DataFrame({incoming_column_name: x})
    bad_posns = vtreat.util.is_bad(sf[incoming_column_name])
    sf.loc[bad_posns, incoming_column_name] = "_NA_"
    counts = sf[incoming_column_name].value_counts()
    n = sf.shape[0]
    counts = counts[counts > 0]
    counts = counts[counts >= min_fraction * n]  # no more than 1/min_fraction symbols
    levels = [str(v) for v in counts.index]
    if len(levels) < 1:
        return None
    return IndicatorCodeTransform(
        incoming_column_name,
        vtreat.util.build_level_codes(incoming_column_name, levels),
        levels=levels,
        sparse_indicators=sparse_indicators
    )


def fit_prevalence_code(incoming_column_name, x):
    sf = pandas.DataFrame({"x": x})
    bad_posns = vtreat.util.is_bad(sf["x"])
    sf.loc[bad_posns, "x"] = "_NA_"
    sf.reset_index(inplace=True, drop=True)
    n = sf.shape[0]
    sf["_ni"] = 1.0
    sf = pandas.DataFrame(sf.groupby("x")["_ni"].sum())
    sf.reset_index(inplace=True, drop=False)
    sf["_hest"] = sf["_ni"] / n
    sf = sf.loc[:, ["x", "_hest"]].copy()
    newcol = incoming_column_name + "_prevalence_code"
    sf.columns = [incoming_column_name, newcol]
    sf[incoming_column_name] = sf[incoming_column_name].astype(str)
    sf.reset_index(inplace=True, drop=True)
    return MappedCodeTransform(
        incoming_column_name, newcol, treatment="prevalence_code", code_book=sf
    )


# noinspection PyPep8Naming
def fit_numeric_outcome_treatment(
    *, X, y, var_list, outcome_name, cols_to_copy, params, imputation_map
):
    if (var_list is None) or (len(var_list) <= 0):
        var_list = [co for co in X.columns]
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
    v_counts = {v: vtreat.util.get_unique_value_count(X[v]) for v in var_list}
    var_list = {v for v in var_list if v_counts[v] > 1}
    if len(var_list) <= 0:
        raise ValueError("no variables")
    xforms = []
    n = X.shape[0]
    all_bad = []
    for vi in var_list:
        n_bad = sum(vtreat.util.is_bad(X[vi]))
        if n_bad >= n:
            all_bad = all_bad + [vi]
        if (n_bad > 0) and (n_bad < n):
            if "missing_indicator" in params["coders"]:
                xforms = xforms + [
                    IndicateMissingTransform(
                        incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                    )
                ]
    var_list = [co for co in var_list if (not (co in set(all_bad)))]
    num_list = [co for co in var_list if vtreat.util.can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    id_like = [co for co in cat_list if v_counts[co] >= n]
    if len(id_like) > 0:
        warnings.warn("variable(s) " + ', '.join(id_like) + " have unique values per-row, dropping")
        cat_list = [co for co in var_list if co not in set(id_like)]
    if "clean_copy" in params["coders"]:
        for vi in num_list:
            xform = fit_clean_code(incoming_column_name=vi, x=X[vi], params=params, imputation_map=imputation_map)
            if xform is not None:
                # noinspection PyTypeChecker
                xforms = xforms + [xform]
    for vi in cat_list:
        if "impact_code" in params["coders"]:
            # noinspection PyTypeChecker
            xforms = xforms + [
                fit_regression_impact_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    y=y,
                    extra_args=None,
                    params=params,
                )
            ]
        if "deviation_code" in params["coders"]:
            # noinspection PyTypeChecker
            xforms = xforms + [
                fit_regression_deviation_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    y=y,
                    extra_args=None,
                    params=params,
                )
            ]
        if "prevalence_code" in params["coders"]:
            # noinspection PyTypeChecker
            xforms = xforms + [
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            ]
        if "indicator_code" in params["coders"]:
            # noinspection PyTypeChecker
            xforms = xforms + [
                fit_indicator_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    min_fraction=params["indicator_min_fraction"],
                    sparse_indicators=params["sparse_indicators"],
                )
            ]
    xforms = [xf for xf in xforms if xf is not None]
    for stp in params["user_transforms"]:
        stp.fit(X=X[var_list], y=y)
    return {
        "outcome_name": outcome_name,
        "cols_to_copy": cols_to_copy,
        "xforms": xforms,
    }


# noinspection PyPep8Naming
def fit_binomial_outcome_treatment(
    *, X, y, outcome_target, var_list, outcome_name, cols_to_copy, params, imputation_map
):
    if (var_list is None) or (len(var_list) <= 0):
        var_list = [co for co in X.columns]
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
    v_counts = {v: vtreat.util.get_unique_value_count(X[v]) for v in var_list}
    var_list = {v for v in var_list if v_counts[v] > 1}
    if len(var_list) <= 0:
        raise ValueError("no variables")
    xforms = []
    n = X.shape[0]
    all_bad = []
    for vi in var_list:
        n_bad = sum(vtreat.util.is_bad(X[vi]))
        if n_bad >= n:
            all_bad = all_bad + [vi]
        if (n_bad > 0) and (n_bad < n):
            if "missing_indicator" in params["coders"]:
                # noinspection PyTypeChecker
                xforms = xforms + [
                    IndicateMissingTransform(
                        incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                    )
                ]
    var_list = [co for co in var_list if (not (co in set(all_bad)))]
    num_list = [co for co in var_list if vtreat.util.can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    id_like = [co for co in cat_list if v_counts[co] >= n]
    if len(id_like) > 0:
        warnings.warn("variable(s) " + ', '.join(id_like) + " have unique values per-row, dropping")
        cat_list = [co for co in var_list if co not in set(id_like)]
    if "clean_copy" in params["coders"]:
        for vi in num_list:
            xform = fit_clean_code(incoming_column_name=vi, x=X[vi], params=params, imputation_map=imputation_map)
            if xform is not None:
                # noinspection PyTypeChecker
                xforms = xforms + [xform]
    extra_args = {"outcome_target": outcome_target, "var_suffix": ""}
    for vi in cat_list:
        if "logit_code" in params["coders"]:
            # noinspection PyTypeChecker
            xforms = xforms + [
                fit_binomial_impact_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    y=y,
                    extra_args=extra_args,
                    params=params,
                )
            ]
        if "prevalence_code" in params["coders"]:
            # noinspection PyTypeChecker
            xforms = xforms + [
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            ]
        if "indicator_code" in params["coders"]:
            # noinspection PyTypeChecker
            xforms = xforms + [
                fit_indicator_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    min_fraction=params["indicator_min_fraction"],
                    sparse_indicators=params["sparse_indicators"],
                )
            ]
    xforms = [xf for xf in xforms if xf is not None]
    for stp in params["user_transforms"]:
        stp.fit(X=X[var_list], y=y)
    return {
        "outcome_name": outcome_name,
        "cols_to_copy": cols_to_copy,
        "xforms": xforms,
    }


# noinspection PyPep8Naming
def fit_multinomial_outcome_treatment(
    *, X, y, var_list, outcome_name, cols_to_copy, params, imputation_map
):
    if (var_list is None) or (len(var_list) <= 0):
        var_list = [co for co in X.columns]
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
    v_counts = {v: vtreat.util.get_unique_value_count(X[v]) for v in var_list}
    var_list = {v for v in var_list if v_counts[v] > 1}
    if len(var_list) <= 0:
        raise ValueError("no variables")
    xforms = []
    n = X.shape[0]
    all_bad = []
    for vi in var_list:
        n_bad = sum(vtreat.util.is_bad(X[vi]))
        if n_bad >= n:
            all_bad = all_bad + [vi]
        if (n_bad > 0) and (n_bad < n):
            if "missing_indicator" in params["coders"]:
                # noinspection PyTypeChecker
                xforms = xforms + [
                    IndicateMissingTransform(
                        incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                    )
                ]
    outcomes = [oi for oi in set(y)]
    var_list = [co for co in var_list if (not (co in set(all_bad)))]
    num_list = [co for co in var_list if vtreat.util.can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    id_like = [co for co in cat_list if v_counts[co] >= n]
    if len(id_like) > 0:
        warnings.warn("variable(s) " + ', '.join(id_like) + " have unique values per-row, dropping")
        cat_list = [co for co in var_list if co not in set(id_like)]
    if "clean_copy" in params["coders"]:
        for vi in num_list:
            xform = fit_clean_code(incoming_column_name=vi, x=X[vi], params=params, imputation_map=imputation_map)
            if xform is not None:
                # noinspection PyTypeChecker
                xforms = xforms + [xform]
    for vi in cat_list:
        for outcome in outcomes:
            if "impact_code" in params["coders"]:
                extra_args = {
                    "outcome_target": outcome,
                    "var_suffix": ("_" + str(outcome)),
                }
                # noinspection PyTypeChecker
                xforms = xforms + [
                    fit_binomial_impact_code(
                        incoming_column_name=vi,
                        x=numpy.asarray(X[vi]),
                        y=y,
                        extra_args=extra_args,
                        params=params,
                    )
                ]
        if "prevalence_code" in params["coders"]:
            # noinspection PyTypeChecker
            xforms = xforms + [
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            ]
        if "indicator_code" in params["coders"]:
            # noinspection PyTypeChecker
            xforms = xforms + [
                fit_indicator_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    min_fraction=params["indicator_min_fraction"],
                    sparse_indicators=params["sparse_indicators"],
                )
            ]
    xforms = [xf for xf in xforms if xf is not None]
    if len(xforms) <= 0:
        raise ValueError("no variables created")
    for stp in params["user_transforms"]:
        stp.fit(X=X[var_list], y=y)
    return {
        "outcome_name": outcome_name,
        "cols_to_copy": cols_to_copy,
        "xforms": xforms,
    }


# noinspection PyPep8Naming
def fit_unsupervised_treatment(*, X, var_list, outcome_name, cols_to_copy, params, imputation_map):
    if (var_list is None) or (len(var_list) <= 0):
        var_list = [co for co in X.columns]
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
    v_counts = {v: vtreat.util.get_unique_value_count(X[v]) for v in var_list}
    var_list = {v for v in var_list if v_counts[v] > 1}
    if len(var_list) <= 0:
        raise ValueError("no variables")
    xforms = []
    n = X.shape[0]
    all_bad = []
    for vi in var_list:
        n_bad = sum(vtreat.util.is_bad(X[vi]))
        if n_bad >= n:
            all_bad = all_bad + [vi]
        if (n_bad > 0) and (n_bad < n):
            if "missing_indicator" in params["coders"]:
                # noinspection PyTypeChecker
                xforms = xforms + [
                    IndicateMissingTransform(
                        incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                    )
                ]
    var_list = [co for co in var_list if (not (co in set(all_bad)))]
    num_list = [co for co in var_list if vtreat.util.can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    id_like = [co for co in cat_list if v_counts[co] >= n]
    if len(id_like) > 0:
        warnings.warn("variable(s) " + ', '.join(id_like) + " have unique values per-row, dropping")
        cat_list = [co for co in var_list if co not in set(id_like)]
    if "clean_copy" in params["coders"]:
        for vi in num_list:
            xform = fit_clean_code(incoming_column_name=vi, x=X[vi], params=params, imputation_map=imputation_map)
            if xform is not None:
                # noinspection PyTypeChecker
                xforms = xforms + [xform]
    for vi in cat_list:
        if "prevalence_code" in params["coders"]:
            # noinspection PyTypeChecker
            xforms = xforms + [
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            ]
        if "indicator_code" in params["coders"]:
            # noinspection PyTypeChecker
            xforms = xforms + [
                fit_indicator_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    min_fraction=params["indicator_min_fraction"],
                    sparse_indicators=params["sparse_indicators"],
                )
            ]
    xforms = [xf for xf in xforms if xf is not None]
    for stp in params["user_transforms"]:
        stp.fit(X=X[var_list], y=None)
    return {
        "outcome_name": outcome_name,
        "cols_to_copy": cols_to_copy,
        "xforms": xforms,
    }


def pre_prep_frame(x, *, col_list, cols_to_copy):
    """Create a copy of pandas.DataFrame x restricted to col_list union cols_to_copy with col_list - cols_to_copy
    converted to only string and numeric types.  New pandas.DataFrame has trivial indexing.  If col_list
    is empty it is interpreted as all columns."""

    if cols_to_copy is None:
        cols_to_copy = []
    if (col_list is None) or (len(col_list) <= 0):
        col_list = [co for co in x.columns]
    x_set = set(x.columns)
    col_set = set(col_list)
    for ci in cols_to_copy:
        if (ci in x_set) and (ci not in col_set):
            col_list = col_list + [ci]
    col_set = set(col_list)
    missing_cols = col_set - x_set
    if len(missing_cols) > 0:
        raise KeyError("referred to not-present columns " + str(missing_cols))
    cset = set(cols_to_copy)
    if len(col_list) <= 0:
        raise ValueError("no variables")
    x = x.loc[:, col_list]
    x = x.reset_index(inplace=False, drop=True)
    for c in x.columns:
        if c in cset:
            continue
        bad_ind = vtreat.util.is_bad(x[c])
        if vtreat.util.can_convert_v_to_numeric(x[c]):
            x[c] = vtreat.util.safe_to_numeric_array(x[c])
        else:
            # https://stackoverflow.com/questions/22231592/pandas-change-data-type-of-series-to-string
            x[c] = numpy.asarray(x[c].apply(str), dtype=str)
        x.loc[bad_ind, c] = numpy.nan
    return x


def perform_transform(*, x, transform, params):
    plan = transform.plan_
    xform_steps = [xfi for xfi in plan["xforms"]]
    user_steps = [stp for stp in params["user_transforms"]]
    # restrict down to to results we are going to use
    if (transform.result_restriction is not None) and (len(transform.result_restriction) > 0):
        xform_steps = [xfi for xfi in xform_steps
                       if len(set(xfi.derived_column_names_).intersection(transform.result_restriction)) > 0]
        user_steps = [stp for stp in user_steps
                      if len(set(stp.derived_vars_).intersection(transform.result_restriction)) > 0]
    # check all required columns are present
    needs = set()
    for xfi in xform_steps:
        if xfi.incoming_column_name_ is not None:
            needs.add(xfi.incoming_column_name_)
    for stp in user_steps:
        if stp.incoming_vars_ is not None:
            needs.update(stp.incoming_vars_)
    missing = needs - set(x.columns)
    if len(missing) > 0:
        raise ValueError("missing required input columns " + str(missing))
    # do the work
    new_frames = [xfi.transform(x) for xfi in (xform_steps + user_steps)]
    new_frames = [frm for frm in new_frames if (frm is not None) and (frm.shape[1] > 0)]
    # see if we want to copy over any columns
    copy_set = set(plan["cols_to_copy"])
    to_copy = [ci for ci in x.columns if ci in copy_set]
    if len(to_copy) > 0:
        cp = x.loc[:, to_copy].copy()
        new_frames = [cp] + new_frames
    if len(new_frames) <= 0:
        raise ValueError("no columns transformed")
    res = pandas.concat(new_frames, axis=1, sort=False)
    res.reset_index(inplace=True, drop=True)
    return res


def limit_to_appropriate_columns(*, res, transform):
    plan = transform.plan_
    to_copy = set(plan["cols_to_copy"])
    to_take = set([
        ci for ci in transform.score_frame_["variable"][transform.score_frame_["has_range"]]])
    if (transform.result_restriction is not None) and (len(transform.result_restriction) > 0):
        to_take = to_take.intersection(transform.result_restriction)
    cols_to_keep = [ci for ci in res.columns if (ci in to_copy) or (ci in to_take)]
    if len(cols_to_keep) <= 0:
        raise ValueError("no columns retained")
    res = res[cols_to_keep].copy()
    res.reset_index(inplace=True, drop=True)
    return res


# val_list is a list single column Pandas data frames
def mean_of_single_column_pandas_list(val_list):
    if val_list is None or len(val_list) <= 0:
        return numpy.nan
    d = pandas.concat(val_list, axis=0, sort=False)
    col = d.columns[0]
    d = d.loc[numpy.logical_not(vtreat.util.is_bad(d[col])), [col]]
    if d.shape[0] < 1:
        return numpy.nan
    return numpy.mean(d[col])


# assumes each y-aware variable produces one derived column
# also clears out refitter_ values to None
def cross_patch_refit_y_aware_cols(*, x, y, res, plan, cross_plan):
    if cross_plan is None or len(cross_plan) <= 1:
        for xf in plan["xforms"]:
            xf.refitter_ = None
        return res
    incoming_colset = set(x.columns)
    derived_colset = set(res.columns)
    for xf in plan["xforms"]:
        if not xf.need_cross_treatment_:
            continue
        incoming_column_name = xf.incoming_column_name_
        derived_column_name = xf.derived_column_names_[0]
        if derived_column_name not in derived_colset:
            continue
        if incoming_column_name not in incoming_colset:
            raise KeyError("missing required column " + incoming_column_name)
        if xf.refitter_ is None:
            raise ValueError(
                "refitter is None: "
                + incoming_column_name
                + " -> "
                + derived_column_name
            )

        # noinspection PyPep8Naming
        def maybe_transform(*, fit, X):
            if fit is None:
                return None
            return fit.transform(X)

        patches = [
            maybe_transform(
                fit=xf.refitter_(
                    incoming_column_name=incoming_column_name,
                    x=x[incoming_column_name][cp["train"]],
                    y=y[cp["train"]],
                    extra_args=xf.extra_args_,
                    params=xf.params_,
                ),
                X=x.loc[cp["app"], [incoming_column_name]],
            )
            for cp in cross_plan
        ]
        # replace any missing sections with global average (slight data leak potential)
        avg = mean_of_single_column_pandas_list(
            [pi for pi in patches if pi is not None]
        )
        if numpy.isnan(avg):
            avg = 0
        res[derived_column_name] = avg
        for i in range(len(cross_plan)):
            pi = patches[i]
            if pi is None:
                continue
            pi.reset_index(inplace=True, drop=True)
            cp = cross_plan[i]
            res.loc[cp["app"], derived_column_name] = numpy.asarray(
                pi[derived_column_name]
            ).reshape((len(pi), ))
        res.loc[vtreat.util.is_bad(res[derived_column_name]), derived_column_name] = avg
    for xf in plan["xforms"]:
        xf.refitter_ = None
    return res


def cross_patch_user_y_aware_cols(*, x, y, res, params, cross_plan):
    if cross_plan is None or len(cross_plan) <= 1:
        return res
    incoming_colset = set(x.columns)
    derived_colset = set(res.columns)
    if len(derived_colset) <= 0:
        return res
    for ut in params["user_transforms"]:
        if not ut.y_aware_:
            continue
        instersect_in = incoming_colset.intersection(set(ut.incoming_vars_))
        instersect_out = derived_colset.intersection(set(ut.derived_vars_))
        if len(instersect_out) <= 0:
            continue
        if len(instersect_out) != len(ut.derived_vars_):
            raise ValueError("not all derived columns are in res frame")
        if len(instersect_in) != len(ut.incoming_vars_):
            raise KeyError("missing required columns")
        patches = [
            ut.fit(X=x.loc[cp["train"], ut.incoming_vars_], y=y[cp["train"]]).transform(
                X=x.loc[cp["app"], ut.incoming_vars_]
            )
            for cp in cross_plan
        ]
        for col in ut.derived_vars_:
            # replace any missing sections with global average (slight data leak potential)
            avg = mean_of_single_column_pandas_list(
                [pi.loc[:, [col]] for pi in patches if pi is not None]
            )
            if numpy.isnan(avg):
                avg = 0
            res[col] = avg
            for i in range(len(cross_plan)):
                pi = patches[i]
                if pi is None:
                    continue
                pi.reset_index(inplace=True, drop=True)
                cp = cross_plan[i]
                res.loc[cp["app"], col] = numpy.asarray(pi[col]).reshape((len(pi), ))
            res.loc[vtreat.util.is_bad(res[col]), col] = avg
    return res


def score_plan_variables(cross_frame, outcome, plan, params,
                         *,
                         is_classification=False):
    def describe_xf(xf):
        description = pandas.DataFrame({"variable": xf.derived_column_names_})
        description["orig_variable"] = xf.incoming_column_name_
        description["treatment"] = xf.treatment_
        description["y_aware"] = xf.need_cross_treatment_
        return description

    def describe_ut(ut):
        description = pandas.DataFrame(
            {"orig_variable": ut.incoming_vars_, "variable": ut.derived_vars_}
        )
        description["treatment"] = ut.treatment_
        description["y_aware"] = ut.y_aware_
        return description

    var_table = pandas.concat(
        [describe_xf(xf) for xf in plan["xforms"]]
        + [
            describe_ut(ut)
            for ut in params["user_transforms"]
            if len(ut.incoming_vars_) > 0
        ],
        sort=False,
    )
    var_table.reset_index(inplace=True, drop=True)
    sf = vtreat.util.score_variables(
        cross_frame,
        variables=var_table["variable"],
        outcome=outcome,
        is_classification=is_classification
    )
    score_frame = pandas.merge(var_table, sf, how="left", on=["variable"], sort=False)
    num_treatment_types = len(score_frame["treatment"].unique())
    score_frame["_one"] = 1.0
    score_frame["vcount"] = score_frame.groupby("treatment")["_one"].transform("sum")
    score_frame["default_threshold"] = 1.0 / (
        score_frame["vcount"] * num_treatment_types
    )
    score_frame.drop(["_one"], axis=1, inplace=True)
    score_frame["recommended"] = numpy.logical_and(
        score_frame["has_range"],
        numpy.logical_and(
            numpy.logical_not(
                numpy.logical_or(
                    numpy.isnan(score_frame["significance"]),
                    numpy.isnan(score_frame["PearsonR"]),
                )
            ),
            numpy.logical_and(
                score_frame["significance"] < score_frame["default_threshold"],
                numpy.logical_or(
                    score_frame["PearsonR"] > 0.0,
                    numpy.logical_not(score_frame["y_aware"]),
                ),
            ),
        ),
    )
    return score_frame


def pseudo_score_plan_variables(*, cross_frame, plan, params):
    def describe_xf(xf):
        description = pandas.DataFrame({"variable": xf.derived_column_names_})
        description["orig_variable"] = xf.incoming_column_name_
        description["treatment"] = xf.treatment_
        description["y_aware"] = xf.need_cross_treatment_
        return description

    def describe_ut(ut):
        description = pandas.DataFrame(
            {"orig_variable": ut.incoming_vars_, "variable": ut.derived_vars_}
        )
        description["treatment"] = ut.treatment_
        description["y_aware"] = ut.y_aware_
        return description

    score_frame = pandas.concat(
        [describe_xf(xf) for xf in plan["xforms"]]
        + [
            describe_ut(ut)
            for ut in params["user_transforms"]
            if len(ut.incoming_vars_) > 0
        ],
        sort=False,
    )
    score_frame.reset_index(inplace=True, drop=True)

    score_frame["has_range"] = [
        vtreat.util.numeric_has_range(cross_frame[c]) for c in score_frame["variable"]
    ]
    score_frame["PearsonR"] = numpy.nan
    score_frame["significance"] = numpy.nan
    score_frame["recommended"] = score_frame["has_range"].copy()
    score_frame["_one"] = 1.0
    score_frame["vcount"] = score_frame.groupby("treatment")["_one"].transform("sum")
    score_frame.drop(["_one"], axis=1, inplace=True)
    return score_frame


class VariableTreatment(ABC):
    def __init__(
            self, *,
            var_list=None,
            outcome_name=None,
            outcome_target=None,
            cols_to_copy=None,
            params=None,
            imputation_map=None,
    ):
        if var_list is None:
            var_list = []
        else:
            var_list = vtreat.util.unique_itmes_in_order(var_list)
        if cols_to_copy is None:
            cols_to_copy = []
        else:
            cols_to_copy = vtreat.util.unique_itmes_in_order(cols_to_copy)
        if outcome_name is not None and outcome_name not in set(cols_to_copy):
            cols_to_copy = cols_to_copy + [outcome_name]
        confused = set(cols_to_copy).intersection(set(var_list))
        if len(confused) > 0:
            raise ValueError("variables in treatment plan and non-treatment: " + ', '.join(confused))
        if imputation_map is None:
            imputation_map = {}  # dict
        self.outcome_name_ = outcome_name
        self.outcome_target_ = outcome_target
        self.var_list_ = [vi for vi in var_list if vi not in set(cols_to_copy)]
        self.cols_to_copy_ = cols_to_copy
        self.params_ = params.copy()
        self.imputation_map_ = imputation_map.copy()
        self.plan_ = None
        self.score_frame_ = None
        self.cross_rows_ = None
        self.cross_plan_ = None
        self.last_fit_x_id_ = None
        self.last_result_columns = None
        self.result_restriction = None
        self.clear()

    def check_column_names(self, col_names):
        to_check = set(self.var_list_)
        if self.outcome_name_ is not None:
            to_check.add(self.outcome_name_)
        if self.cols_to_copy_ is not None:
            to_check.update(self.cols_to_copy_)
        seen = [c for c in col_names if c in to_check]
        if len(seen) != len(set(seen)):
            raise ValueError("duplicate column names in frame")

    def clear(self):
        self.plan_ = None
        self.score_frame_ = None
        self.cross_rows_ = None
        self.cross_plan_ = None
        self.last_fit_x_id_ = None
        self.last_result_columns = None
        self.result_restriction = None

    def get_result_restriction(self):
        if self.result_restriction is None:
            return None
        return self.result_restriction.copy()

    def set_result_restriction(self, new_vars):
        self.result_restriction = None
        if (new_vars is not None) and (len(new_vars) > 0):
            self.result_restriction = set(new_vars)

    def merge_params(self, p):
        raise NotImplementedError("base class called")

    # display methods

    def __repr__(self):
        fmted = str(self.__class__.__module__) + "." + str(self.__class__.__name__) + '('
        if self.outcome_name_ is not None:
            fmted = fmted + "outcome_name=" + pprint.pformat(self.outcome_name_) + ", "
        if self.outcome_target_ is not None:
            fmted = fmted + "outcome_target=" + pprint.pformat(self.outcome_target_) + ", "
        if (self.var_list_ is not None) and (len(self.var_list_) > 0):
            fmted = fmted + "var_list=" + pprint.pformat(self.var_list_) + ", "
        if (self.cols_to_copy_ is not None) and (len(self.cols_to_copy_) > 0):
            fmted = fmted + "cols_to_copy=" + pprint.pformat(self.cols_to_copy_) + ", "
        # if (self.params_ is not None) and (len(self.params_) > 0):
        #     fmted = fmted + "params=" + pprint.pformat(self.params_) + ",\n"
        # if (self.imputation_map_ is not None) and (len(self.imputation_map_) > 0):
        #     fmted = fmted + "imputation_map=" + pprint.pformat(self.imputation_map_) + ",\n"
        fmted = fmted + ')'
        return fmted

    def __str__(self):
        return self.__repr__()

    # sklearn pipeline step methods

    # https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X=X, y=y)
        return self

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit_transform(self, X, y=None, **fit_params):
        raise NotImplementedError("base class method called")

    # noinspection PyPep8Naming
    def transform(self, X):
        raise NotImplementedError("base class method called")

    # https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_params(self, deep=True):
        """
        vtreat exposes a subset of controls as tunable parameters, users can choose this set
        by specifying the tunable_params list in object construction parameters
        """
        return {ti: self.params_[ti] for ti in self.params_["tunable_params"]}

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def set_params(self, **params):
        """
        vtreat exposes a subset of controls as tunable parameters, users can choose this set
        by specifying the tunable_params list in object construction parameters
        """
        for (k, v) in params.items():
            if k in self.params_["tunable_params"]:
                self.params_[k] = v
        return self

    # extra methods to look more like sklearn objects

    # https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    # noinspection PyPep8Naming
    def fit_predict(self, X, y=None, **fit_params):
        return self.fit_transform(X=X, y=y, **fit_params)

    # noinspection PyPep8Naming
    def predict(self, X):
        return self.transform(X)

    # noinspection PyPep8Naming
    def predict_proba(self, X):
        return self.transform(X)

    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/compose/_column_transformer.py

    def get_feature_names(self, input_features=None):
        if self.score_frame_ is None:
            raise ValueError("get_feature_names called on uninitialized vtreat transform")
        if self.params_['filter_to_recommended']:
            new_vars = [self.score_frame_['variable'][i] for i in range(self.score_frame_.shape[0])
                        if self.score_frame_['has_range'][i] and self.score_frame_['recommended'][i]
                        and (input_features is None or self.score_frame_['orig_variable'][i] in input_features)]
        else:
            new_vars = [self.score_frame_['variable'][i] for i in range(self.score_frame_.shape[0])
                        if self.score_frame_['has_range'][i]
                        and (input_features is None or self.score_frame_['orig_variable'][i] in input_features)]
        new_vars = new_vars + self.cols_to_copy_
        return new_vars
