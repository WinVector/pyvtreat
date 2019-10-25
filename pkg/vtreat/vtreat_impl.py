# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:07:57 2019

@author: johnmount
"""

import numpy
import pandas

import vtreat.util
import vtreat.transform


class VarTransform:
    """build a treatment plan for a numeric outcome (regression)"""

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
        col = numpy.asarray(data_frame[self.incoming_column_name_].copy()).astype(float)
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
    y = numpy.asarray(numpy.asarray(y) == outcome_target, dtype=numpy.float64)
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
                v = pandas.SparseArray(v, fill_value=0.0)
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
    counts = counts[counts >= min_fraction * n]  # no more than 1/min_fraction symbols
    levels = [v for v in counts.index]
    if len(levels) < 1:
        return None
    return IndicatorCodeTransform(
        incoming_column_name,
        [incoming_column_name + "_lev_" + lev for lev in levels],
        levels=levels,
        sparse_indicators=sparse_indicators,
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
    *, X, y, var_list, outcome_name, cols_to_copy, params
):
    if (var_list is None) or (len(var_list) <= 0):
        var_list = [co for co in X.columns]
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
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
    if "clean_copy" in params["coders"]:
        for vi in num_list:
            summaryi = vtreat.util.characterize_numeric(X[vi])
            if summaryi["varies"] and summaryi["has_range"]:
                xforms = xforms + [
                    CleanNumericTransform(
                        incoming_column_name=vi, replacement_value=summaryi["mean"]
                    )
                ]
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
    *, X, y, outcome_target, var_list, outcome_name, cols_to_copy, params
):
    if (var_list is None) or (len(var_list) <= 0):
        var_list = [co for co in X.columns]
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
    if len(var_list) <= 0:
        raise ValueError("no variables")
    xforms = []
    n = X.shape[0]
    all_badd = []
    for vi in var_list:
        n_bad = sum(vtreat.util.is_bad(X[vi]))
        if n_bad >= n:
            all_badd = all_badd + [vi]
        if (n_bad > 0) and (n_bad < n):
            if "missing_indicator" in params["coders"]:
                # noinspection PyTypeChecker
                xforms = xforms + [
                    IndicateMissingTransform(
                        incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                    )
                ]
    var_list = [co for co in var_list if (not (co in set(all_badd)))]
    num_list = [co for co in var_list if vtreat.util.can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    if "clean_copy" in params["coders"]:
        for vi in num_list:
            summaryi = vtreat.util.characterize_numeric(X[vi])
            if summaryi["varies"] and summaryi["has_range"]:
                # noinspection PyTypeChecker
                xforms = xforms + [
                    CleanNumericTransform(
                        incoming_column_name=vi, replacement_value=summaryi["mean"]
                    )
                ]
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
    *, X, y, var_list, outcome_name, cols_to_copy, params
):
    if (var_list is None) or (len(var_list) <= 0):
        var_list = [co for co in X.columns]
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
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
    if "clean_copy" in params["coders"]:
        for vi in num_list:
            summaryi = vtreat.util.characterize_numeric(X[vi])
            if summaryi["varies"] and summaryi["has_range"]:
                # noinspection PyTypeChecker
                xforms = xforms + [
                    CleanNumericTransform(
                        incoming_column_name=vi, replacement_value=summaryi["mean"]
                    )
                ]
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
def fit_unsupervised_treatment(*, X, var_list, outcome_name, cols_to_copy, params):
    if (var_list is None) or (len(var_list) <= 0):
        var_list = [co for co in X.columns]
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
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
    if "clean_copy" in params["coders"]:
        for vi in num_list:
            summaryi = vtreat.util.characterize_numeric(X[vi])
            if summaryi["varies"] and summaryi["has_range"]:
                # noinspection PyTypeChecker
                xforms = xforms + [
                    CleanNumericTransform(
                        incoming_column_name=vi, replacement_value=summaryi["mean"]
                    )
                ]
    for vi in cat_list:
        if "prevalence_code" in params["coders"]:
            # noinspection PyTypeChecker
            xforms = xforms + [
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            ]
        if "indicator_code" in params["coders"]:
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
            x[c] = numpy.asarray(x[c] + 0, dtype=float)
        else:
            # https://stackoverflow.com/questions/22231592/pandas-change-data-type-of-series-to-string
            x[c] = numpy.asarray(x[c].apply(str), dtype=str)
        x.loc[bad_ind, c] = numpy.nan
    return x


def perform_transform(*, x, transform, params):
    plan = transform.plan_
    new_frames = [xfi.transform(x) for xfi in plan["xforms"]]
    for stp in params["user_transforms"]:
        frm = stp.transform(X=x)
        if frm is not None and frm.shape[1] > 0:
            new_frames = new_frames + [frm]
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
    if ("filter_to_recommended" in transform.params_.keys()) and transform.params_[
        "filter_to_recommended"
    ]:
        to_take = set(
            [
                ci
                for ci in transform.score_frame_["variable"][
                    transform.score_frame_["recommended"]
                ]
            ]
        )
    else:
        to_take = set(
            [
                ci
                for ci in transform.score_frame_["variable"][
                    transform.score_frame_["has_range"]
                ]
            ]
        )
    cols_to_keep = [ci for ci in res.columns if ci in to_copy or ci in to_take]
    if len(cols_to_keep) <= 0:
        raise ValueError("no columns retained")
    return res[cols_to_keep]


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
            ).reshape((len(pi)))
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
                res.loc[cp["app"], col] = numpy.asarray(pi[col]).reshape((len(pi)))
            res.loc[vtreat.util.is_bad(res[col]), col] = avg
    return res


def score_plan_variables(cross_frame, outcome, plan, params):
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
        cross_frame, variables=var_table["variable"], outcome=outcome
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

    def has_range(x):
        x = numpy.asarray(pandas.Series(x))
        return numpy.max(x) > numpy.min(x)

    score_frame["has_range"] = [
        has_range(cross_frame[c]) for c in score_frame["variable"]
    ]
    score_frame["PearsonR"] = numpy.nan
    score_frame["significance"] = numpy.nan
    score_frame["recommended"] = score_frame["has_range"].copy()
    score_frame["_one"] = 1.0
    score_frame["vcount"] = score_frame.groupby("treatment")["_one"].transform("sum")
    score_frame.drop(["_one"], axis=1, inplace=True)
    return score_frame
