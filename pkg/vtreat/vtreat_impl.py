# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:07:57 2019

@author: johnmount
"""

import numpy
import pandas
import statistics

import vtreat.util


def characterize_numeric(x):
    """compute na count, min,max,mean of a numeric vector"""
    x = numpy.asarray(x).astype(float)
    not_nan = numpy.logical_not(numpy.isnan(x))
    n_not_nan = sum(not_nan)
    n = len(x)
    if n_not_nan <= 0:
        return {
            "n": n,
            "n_not_nan": n_not_nan,
            "min": None,
            "mean": None,
            "max": None,
            "varies": False,
            "has_range": False,
        }
    x = x[not_nan]
    mn = numpy.min(x)
    mx = numpy.max(x)
    return {
        "n": n,
        "n_not_nan": n_not_nan,
        "min": mn,
        "mean": numpy.mean(x),
        "max": mx,
        "varies": (mx > mn) or ((n_not_nan > 0) and (n_not_nan < n)),
        "has_range": (mx > mn),
    }


class VarTransform:
    """build a treatment plan for a numeric outcome (regression)"""

    def __init__(self, incoming_column_name, derived_column_names, treatment):
        self.incoming_column_name_ = incoming_column_name
        self.derived_column_names_ = derived_column_names.copy()
        self.treatment_ = treatment
        self.need_cross_treatment_ = False
        self.refitter_ = None

    def transform(self, data_frame):
        return None


class CleanNumericTransform(VarTransform):
    def __init__(self, incoming_column_name, replacement_value):
        VarTransform.__init__(
            self, incoming_column_name, [incoming_column_name], "clean_copy"
        )
        self.replacement_value_ = replacement_value

    def transform(self, data_frame):
        col = numpy.asarray(data_frame[self.incoming_column_name_].copy()).astype(float)
        na_posns = numpy.isnan(col)
        col[na_posns] = self.replacement_value_
        res = pandas.DataFrame({self.derived_column_names_[0]: col})
        return res


class IndicateMissingTransform(VarTransform):
    def __init__(self, incoming_column_name, derived_column_name):
        VarTransform.__init__(
            self, incoming_column_name, [derived_column_name], "missing_indicator"
        )

    def transform(self, data_frame):
        col = data_frame[self.incoming_column_name_].isnull()
        res = pandas.DataFrame({self.derived_column_names_[0]: col})
        return res.astype(float)


def can_convert_v_to_numeric(x):
    """check if non-empty vector can convert to numeric"""
    try:
        x + 0
        return True
    except:
        return False


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
        na_posns = sf[incoming_column_name].isnull()
        sf.loc[na_posns, incoming_column_name] = "_NA_"
        res = pandas.merge(
            sf, self.code_book_, on=[self.incoming_column_name_], how="left", sort=False
        )  # ordered by left table rows
        res = res[[derived_column_name]].copy()
        res.loc[res[derived_column_name].isnull(), derived_column_name] = 0
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


def grouped_by_x_statistics(x, y):
    """compute some grouped by x summaries of y"""
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


def fit_regression_impact_code(*, incoming_column_name, x, y, extra_args):
    sf = grouped_by_x_statistics(x, y)
    if sf.shape[0] <= 1:
        return None
    sf["_impact_code"] = sf["_hest"] - sf["_gm"]
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
    )


def fit_regression_deviation_code(*, incoming_column_name, x, y, extra_args):
    sf = grouped_by_x_statistics(x, y)
    if sf.shape[0] <= 1:
        return None
    sf["_deviance_code"] = numpy.sqrt(sf["_var"])
    sf = sf.loc[:, ["x", "_deviance_code"]].copy()
    newcol = incoming_column_name + "_deviance_code"
    sf.columns = [incoming_column_name, newcol]
    return YAwareMappedCodeTransform(
        incoming_column_name=incoming_column_name,
        derived_column_name=newcol,
        treatment="deviance_code",
        code_book=sf,
        refitter=fit_regression_deviation_code,
        extra_args=extra_args,
    )


def fit_binomial_impact_code(*, incoming_column_name, x, y, extra_args):
    outcome_target = (extra_args["outcome_target"],)
    var_suffix = extra_args["var_suffix"]
    y = numpy.asarray(numpy.asarray(y) == outcome_target, dtype=numpy.float64)
    sf = grouped_by_x_statistics(x, y)
    if sf.shape[0] <= 1:
        return None
    sf["_logit_code"] = numpy.log(sf["_hest"]) - numpy.log(sf["_gm"])
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
    )


class IndicatorCodeTransform(VarTransform):
    def __init__(self, incoming_column_name, derived_column_names, levels):
        VarTransform.__init__(
            self, incoming_column_name, derived_column_names, "indicator"
        )
        self.levels_ = levels

    def transform(self, data_frame):
        incoming_column_name = self.incoming_column_name_
        sf = pandas.DataFrame({incoming_column_name: data_frame[incoming_column_name]})
        na_posns = sf[incoming_column_name].isnull()
        sf.loc[na_posns, incoming_column_name] = "_NA_"
        col = sf[self.incoming_column_name_]

        def f(i):
            v = col == self.levels_[i]
            return numpy.asarray(v) + 0

        res = [
            pandas.DataFrame({self.derived_column_names_[i]: f(i)})
            for i in range(len(self.levels_))
        ]
        res = pandas.concat(res, axis=1)
        res.reset_index(inplace=True, drop=True)
        return res


def fit_indicator_code(incoming_column_name, x):
    sf = pandas.DataFrame({incoming_column_name: x})
    na_posns = sf[incoming_column_name].isnull()
    sf.loc[na_posns, incoming_column_name] = "_NA_"
    counts = sf[incoming_column_name].value_counts()
    n = sf.shape[0]
    counts = counts[counts >= n / 20]  # no more than 20 symbols
    levels = [v for v in counts.index]
    if len(levels) < 1:
        return None
    return IndicatorCodeTransform(
        incoming_column_name,
        [incoming_column_name + "_lev_" + lev for lev in levels],
        levels=levels,
    )


def fit_prevalence_code(incoming_column_name, x):
    sf = pandas.DataFrame({"x": x})
    na_posns = sf["x"].isnull()
    sf.loc[na_posns, "x"] = "_NA_"
    sf.reset_index(inplace=True, drop=True)
    n = sf.shape[0]
    sf["_ni"] = 1.0
    sf = pandas.DataFrame(sf.groupby("x")["_ni"].sum())
    sf.reset_index(inplace=True, drop=False)
    # adjusted from ni to ni-1 lto make
    # rare levels look like new levels.
    sf["_hest"] = (sf["_ni"] - 1.0) / n
    sf = sf.loc[:, ["x", "_hest"]].copy()
    newcol = incoming_column_name + "_prevalence_code"
    sf.columns = [incoming_column_name, newcol]
    sf[incoming_column_name] = sf[incoming_column_name].astype(str)
    sf.reset_index(inplace=True, drop=True)
    return MappedCodeTransform(
        incoming_column_name, newcol, treatment="prevalance", code_book=sf
    )


def fit_numeric_outcome_treatment(
    *, X, y, var_list, outcome_name, cols_to_copy
):
    if (var_list is None) or (len(var_list) <= 0):
        var_list = [co for co in X.columns]
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
    xforms = []
    n = X.shape[0]
    all_null = []
    for vi in var_list:
        n_null = sum(X[vi].isnull())
        if n_null >= n:
            all_null = all_null + [vi]
        if (n_null > 0) and (n_null < n):
            xforms = xforms + [
                IndicateMissingTransform(
                    incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                )
            ]
    var_list = [co for co in var_list if (not (co in set(all_null)))]
    num_list = [co for co in var_list if can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    for vi in num_list:
        summaryi = characterize_numeric(X[vi])
        if summaryi["varies"] and summaryi["has_range"]:
            xforms = xforms + [
                CleanNumericTransform(
                    incoming_column_name=vi, replacement_value=summaryi["mean"]
                )
            ]
    for vi in cat_list:
        xforms = xforms + [
            fit_regression_impact_code(
                incoming_column_name=vi, x=numpy.asarray(X[vi]), y=y, extra_args=None
            )
        ]
        xforms = xforms + [
            fit_regression_deviation_code(
                incoming_column_name=vi, x=numpy.asarray(X[vi]), y=y, extra_args=None
            )
        ]
        xforms = xforms + [
            fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
        ]
        xforms = xforms + [
            fit_indicator_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
        ]
    xforms = [xf for xf in xforms if xf is not None]
    return {"outcome_name": outcome_name, "cols_to_copy": cols_to_copy, "xforms": xforms}


def fit_binomial_outcome_treatment(
    *, X, y, outcome_target, var_list, outcome_name, cols_to_copy
):
    if (var_list is None) or (len(var_list) <= 0):
        var_list = [co for co in X.columns]
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
    xforms = []
    n = X.shape[0]
    all_null = []
    for vi in var_list:
        n_null = sum(X[vi].isnull())
        if n_null >= n:
            all_null = all_null + [vi]
        if (n_null > 0) and (n_null < n):
            xforms = xforms + [
                IndicateMissingTransform(
                    incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                )
            ]
    var_list = [co for co in var_list if (not (co in set(all_null)))]
    num_list = [co for co in var_list if can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    for vi in num_list:
        summaryi = characterize_numeric(X[vi])
        if summaryi["varies"] and summaryi["has_range"]:
            xforms = xforms + [
                CleanNumericTransform(
                    incoming_column_name=vi, replacement_value=summaryi["mean"]
                )
            ]
    extra_args = {"outcome_target": outcome_target, "var_suffix": ""}
    for vi in cat_list:
        xforms = xforms + [
            fit_binomial_impact_code(
                incoming_column_name=vi,
                x=numpy.asarray(X[vi]),
                y=y,
                extra_args=extra_args,
            )
        ]
        xforms = xforms + [
            fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
        ]
        xforms = xforms + [
            fit_indicator_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
        ]
    xforms = [xf for xf in xforms if xf is not None]
    return {"outcome_name": outcome_name, "cols_to_copy": cols_to_copy, "xforms": xforms}


def fit_multinomial_outcome_treatment(
    *, X, y, var_list, outcome_name, cols_to_copy
):
    if (var_list is None) or (len(var_list) <= 0):
        var_list = [co for co in X.columns]
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
    xforms = []
    n = X.shape[0]
    all_null = []
    for vi in var_list:
        n_null = sum(X[vi].isnull())
        if n_null >= n:
            all_null = all_null + [vi]
        if (n_null > 0) and (n_null < n):
            xforms = xforms + [
                IndicateMissingTransform(
                    incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                )
            ]
    outcomes = [oi for oi in set(y)]
    var_list = [co for co in var_list if (not (co in set(all_null)))]
    num_list = [co for co in var_list if can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    for vi in num_list:
        summaryi = characterize_numeric(X[vi])
        if summaryi["varies"] and summaryi["has_range"]:
            xforms = xforms + [
                CleanNumericTransform(
                    incoming_column_name=vi, replacement_value=summaryi["mean"]
                )
            ]
    for vi in cat_list:
        for outcome in outcomes:
            extra_args = {"outcome_target": outcome, "var_suffix": ("_" + str(outcome))}
            xforms = xforms + [
                fit_binomial_impact_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    y=y,
                    extra_args=extra_args,
                )
            ]
        xforms = xforms + [
            fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
        ]
        xforms = xforms + [
            fit_indicator_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
        ]
    xforms = [xf for xf in xforms if xf is not None]
    return {"outcome_name": outcome_name, "cols_to_copy": cols_to_copy, "xforms": xforms}


def fit_unsupervised_treatment(
    *, X, var_list, outcome_name, cols_to_copy
):
    if (var_list is None) or (len(var_list) <= 0):
        var_list = [co for co in X.columns]
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
    xforms = []
    n = X.shape[0]
    all_null = []
    for vi in var_list:
        n_null = sum(X[vi].isnull())
        if n_null >= n:
            all_null = all_null + [vi]
        if (n_null > 0) and (n_null < n):
            xforms = xforms + [
                IndicateMissingTransform(
                    incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                )
            ]
    var_list = [co for co in var_list if (not (co in set(all_null)))]
    num_list = [co for co in var_list if can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    for vi in num_list:
        summaryi = characterize_numeric(X[vi])
        if summaryi["varies"] and summaryi["has_range"]:
            xforms = xforms + [
                CleanNumericTransform(
                    incoming_column_name=vi, replacement_value=summaryi["mean"]
                )
            ]
    for vi in cat_list:
        xforms = xforms + [
            fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
        ]
        xforms = xforms + [
            fit_indicator_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
        ]
    xforms = [xf for xf in xforms if xf is not None]
    return {"outcome_name": outcome_name, "cols_to_copy": cols_to_copy, "xforms": xforms}


def perform_transform(*, x, plan):
    x = x.reset_index(inplace=False, drop=True)
    new_frames = [xfi.transform(x) for xfi in plan["xforms"]]
    # see if we want to copy over any columns
    cols = set([c for c in x.columns])
    to_copy = set(plan["cols_to_copy"].copy())
    to_copy = list(to_copy.intersection(cols))
    if len(to_copy) > 0:
        cp = x.loc[:, to_copy].copy()
        new_frames = [cp] + new_frames
    res = pandas.concat(new_frames, axis=1)
    res.reset_index(inplace=True, drop=True)
    return res


# assumes each y-aware variable produces one derived column
def cross_patch_refit_y_aware_cols(*, x, y, res, plan, cross_plan):
    for xf in plan["xforms"]:
        if not xf.need_cross_treatment_:
            continue
        incoming_column_name = xf.incoming_column_name_
        derived_column_name = xf.derived_column_names_[0]
        patches = [
            xf.refitter_(
                incoming_column_name=incoming_column_name,
                x=x[incoming_column_name][cp["train"]],
                y=y[cp["train"]],
                extra_args=xf.extra_args_,
            ).transform(x.loc[cp["app"], [incoming_column_name]])
            for cp in cross_plan
        ]
        # replace any missing sections with global average (sligth data leak potential)
        vals = pandas.concat([pi for pi in patches if pi is not None], axis=0)
        avg = numpy.mean(
            vals[numpy.logical_not(numpy.asarray(vals[derived_column_name].isnull()))]
        )[0]
        res[derived_column_name] = numpy.nan
        for i in range(len(cross_plan)):
            pi = patches[i]
            if pi is None:
                continue
            cp = cross_plan[i]
            res.loc[cp["app"], derived_column_name] = numpy.asarray(pi).reshape((len(pi)))
        res.loc[res[derived_column_name].isnull(), derived_column_name] = avg
    return res


def score_plan_variables(cross_frame, outcome, plan):
    def describe(xf):
        description = pandas.DataFrame({"variable": xf.derived_column_names_})
        description["treatment"] = xf.treatment_
        description["y_aware"] = xf.need_cross_treatment_
        return description

    var_table = pandas.concat([describe(xf) for xf in plan["xforms"]])
    var_table.reset_index(inplace=True, drop=True)
    sf = vtreat.util.score_variables(
        cross_frame, variables=var_table["variable"], outcome=outcome
    )
    score_frame = pandas.merge(var_table, sf, how="left", on=["variable"], sort=False)
    score_frame["_one"] = 1.0
    score_frame["vcount"] = score_frame.groupby("treatment")["_one"].transform("sum")
    score_frame.drop(["_one"], axis=1, inplace=True)
    score_frame["recommended"] = numpy.logical_and(
        score_frame["significance"] < 0.05,
        numpy.logical_and(
            score_frame["significance"] < 1 / score_frame["vcount"],
            numpy.logical_or(
                score_frame["PearsonR"] > 0, numpy.logical_not(score_frame["y_aware"])
            ),
        ),
    )
    return score_frame
