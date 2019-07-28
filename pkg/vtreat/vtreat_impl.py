# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:07:57 2019

@author: johnmount
"""

import numpy
import pandas

import vtreat.util
import vtreat.transform


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
        raise Exception("base method called")


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


def fit_regression_impact_code(*, incoming_column_name, x, y, extra_args, params):
    sf = vtreat.util.grouped_by_x_statistics(x, y)
    if sf.shape[0] <= 1:
        return None
    if params['use_hierarchical_estimate']:
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
        params=params
    )


def fit_regression_deviation_code(*, incoming_column_name, x, y, extra_args, params):
    sf = vtreat.util.grouped_by_x_statistics(x, y)
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
        params=params
    )


def fit_binomial_impact_code(*, incoming_column_name, x, y, extra_args, params):
    outcome_target = (extra_args["outcome_target"],)
    var_suffix = extra_args["var_suffix"]
    y = numpy.asarray(numpy.asarray(y) == outcome_target, dtype=numpy.float64)
    sf = vtreat.util.grouped_by_x_statistics(x, y)
    if sf.shape[0] <= 1:
        return None
    if params['use_hierarchical_estimate']:
        sf["_logit_code"] = numpy.log(sf["_hest"]) - numpy.log(sf["_gm"])
    else:
        sf["_logit_code"] = numpy.log(sf["_group_mean"]) - numpy.log(sf["_gm"])
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
        params=params
    )


class IndicatorCodeTransform(VarTransform):
    def __init__(self, incoming_column_name, derived_column_names, levels):
        VarTransform.__init__(
            self, incoming_column_name, derived_column_names, "indicator_code"
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


def fit_indicator_code(*, incoming_column_name, x, min_fraction):
    sf = pandas.DataFrame({incoming_column_name: x})
    na_posns = sf[incoming_column_name].isnull()
    sf.loc[na_posns, incoming_column_name] = "_NA_"
    counts = sf[incoming_column_name].value_counts()
    n = sf.shape[0]
    counts = counts[counts >= min_fraction*n]  # no more than 1/min_fraction symbols
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
        incoming_column_name, newcol, treatment="prevalence_code", code_book=sf
    )


def fit_numeric_outcome_treatment(
        *, X, y, var_list, outcome_name, cols_to_copy, params
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
            if 'missing_indicator' in params['coders']:
                xforms = xforms + [
                    IndicateMissingTransform(
                        incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                    )
                ]
    var_list = [co for co in var_list if (not (co in set(all_null)))]
    num_list = [co for co in var_list if vtreat.util.can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    if 'clean_copy' in params['coders']:
        for vi in num_list:
            summaryi = characterize_numeric(X[vi])
            if summaryi["varies"] and summaryi["has_range"]:
                xforms = xforms + [
                    CleanNumericTransform(
                        incoming_column_name=vi, replacement_value=summaryi["mean"]
                    )
                ]
    for vi in cat_list:
        if 'impact_code' in params['coders']:
            xforms = xforms + [
                fit_regression_impact_code(
                    incoming_column_name=vi, x=numpy.asarray(X[vi]), y=y, extra_args=None, params=params
                )
            ]
        if 'deviance_code' in params['coders']:
            xforms = xforms + [
                fit_regression_deviation_code(
                    incoming_column_name=vi, x=numpy.asarray(X[vi]), y=y, extra_args=None, params=params
                )
            ]
        if 'prevalence_code' in params['coders']:
            xforms = xforms + [
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            ]
        if 'indicator_code' in params['coders']:
            xforms = xforms + [
                fit_indicator_code(incoming_column_name=vi,
                                   x=numpy.asarray(X[vi]),
                                   min_fraction=params['indicator_min_fracton'])
            ]
    xforms = [xf for xf in xforms if xf is not None]
    for stp in params['user_transforms']:
        stp.fit(X=X[var_list], y=y)
    return {"outcome_name": outcome_name, "cols_to_copy": cols_to_copy, "xforms": xforms}


def fit_binomial_outcome_treatment(
        *, X, y, outcome_target, var_list, outcome_name, cols_to_copy, params
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
            if 'missing_indicator' in params['coders']:
                xforms = xforms + [
                    IndicateMissingTransform(
                        incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                    )
                ]
    var_list = [co for co in var_list if (not (co in set(all_null)))]
    num_list = [co for co in var_list if vtreat.util.can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    if 'clean_copy' in params['coders']:
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
        if 'logit_code' in params['coders']:
            xforms = xforms + [
                fit_binomial_impact_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    y=y,
                    extra_args=extra_args,
                    params=params
                )
            ]
        if 'prevalence_code' in params['coders']:
            xforms = xforms + [
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            ]
        if 'indicator_code' in params['coders']:
            xforms = xforms + [
                fit_indicator_code(incoming_column_name=vi,
                                   x=numpy.asarray(X[vi]),
                                   min_fraction=params['indicator_min_fracton'])
            ]
    xforms = [xf for xf in xforms if xf is not None]
    if len(xforms) <= 0:
        raise Exception("no variables created")
    return {"outcome_name": outcome_name, "cols_to_copy": cols_to_copy, "xforms": xforms}


def fit_multinomial_outcome_treatment(
        *, X, y, var_list, outcome_name, cols_to_copy, params
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
            if 'missing_indicator' in params['coders']:
                xforms = xforms + [
                    IndicateMissingTransform(
                        incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                    )
                ]
    outcomes = [oi for oi in set(y)]
    var_list = [co for co in var_list if (not (co in set(all_null)))]
    num_list = [co for co in var_list if vtreat.util.can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    if 'clean_copy' in params['coders']:
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
            if 'impact_code' in params['coders']:
                extra_args = {"outcome_target": outcome, "var_suffix": ("_" + str(outcome))}
                xforms = xforms + [
                    fit_binomial_impact_code(
                        incoming_column_name=vi,
                        x=numpy.asarray(X[vi]),
                        y=y,
                        extra_args=extra_args,
                        params=params
                    )
                ]
        if 'prevalence_code' in params['coders']:
            xforms = xforms + [
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            ]
        if 'indicator_code' in params['coders']:
            xforms = xforms + [
                fit_indicator_code(incoming_column_name=vi,
                                   x=numpy.asarray(X[vi]),
                                   min_fraction=params['indicator_min_fracton'])
            ]
    xforms = [xf for xf in xforms if xf is not None]
    if len(xforms) <= 0:
        raise Exception("no variables created")
    return {"outcome_name": outcome_name, "cols_to_copy": cols_to_copy, "xforms": xforms}


def fit_unsupervised_treatment(
        *, X, var_list, outcome_name, cols_to_copy, params
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
            if 'missing_indicator' in params['coders']:
                xforms = xforms + [
                    IndicateMissingTransform(
                        incoming_column_name=vi, derived_column_name=vi + "_is_bad"
                    )
                ]
    var_list = [co for co in var_list if (not (co in set(all_null)))]
    num_list = [co for co in var_list if vtreat.util.can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    if 'clean_copy' in params['coders']:
        for vi in num_list:
            summaryi = characterize_numeric(X[vi])
            if summaryi["varies"] and summaryi["has_range"]:
                xforms = xforms + [
                    CleanNumericTransform(
                        incoming_column_name=vi, replacement_value=summaryi["mean"]
                    )
                ]
    for vi in cat_list:
        if 'prevalence_code' in params['coders']:
            xforms = xforms + [
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            ]
        if 'indicator_code' in params['coders']:
            xforms = xforms + [
                fit_indicator_code(incoming_column_name=vi,
                                   x=numpy.asarray(X[vi]),
                                   min_fraction=params['indicator_min_fracton'])
            ]
    xforms = [xf for xf in xforms if xf is not None]
    if len(xforms) <= 0:
        raise Exception("no variables created")
    return {"outcome_name": outcome_name, "cols_to_copy": cols_to_copy, "xforms": xforms}


def perform_transform(*, x, transform, params):
    plan = transform.plan_
    x = x.reset_index(inplace=False, drop=True)
    new_frames = [xfi.transform(x) for xfi in plan["xforms"]]
    for stp in params['user_transforms']:
        frm = stp.transform(X=x)
        if frm is not None and frm.shape[1]>0:
            new_frames = new_frames + [ frm ]
    # see if we want to copy over any columns
    copy_set = set(plan["cols_to_copy"])
    to_copy = [ci for ci in x.columns if ci in copy_set]
    if len(to_copy) > 0:
        cp = x.loc[:, to_copy].copy()
        new_frames = [cp] + new_frames
    if len(new_frames) <= 0:
        raise Exception("no columns transformed")
    res = pandas.concat(new_frames, axis=1)
    res.reset_index(inplace=True, drop=True)
    return res


def limit_to_appropriate_columns(*, res, transform):
    plan = transform.plan_
    to_copy = set(plan["cols_to_copy"])
    if transform.params_['filter_to_recommended']:
        to_take = set([ci for ci in transform.score_frame_['variable'][transform.score_frame_['recommended']]])
    else:
        to_take = set([ci for ci in transform.score_frame_['variable'][transform.score_frame_['has_range']]])
    cols_to_keep = [ci for ci in res.columns if ci in to_copy or ci in to_take]
    if len(cols_to_keep) <= 0:
        raise Exception("no columns retained")
    return res[cols_to_keep]

# TODO: user transforms for Binomial case
# TODO: user transforms for Multinomial case
# TODO: user transforms for Unsupervised case

# assumes each y-aware variable produces one derived column
# also clears out refitter_ values to None
# TODO: patch in user transforms
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
            raise Exception("missing required column " + incoming_column_name)
        if xf.refitter_ is None:
            raise Exception("refitter is None: " + incoming_column_name + " -> " + derived_column_name)
        patches = [
            xf.refitter_(
                incoming_column_name=incoming_column_name,
                x=x[incoming_column_name][cp["train"]],
                y=y[cp["train"]],
                extra_args=xf.extra_args_,
                params=xf.params_
            ).transform(x.loc[cp["app"], [incoming_column_name]])
            for cp in cross_plan
        ]
        # replace any missing sections with global average (slight data leak potential)
        vals = pandas.concat([pi for pi in patches if pi is not None], axis=0)
        avg = numpy.mean(
            vals[numpy.logical_not(numpy.asarray(vals[derived_column_name].isnull()))]
        )[0]
        res[derived_column_name] = numpy.nan
        for i in range(len(cross_plan)):
            pi = patches[i]
            if pi is None:
                continue
            pi.reset_index(inplace=True, drop=True)
            cp = cross_plan[i]
            res.loc[cp["app"], derived_column_name] = numpy.asarray(pi).reshape((len(pi)))
        res.loc[res[derived_column_name].isnull(), derived_column_name] = avg
    for xf in plan["xforms"]:
        xf.refitter_ = None
    return res


def score_plan_variables(cross_frame, outcome, plan, params):
    def describe_xf(xf):
        description = pandas.DataFrame({
            "variable": xf.derived_column_names_})
        description["orig_variable"] = xf.incoming_column_name_
        description["treatment"] = xf.treatment_
        description["y_aware"] = xf.need_cross_treatment_
        return description

    def describe_ut(ut):
        description = pandas.DataFrame({
            "orig_variable": ut.incoming_vars_,
            "variable": ut.derived_vars_})
        description["treatment"] = ut.treatment_
        description["y_aware"] = ut.y_aware_
        return description

    var_table = pandas.concat(
        [describe_xf(xf) for xf in plan["xforms"]] +
        [describe_ut(ut) for ut in params['user_transforms'] if len(ut.incoming_vars_)>0 ])
    var_table.reset_index(inplace=True, drop=True)
    sf = vtreat.util.score_variables(
        cross_frame, variables=var_table["variable"], outcome=outcome
    )
    score_frame = pandas.merge(var_table, sf, how="left", on=["variable"], sort=False)
    score_frame["_one"] = 1.0
    score_frame["vcount"] = score_frame.groupby("treatment")["_one"].transform("sum")
    score_frame.drop(["_one"], axis=1, inplace=True)
    score_frame["recommended"] = numpy.logical_and(
        score_frame["has_range"],
        numpy.logical_and(
            numpy.logical_not(numpy.logical_or(
                numpy.isnan(score_frame["significance"]),
                numpy.isnan(score_frame["PearsonR"]))),
            numpy.logical_and(
                score_frame["significance"] < 1 / score_frame["vcount"],
                numpy.logical_or(
                    score_frame["PearsonR"] > 0,
                    numpy.logical_not(score_frame["y_aware"])))))
    return score_frame


def pseudo_score_plan_variables(cross_frame, plan):
    def describe(xf):
        description = pandas.DataFrame({
            "variable": xf.derived_column_names_})
        description["orig_variable"] = xf.incoming_column_name_
        description["treatment"] = xf.treatment_
        description["y_aware"] = xf.need_cross_treatment_
        return description

    score_frame = pandas.concat([describe(xf) for xf in plan["xforms"]])
    score_frame.reset_index(inplace=True, drop=True)
    score_frame["has_range"] = [numpy.max(cross_frame[c]) > numpy.min(cross_frame[c]) for c in score_frame['variable']]
    score_frame["PearsonR"] = numpy.nan
    score_frame["significance"] = numpy.nan
    score_frame["recommended"] = score_frame["has_range"].copy()
    score_frame["_one"] = 1.0
    score_frame["vcount"] = score_frame.groupby("treatment")["_one"].transform("sum")
    score_frame.drop(["_one"], axis=1, inplace=True)
    return score_frame
