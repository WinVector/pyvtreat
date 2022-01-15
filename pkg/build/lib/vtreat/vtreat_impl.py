"""
vtreat main implementation
"""

import abc
import math
import pprint
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy
import pandas

import vtreat.util
import vtreat.transform

import sklearn.base


bad_sentinel = "_NA_"


def replace_bad_with_sentinel(ar: List) -> numpy.ndarray:
    """
    Replace None/NaN entries in iterable with '_NA_'
    :param ar: iterable
    :return: one dimensional numpy array
    """

    ar = numpy.array(ar)
    bads = pandas.isnull(ar)
    ar[bads] = bad_sentinel
    return ar


def ready_data_frame(d) -> Tuple[pandas.DataFrame, type]:
    """
    Convert an array-like object to a data frame for processing.

    :param d: data frame like object to work with
    :return: dataframe with string-named columns
    """
    orig_type = type(d)
    if not isinstance(d, pandas.DataFrame):
        d = pandas.DataFrame(d)
    else:
        d = d.copy()
    d.columns = [str(c) for c in d.columns]  # convert column names to strings
    return d, orig_type


def back_to_orig_type_data_frame(
    d: pandas.DataFrame, orig_type: type
) -> Tuple[Any, List[str]]:
    """
    Convert data frame back to ndarray if that was the original type.

    :param d: data frame
    :param orig_type: type of original object
    :return: converted result
    """
    if not isinstance(d, pandas.DataFrame):
        raise TypeError(
            "Expected result to be a pandas.DataFrame, found: " + str(type(d))
        )
    columns = [c for c in d.columns]
    if orig_type == numpy.ndarray:
        d = numpy.asarray(d)
    return d, columns


class VarTransform(abc.ABC):
    """
    Base class for vtreat transforms
    """

    incoming_column_name_: str
    incoming_column_is_numeric_: bool
    derived_column_names_: List[str]
    need_cross_treatment_: bool
    treatment_: str
    refitter_: Optional[Callable]
    extra_args_: Optional[Dict[str, Any]]
    params_: Optional[Dict[str, Any]]

    def __init__(
        self,
        *,
        incoming_column_name: str,
        incoming_column_is_numeric: bool,
        derived_column_names: Iterable[str],
        treatment: str,
    ):
        """

        :param incoming_column_name:
        :param incoming_column_is_numeric:
        :param derived_column_names:
        :param treatment:
        """

        assert isinstance(incoming_column_name, str)
        assert isinstance(incoming_column_is_numeric, bool)
        assert not isinstance(derived_column_names, str)
        derived_column_names = list(derived_column_names)
        assert len(derived_column_names) > 0
        assert numpy.all([isinstance(dni, str) for dni in derived_column_names])
        assert isinstance(treatment, str)
        self.incoming_column_name_ = incoming_column_name
        self.incoming_column_is_numeric_ = incoming_column_is_numeric
        self.derived_column_names_ = derived_column_names
        self.treatment_ = treatment
        self.need_cross_treatment_ = False
        self.refitter_ = None
        self.extra_args_ = None
        self.params_ = None

    @abc.abstractmethod
    def transform(self, data_frame: pandas.DataFrame) -> pandas.DataFrame:
        """
        return a transformed data frame

        :rtype: pandas.DataFrame
        :param data_frame: incoming values
        :return: transformed values
        """

    @abc.abstractmethod
    def description_matrix(self) -> pandas.DataFrame:
        """
        Return description of transform as a data frame.

        :return: description of transform.
        """


class TreatmentPlan:
    """
    Class to carry treatment plans.
    """

    outcome_name: Optional[str]
    cols_to_copy: Tuple[str, ...]
    num_list: Tuple[str, ...]
    cat_list: Tuple[str, ...]
    xforms: Tuple[VarTransform, ...]

    def __init__(
            self,
            *,
            outcome_name: Optional[str] = None,
            cols_to_copy: Optional[Iterable[str]] = None,
            num_list: Optional[Iterable[str]] = None,
            cat_list: Optional[Iterable[str]] = None,
            xforms: Iterable[Optional[VarTransform]]):
        self.outcome_name = outcome_name
        if cols_to_copy is None:
            self.cols_to_copy = tuple()
        else:
            assert not isinstance(cols_to_copy, str)
            self.cols_to_copy = tuple(cols_to_copy)
        if num_list is None:
            self.num_list = tuple()
        else:
            assert not isinstance(num_list, str)
            self.num_list = tuple(num_list)
        if cat_list is None:
            self.cat_list = tuple()
        else:
            assert not isinstance(cat_list, str)
            self.cat_list = tuple(cat_list)
        non_empty_xforms: List[VarTransform] = [x for x in xforms if x is not None]
        self.xforms = tuple(non_empty_xforms)
        for c in self.cols_to_copy:
            assert isinstance(c, str)
        for c in self.num_list:
            assert isinstance(c, str)
        for c in self.cat_list:
            assert isinstance(c, str)
        if len(self.xforms) < 1:
            raise ValueError("no treatments generated")
        for x in self.xforms:
            assert isinstance(x, VarTransform)


class MappedCodeTransform(VarTransform):
    """Class for transforms that are a dictionary mapping of strings to numeric values"""

    def __init__(
        self,
        *,
        incoming_column_name: str,
        derived_column_name: str,
        treatment: str,
        code_book: pandas.DataFrame,
    ):
        """

        :param incoming_column_name:
        :param derived_column_name:
        :param treatment:
        :param code_book: Pandas dataframe mapping values to impact codes
        """

        VarTransform.__init__(
            self,
            incoming_column_name=incoming_column_name,
            incoming_column_is_numeric=False,
            derived_column_names=[derived_column_name],
            treatment=treatment,
        )
        self.code_book_ = code_book

    def transform(self, data_frame: pandas.DataFrame) -> pandas.DataFrame:
        """
        return a transformed data frame

        :rtype: pandas.DataFrame
        :param data_frame: incoming values
        :return: transformed values
        """

        incoming_column_name = self.incoming_column_name_
        derived_column_name = self.derived_column_names_[0]
        sf = pandas.DataFrame({incoming_column_name: data_frame[incoming_column_name]})
        bad_posns = vtreat.util.is_bad(sf[incoming_column_name])
        sf.loc[bad_posns, incoming_column_name] = bad_sentinel
        res = pandas.merge(
            sf, self.code_book_, on=[self.incoming_column_name_], how="left", sort=False
        )  # ordered by left table rows
        # could also try pandas .map()
        res = res[[derived_column_name]].copy()
        res.loc[vtreat.util.is_bad(res[derived_column_name]), derived_column_name] = 0.0
        return res

    def description_matrix(self) -> pandas.DataFrame:
        """
        Return description of transform as a data frame.

        :return: description of transform.
        """

        description = pandas.DataFrame(
            {
                "treatment_class": "MappedCodeTransform",
                "treatment": self.treatment_,
                "orig_var": self.incoming_column_name_,
                "orig_was_numeric": self.incoming_column_is_numeric_,
                "variable": self.derived_column_names_[0],
                "value": replace_bad_with_sentinel(
                    self.code_book_[self.incoming_column_name_]
                ),
                "replacement": self.code_book_[self.derived_column_names_[0]].copy(),
            }
        )
        return description


class YAwareMappedCodeTransform(MappedCodeTransform):
    """Class for transforms that are a y-aware dictionary mapping of values"""

    def __init__(
        self,
        incoming_column_name: str,
        derived_column_name: str,
        treatment: str,
        code_book: pandas.DataFrame,
        refitter,
        extra_args: Optional[Dict[str, Any]],
        params: Dict[str, Any],
    ):
        """

        :param incoming_column_name: name of incoming column
        :param derived_column_name: name of incoming column
        :param treatment: name of treatment
        :param code_book: pandas data frame mapping values to codes
        :param refitter: function to re-fit
        :param extra_args: extra args for fit_* functions
        :param params: configuration control parameters
        """

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
    """Class for numeric column cleaner."""

    def __init__(self,
                 *,
                 incoming_column_name: str,
                 replacement_value: float):
        """

        :param incoming_column_name:
        :param replacement_value:
        """

        VarTransform.__init__(
            self,
            incoming_column_name=incoming_column_name,
            incoming_column_is_numeric=True,
            derived_column_names=[incoming_column_name],
            treatment="clean_copy",
        )
        self.replacement_value_ = replacement_value

    def transform(self, data_frame: pandas.DataFrame) -> pandas.DataFrame:
        """
        return a transformed data frame

        :rtype: pandas.DataFrame
        :param data_frame: incoming values
        :return: transformed values
        """

        col = vtreat.util.safe_to_numeric_array(data_frame[self.incoming_column_name_])
        bad_posns = vtreat.util.is_bad(col)
        col[bad_posns] = self.replacement_value_
        res = pandas.DataFrame({self.derived_column_names_[0]: col})
        return res

    def description_matrix(self) -> pandas.DataFrame:
        """
        Return description of transform as a data frame.

        :return: description of transform.
        """

        description = pandas.DataFrame(
            {
                "treatment_class": ["CleanNumericTransform"],
                "treatment": [self.treatment_],
                "orig_var": [self.incoming_column_name_],
                "orig_was_numeric": [self.incoming_column_is_numeric_],
                "variable": [self.derived_column_names_[0]],
                "value": [bad_sentinel],
                "replacement": [self.replacement_value_],
            }
        )
        return description


class IndicateMissingTransform(VarTransform):
    """Class for missing value indicator."""

    def __init__(self,
                 *,
                 incoming_column_name: str,
                 incoming_column_is_numeric: bool,
                 derived_column_name: str):
        """

        :param incoming_column_name:
        :param derived_column_name:
        """

        VarTransform.__init__(
            self,
            incoming_column_name=incoming_column_name,
            incoming_column_is_numeric=incoming_column_is_numeric,
            derived_column_names=[derived_column_name],
            treatment="missing_indicator",
        )

    def transform(self, data_frame: pandas.DataFrame) -> pandas.DataFrame:
        """
        return a transformed data frame

        :rtype: pandas.DataFrame
        :param data_frame: incoming values
        :return: transformed values
        """

        col = vtreat.util.is_bad(data_frame[self.incoming_column_name_])
        res = pandas.DataFrame({self.derived_column_names_[0]: col + 0.0})
        return res

    def description_matrix(self) -> pandas.DataFrame:
        """
        Return description of transform as a data frame.

        :return: description of transform.
        """
        description = pandas.DataFrame(
            {
                "treatment_class": ["IndicateMissingTransform"],
                "treatment": [self.treatment_],
                "orig_var": [self.incoming_column_name_],
                "orig_was_numeric": [self.incoming_column_is_numeric_],
                "variable": [self.derived_column_names_[0]],
                "value": [bad_sentinel],
                "replacement": [1.0],
            }
        )
        return description


def fit_clean_code(
    *,
    incoming_column_name: str,
    x,
    params: Dict[str, Any],
    imputation_map: Dict[str, Any],
) -> Optional[VarTransform]:
    """
    Fit numeric clean column imputation transform

    :param incoming_column_name: name of column
    :param x: training values for column
    :param params: control parameter dictionary
    :param imputation_map: per-column map to imputation strategies or values
    :return: transform
    """

    if not vtreat.util.numeric_has_range(x):
        return None
    replacement = params["missingness_imputation"]
    try:
        replacement = imputation_map[incoming_column_name]
    except KeyError:
        pass
    if vtreat.util.can_convert_v_to_numeric(replacement):
        replacement_value = 0.0 + replacement
    elif callable(replacement):
        replacement_value = vtreat.util.summarize_column(x, fn=replacement)
    else:
        raise TypeError(
            "unexpected imputation type "
            + str(type(replacement))
            + " ("
            + incoming_column_name
            + ")"
        )
    if (
        pandas.isnull(replacement_value)
        or math.isnan(replacement_value)
        or math.isinf(replacement_value)
    ):
        raise ValueError(
            "replacement was bad "
            + incoming_column_name
            + ": "
            + str(replacement_value)
        )
    return CleanNumericTransform(
        incoming_column_name=incoming_column_name, replacement_value=replacement_value
    )


def fit_regression_impact_code(
    *,
    incoming_column_name: str,
    x,
    y,
    extra_args: Optional[Dict[str, Any]],
    params: Dict[str, Any],
) -> Optional[VarTransform]:
    """
    Fit regression impact code transform

    :param incoming_column_name:
    :param x: training explanatory values
    :param y: training dependent values
    :param extra_args: optional extra arguments for fit_ methods
    :param params: control parameter dictionary
    :return:
    """

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


def fit_regression_deviation_code(
    *,
    incoming_column_name: str,
    x,
    y,
    extra_args: Optional[Dict[str, Any]],
    params: Dict[str, Any],
) -> Optional[VarTransform]:
    """
    Fit regression deviation code transform

    :param incoming_column_name:
    :param x: training explanatory values
    :param y: training dependent values
    :param extra_args: optional extra arguments for fit_ methods
    :param params: control parameter dictionary
    :return:
    """

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


def fit_binomial_impact_code(
    *,
    incoming_column_name: str,
    x,
    y,
    extra_args: Dict[str, Any],
    params: Dict[str, Any],
) -> Optional[VarTransform]:
    """
    Fit categorical impact code.

    :param incoming_column_name:
    :param x: training explanatory values
    :param y: training dependent values
    :param extra_args: required extra arguments for fit_ methods
    :param params: control parameter dictionary
    :return:
    """
    outcome_target = (
        extra_args["outcome_target"],
    )  # TODO: document why this is a tuple
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
    """Class for indicator codes"""

    def __init__(
        self,
        *,
        incoming_column_name: str,
        derived_column_names: List[str],
        levels: List,
        sparse_indicators: bool = False,
    ):
        """

        :param incoming_column_name:
        :param derived_column_names:
        :param levels: leves we are encoding to indicators
        :param sparse_indicators: if True use sparse data structure
        """
        VarTransform.__init__(
            self,
            incoming_column_name=incoming_column_name,
            incoming_column_is_numeric=False,
            derived_column_names=derived_column_names,
            treatment="indicator_code",
        )
        self.levels_ = levels
        self.sparse_indicators_ = sparse_indicators

    def transform(self, data_frame):
        """
        return a transformed data frame

        :rtype: pandas.DataFrame
        :param data_frame: incoming values
        :return: transformed values
        """

        incoming_column_name = self.incoming_column_name_
        sf = pandas.DataFrame({incoming_column_name: data_frame[incoming_column_name]})
        bad_posns = vtreat.util.is_bad(sf[incoming_column_name])
        sf.loc[bad_posns, incoming_column_name] = bad_sentinel
        col = sf[self.incoming_column_name_]

        def f(i: int):
            """transform one column"""
            v = numpy.asarray(col == self.levels_[i]) + 0.0  # return numeric 0/1 coding
            if self.sparse_indicators_:
                v = pandas.arrays.SparseArray(v, fill_value=0.0)
            return v

        res = None
        for ii in range(len(self.levels_)):
            if res is None:
                res = pandas.DataFrame({self.derived_column_names_[ii]: f(ii)})
            else:
                res[self.derived_column_names_[ii]] = f(ii)
        return res

    def description_matrix(self) -> pandas.DataFrame:
        """
        Return description of transform as a data frame.

        :return: description of transform.
        """

        description = pandas.DataFrame(
            {
                "treatment_class": "IndicatorCodeTransform",
                "treatment": self.treatment_,
                "orig_var": self.incoming_column_name_,
                "orig_was_numeric": self.incoming_column_is_numeric_,
                "variable": self.derived_column_names_.copy(),
                "value": replace_bad_with_sentinel(self.levels_),
                "replacement": 1.0,
            }
        )
        return description


def fit_indicator_code(
    *,
    incoming_column_name: str,
    x,
    min_fraction: float = 0.0,
    max_levels: Optional[int] = None,
    sparse_indicators: bool = False,
) -> Optional[VarTransform]:
    """
    Fit indicator codes

    :param incoming_column_name:
    :param x: training explanatory variables
    :param min_fraction:
    :param max_levels:
    :param sparse_indicators:
    :return:
    """

    sf = pandas.DataFrame({incoming_column_name: x})
    bad_posns = vtreat.util.is_bad(sf[incoming_column_name])
    sf.loc[bad_posns, incoming_column_name] = bad_sentinel
    counts = sf[incoming_column_name].value_counts()
    n = sf.shape[0]
    counts = counts[counts > 0]
    counts = counts[counts >= min_fraction * n]  # no more than 1/min_fraction symbols
    levels = [str(v) for v in counts.index]
    if (max_levels is not None) and (len(levels) > max_levels):
        level_frame = pandas.DataFrame({"levels": levels, "counts": counts})
        level_frame.sort_values(
            by=["counts", "levels"],
            ascending=[False, True],
            inplace=True,
            axis=0,
            ignore_index=True,
        )
        value_vec = level_frame["levels"].values
        levels = [value_vec[i] for i in range(max_levels)]
    if len(levels) < 1:
        return None
    return IndicatorCodeTransform(
        incoming_column_name=incoming_column_name,
        derived_column_names=vtreat.util.build_level_codes(incoming_column_name, levels),
        levels=levels,
        sparse_indicators=sparse_indicators,
    )


def fit_prevalence_code(incoming_column_name: str, x) -> Optional[VarTransform]:
    """
    Fit a prevalence code

    :param incoming_column_name:
    :param x: training explanatory values
    :return:
    """

    sf = pandas.DataFrame({"x": x})
    bad_posns = vtreat.util.is_bad(sf["x"])
    sf.loc[bad_posns, "x"] = bad_sentinel
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
        incoming_column_name=incoming_column_name,
        derived_column_name=newcol,
        treatment="prevalence_code",
        code_book=sf
    )


# noinspection PyPep8Naming
def _prepare_variable_lists(
    *,
    X,
    cols_to_copy: Optional[Iterable[str]],
    var_list: Optional[Iterable[str]],
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Prepare lists of variables for variable treatment.

    :param X: dependent variables
    :param cols_to_copy: columns to copy
    :param var_list: dependent variable names, if empty all non outcome and copy columns are used
    :return: cat_list, cols_to_copy, mis_list, num_list, var_list lists
    """
    if var_list is None:
        var_list = list(X.columns)
    else:
        var_list = list(var_list)
    if len(var_list) < 1:
        var_list = list(X.columns)
    assert len(var_list) > 0
    if cols_to_copy is None:
        cols_to_copy = []
    else:
        cols_to_copy = list(cols_to_copy)
    copy_set = set(cols_to_copy)
    var_list = [co for co in var_list if (not (co in copy_set))]
    v_counts = {v: vtreat.util.get_unique_value_count(X[v]) for v in var_list}
    var_list = {v for v in var_list if v_counts[v] > 1}
    if len(var_list) <= 0:
        raise ValueError("no variables")
    n = X.shape[0]
    all_bad: List[str] = []
    mis_list: List[str] = []
    for vi in var_list:
        n_bad = numpy.sum(vtreat.util.is_bad(X[vi]))
        if n_bad >= n:
            all_bad.append(vi)
        if (n_bad > 0) and (n_bad < n):
            mis_list.append(vi)
    var_list = [co for co in var_list if (not (co in set(all_bad)))]
    num_list = [co for co in var_list if vtreat.util.can_convert_v_to_numeric(X[co])]
    cat_list = [co for co in var_list if co not in set(num_list)]
    id_like = [co for co in cat_list if v_counts[co] >= n]
    if len(id_like) > 0:
        warnings.warn(
            "variable(s) "
            + ", ".join(id_like)
            + " have unique values per-row, dropping"
        )
        cat_list = [co for co in var_list if co not in set(id_like)]
    return cat_list, cols_to_copy, mis_list, num_list, var_list


# noinspection PyPep8Naming
def fit_numeric_outcome_treatment(
    *,
    X,
    y,
    var_list: Optional[Iterable[str]],
    outcome_name: str,
    cols_to_copy: Optional[Iterable[str]],
    params: Dict[str, Any],
    imputation_map: Dict[str, Any],
) -> TreatmentPlan:
    """
    Fit set of treatments in a regression situation.

    :param X: training explanatory values
    :param y: training dependent values
    :param var_list: list of dependent variable names, if empty all non outcome and copy columns are used
    :param outcome_name: name for outcome column
    :param cols_to_copy: list of columns to copy to output
    :param params: control parameter dictionary
    :param imputation_map: per-column map to imputation strategies or values
    :return: transform plan
    """
    cat_list, cols_to_copy, mis_list, num_list, var_list = _prepare_variable_lists(
        X=X, cols_to_copy=cols_to_copy, var_list=var_list
    )
    xforms: List[Optional[VarTransform]] = []
    if "missing_indicator" in params["coders"]:
        num_set = set(num_list)
        for vi in mis_list:
            xforms.append(
                IndicateMissingTransform(
                    incoming_column_name=vi,
                    incoming_column_is_numeric=vi in num_set,
                    derived_column_name=vi + "_is_bad"
                )
            )
    if "clean_copy" in params["coders"]:
        for vi in num_list:
            xform = fit_clean_code(
                incoming_column_name=vi,
                x=X[vi],
                params=params,
                imputation_map=imputation_map,
            )
            if xform is not None:
                xforms.append(xform)
    for vi in cat_list:
        if "impact_code" in params["coders"]:
            xforms.append(
                fit_regression_impact_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    y=y,
                    extra_args=None,
                    params=params,
                )
            )
        if "deviation_code" in params["coders"]:
            xforms.append(
                fit_regression_deviation_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    y=y,
                    extra_args=None,
                    params=params,
                )
            )
        if "prevalence_code" in params["coders"]:
            xforms.append(
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            )
        if "indicator_code" in params["coders"]:
            xforms.append(
                fit_indicator_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    min_fraction=params["indicator_min_fraction"],
                    sparse_indicators=params["sparse_indicators"],
                )
            )
    for stp in params["user_transforms"]:
        stp.fit(X=X[var_list], y=y)
    return TreatmentPlan(
        outcome_name=outcome_name,
        cols_to_copy=cols_to_copy,
        num_list=num_list,
        cat_list=cat_list,
        xforms=xforms,
    )


# noinspection PyPep8Naming
def fit_binomial_outcome_treatment(
    *,
    X,
    y,
    outcome_target,
    var_list: Optional[Iterable[str]],
    outcome_name: str,
    cols_to_copy: Optional[Iterable[str]],
    params: Dict[str, Any],
    imputation_map: Dict[str, Any],
) -> TreatmentPlan:
    """

    :param X: training explanatory values
    :param y: training dependent values
    :param outcome_target: dependent value to consider positive or in class
    :param var_list: list of variables to process, if empty all non outcome and copy columns are used
    :param outcome_name: name for outcome column
    :param cols_to_copy: list of columns to copy to output
    :param params: control parameter dictionary
    :param imputation_map: per-column map to imputation strategies or values
    :return: transform plan
    """
    cat_list, cols_to_copy, mis_list, num_list, var_list = _prepare_variable_lists(
        X=X, cols_to_copy=cols_to_copy, var_list=var_list
    )
    xforms: List[Optional[VarTransform]] = []
    if "missing_indicator" in params["coders"]:
        num_set = set(num_list)
        for vi in mis_list:
            xforms.append(
                IndicateMissingTransform(
                    incoming_column_name=vi,
                    incoming_column_is_numeric=vi in num_set,
                    derived_column_name=vi + "_is_bad"
                )
            )
    if "clean_copy" in params["coders"]:
        for vi in num_list:
            xform = fit_clean_code(
                incoming_column_name=vi,
                x=X[vi],
                params=params,
                imputation_map=imputation_map,
            )
            if xform is not None:
                xforms.append(xform)
    extra_args = {"outcome_target": outcome_target, "var_suffix": ""}
    for vi in cat_list:
        if "logit_code" in params["coders"]:
            xforms.append(
                fit_binomial_impact_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    y=y,
                    extra_args=extra_args,
                    params=params,
                )
            )
        if "prevalence_code" in params["coders"]:
            xforms.append(
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            )
        if "indicator_code" in params["coders"]:
            xforms.append(
                fit_indicator_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    min_fraction=params["indicator_min_fraction"],
                    sparse_indicators=params["sparse_indicators"],
                )
            )
    for stp in params["user_transforms"]:
        stp.fit(X=X[var_list], y=y)
    return TreatmentPlan(
        outcome_name=outcome_name,
        cols_to_copy=cols_to_copy,
        num_list=num_list,
        cat_list=cat_list,
        xforms=xforms,
    )


# noinspection PyPep8Naming
def fit_multinomial_outcome_treatment(
    *,
    X,
    y,
    var_list: Optional[Iterable[str]],
    outcome_name: str,
    cols_to_copy: Optional[Iterable[str]],
    params: Dict[str, Any],
    imputation_map: Dict[str, Any],
) -> TreatmentPlan:
    """
    Fit a variable treatment for multinomial outcomes.

    :param X: training explanatory values
    :param y: training dependent values
    :param var_list: list of variables to process, if empty all non outcome and copy columns are used
    :param outcome_name: name for outcome column
    :param cols_to_copy: list of columns to copy to output
    :param params: control parameter dictionary
    :param imputation_map: per-column map to imputation strategies or values
    :return:
    """

    outcomes = [oi for oi in set(y)]
    assert len(outcomes) > 1
    cat_list, cols_to_copy, mis_list, num_list, var_list = _prepare_variable_lists(
        X=X, cols_to_copy=cols_to_copy, var_list=var_list
    )
    xforms: List[Optional[VarTransform]] = []
    if "missing_indicator" in params["coders"]:
        num_set = set(num_list)
        for vi in mis_list:
            xforms.append(
                IndicateMissingTransform(
                    incoming_column_name=vi,
                    incoming_column_is_numeric=vi in num_set,
                    derived_column_name=vi + "_is_bad"
                )
            )
    if "clean_copy" in params["coders"]:
        for vi in num_list:
            xform = fit_clean_code(
                incoming_column_name=vi,
                x=X[vi],
                params=params,
                imputation_map=imputation_map,
            )
            if xform is not None:
                xforms.append(xform)
    for vi in cat_list:
        for outcome in outcomes:
            if "impact_code" in params["coders"]:
                extra_args = {
                    "outcome_target": outcome,
                    "var_suffix": ("_" + str(outcome)),
                }
                xforms.append(
                    fit_binomial_impact_code(
                        incoming_column_name=vi,
                        x=numpy.asarray(X[vi]),
                        y=y,
                        extra_args=extra_args,
                        params=params,
                    )
                )
        if "prevalence_code" in params["coders"]:
            xforms.append(
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            )
        if "indicator_code" in params["coders"]:
            xforms.append(
                fit_indicator_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    min_fraction=params["indicator_min_fraction"],
                    sparse_indicators=params["sparse_indicators"],
                )
            )
    for stp in params["user_transforms"]:
        stp.fit(X=X[var_list], y=y)
    return TreatmentPlan(
        outcome_name=outcome_name,
        cols_to_copy=cols_to_copy,
        num_list=num_list,
        cat_list=cat_list,
        xforms=xforms,
    )


# noinspection PyPep8Naming
def fit_unsupervised_treatment(
    *,
    X,
    var_list: Optional[Iterable[str]],
    outcome_name: str,
    cols_to_copy: Optional[Iterable[str]],
    params: Dict[str, Any],
    imputation_map: Dict[str, Any],
) -> TreatmentPlan:
    """
    Fit a data treatment in the unsupervised case.

    :param X: training explanatory values
    :param var_list: list of variables to process, if empty all non copy columns are used
    :param outcome_name: name for outcome column
    :param cols_to_copy: list of columns to copy to output
    :param params: control parameter dictionary
    :param imputation_map: per-column map to imputation strategies or values
    :return:
    """

    cat_list, cols_to_copy, mis_list, num_list, var_list = _prepare_variable_lists(
        X=X, cols_to_copy=cols_to_copy, var_list=var_list
    )
    xforms: List[Optional[VarTransform]] = []
    if "missing_indicator" in params["coders"]:
        num_set = set(num_list)
        for vi in mis_list:
            xforms.append(
                IndicateMissingTransform(
                    incoming_column_name=vi,
                    incoming_column_is_numeric=vi in num_set,
                    derived_column_name=vi + "_is_bad"
                )
            )
    if "clean_copy" in params["coders"]:
        for vi in num_list:
            xform = fit_clean_code(
                incoming_column_name=vi,
                x=X[vi],
                params=params,
                imputation_map=imputation_map,
            )
            if xform is not None:
                xforms.append(xform)
    for vi in cat_list:
        if "prevalence_code" in params["coders"]:
            xforms.append(
                fit_prevalence_code(incoming_column_name=vi, x=numpy.asarray(X[vi]))
            )
        if "indicator_code" in params["coders"]:
            xforms.append(
                fit_indicator_code(
                    incoming_column_name=vi,
                    x=numpy.asarray(X[vi]),
                    min_fraction=params["indicator_min_fraction"],
                    max_levels=params["indicator_max_levels"],
                    sparse_indicators=params["sparse_indicators"],
                )
            )
    for stp in params["user_transforms"]:
        stp.fit(X=X[var_list], y=None)
    return TreatmentPlan(
        outcome_name=outcome_name,
        cols_to_copy=cols_to_copy,
        num_list=num_list,
        cat_list=cat_list,
        xforms=xforms,
    )


def pre_prep_frame(
    x: pandas.DataFrame,
    *,
    col_list: Optional[Iterable[str]],
    cols_to_copy: Optional[Iterable[str]],
    cat_cols: Optional[Iterable[str]] = None,
) -> pandas.DataFrame:
    """
    Create a copy of pandas.DataFrame x restricted to col_list union cols_to_copy with col_list - cols_to_copy
    converted to only string and numeric types.  New pandas.DataFrame has trivial indexing.  If col_list
    is empty it is interpreted as all columns.

    :param x:
    :param col_list:
    :param cols_to_copy:
    :param cat_cols:
    :return:
    """

    if cols_to_copy is None:
        cols_to_copy = []
    else:
        cols_to_copy = list(cols_to_copy)
    if col_list is None:
        col_list = []
    else:
        col_list = list(col_list)
    if len(col_list) <= 0:
        col_list = list(x.columns)
    x_set = set(x.columns)
    col_set = set(col_list)
    for ci in cols_to_copy:
        if (ci in x_set) and (ci not in col_set):
            col_list.append(ci)
    col_set = set(col_list)
    missing_cols = col_set - x_set
    if len(missing_cols) > 0:
        raise KeyError("referred to not-present columns " + str(missing_cols))
    cset = set(cols_to_copy)
    if len(col_list) <= 0:
        raise ValueError("no variables")
    x = x.loc[:, col_list]
    x = x.reset_index(inplace=False, drop=True)
    cat_col_set = None
    if cat_cols is not None:
        cat_col_set = set(cat_cols)
    for c in x.columns:
        if c in cset:
            continue
        bad_ind = vtreat.util.is_bad(x[c])
        if cat_col_set is not None:
            numeric_col = c not in cat_col_set
        else:
            numeric_col = vtreat.util.can_convert_v_to_numeric(x[c])
        if numeric_col:
            x[c] = vtreat.util.safe_to_numeric_array(x[c])
        else:
            # https://stackoverflow.com/questions/22231592/pandas-change-data-type-of-series-to-string
            x[c] = x[c].astype(str)
        x.loc[bad_ind, c] = None
    return x


def _mean_of_single_column_pandas_list(val_list: Iterable[pandas.DataFrame]) -> float:
    """
    Compute the mean of non-nan positions of a bunch of single column data frames

    :param val_list: a list of single column Pandas data frames
    :return: mean, or numpy.nan if there are no values
    """

    if val_list is None:
        return numpy.nan
    val_list = [v for v in val_list]
    if len(val_list) <= 0:
        return numpy.nan
    if len(val_list) <= 1:
        d = val_list[0]
    else:
        d = pandas.concat(val_list, axis=0, sort=False)
    col = d.columns[0]
    d = d.loc[numpy.logical_not(vtreat.util.is_bad(d[col])), [col]]
    if d.shape[0] < 1:
        return numpy.nan
    res = numpy.mean(d[col].values)
    assert isinstance(res, float)  # type hint for PyCharm IDE
    return res


def cross_patch_refit_y_aware_cols(
    *, x: pandas.DataFrame, y, res: pandas.DataFrame, plan: TreatmentPlan, cross_plan
) -> None:
    """
    Re fit the y-aware columns according to cross plan.
    Clears out refitter_ values to None.
    Assumes each y-aware variable produces one derived column.

    :param x: explanatory values
    :param y: dependent values
    :param res: transformed frame to patch results into, altered
    :param plan: fitting plan
    :param cross_plan: cross validation plan
    :return: no return, res is altered in place
    """

    if cross_plan is None or len(cross_plan) <= 1:
        for xf in plan.xforms:
            xf.refitter_ = None
        return
    incoming_colset = set(x.columns)
    derived_colset = set(res.columns)
    for xf in plan.xforms:
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
            """conditional patching"""
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
        avg = _mean_of_single_column_pandas_list(
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
            ).reshape((len(pi),))
        res.loc[vtreat.util.is_bad(res[derived_column_name]), derived_column_name] = avg
    for xf in plan.xforms:
        xf.refitter_ = None


def cross_patch_user_y_aware_cols(
    *, x: pandas.DataFrame, y, res: pandas.DataFrame, params: Dict[str, Any], cross_plan
) -> None:
    """
    Re fit the user y-aware columns according to cross plan.
    Assumes each y-aware variable produces one derived column.

    :param x: explanatory values
    :param y: dependent values
    :param res: transformed frame to patch results into, altered
    :param params: control parameter dictionary
    :param cross_plan: cross validation plan
    :return: no return, res altered in place
    """
    if cross_plan is None or len(cross_plan) <= 1:
        return
    incoming_colset = set(x.columns)
    derived_colset = set(res.columns)
    if len(derived_colset) <= 0:
        return
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
            avg = _mean_of_single_column_pandas_list(
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
                res.loc[cp["app"], col] = numpy.asarray(pi[col]).reshape((len(pi),))
            res.loc[vtreat.util.is_bad(res[col]), col] = avg


def score_plan_variables(
    cross_frame: pandas.DataFrame,
    outcome,
    plan: TreatmentPlan,
    params: Dict[str, Any],
    *,
    is_classification: bool = False,
) -> pandas.DataFrame:
    """
    Quality score variables to build up score frame.

    :param cross_frame: cross transformed explanatory variables
    :param outcome: dependent variable
    :param plan: treatment plan
    :param params: control parameter dictionary
    :param is_classification: logical, if True classification if False regression
    :return: score frame
    """

    def describe_xf(xf):
        """describe variable transform"""
        description = pandas.DataFrame({"variable": xf.derived_column_names_})
        description["orig_variable"] = xf.incoming_column_name_
        description["treatment"] = xf.treatment_
        description["y_aware"] = xf.need_cross_treatment_
        return description

    def describe_ut(ut):
        """describe user variable transform"""
        description = pandas.DataFrame(
            {"orig_variable": ut.incoming_vars_, "variable": ut.derived_vars_}
        )
        description["treatment"] = ut.treatment_
        description["y_aware"] = ut.y_aware_
        return description

    var_table = pandas.concat(
        [describe_xf(xf) for xf in plan.xforms]
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
        is_classification=is_classification,
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


def pseudo_score_plan_variables(
    *, cross_frame, plan: TreatmentPlan, params: Dict[str, Any]
) -> pandas.DataFrame:
    """
    Build a score frame look-alike for unsupervised case.

    :param cross_frame: cross transformed explanatory variables
    :param plan: treatment plan
    :param params: control parameter dictionary
    :return: score frame
    """

    def describe_xf(xf):
        """describe variable transform"""
        description = pandas.DataFrame({"variable": xf.derived_column_names_})
        description["orig_variable"] = xf.incoming_column_name_
        description["treatment"] = xf.treatment_
        description["y_aware"] = xf.need_cross_treatment_
        return description

    def describe_ut(ut):
        """describe user variable transform"""
        description = pandas.DataFrame(
            {"orig_variable": ut.incoming_vars_, "variable": ut.derived_vars_}
        )
        description["treatment"] = ut.treatment_
        description["y_aware"] = ut.y_aware_
        return description

    score_frame = pandas.concat(
        [describe_xf(xf) for xf in plan.xforms]
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


class VariableTreatment(abc.ABC, sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Class for variable treatments, implements much of the sklearn pipeline/transformer
    API. https://sklearn-template.readthedocs.io/en/latest/user_guide.html#transformer
    """

    result_restriction: Optional[Set[str]]
    outcome_name_: Optional[str]
    outcome_target_: Optional[Any]
    var_list_: List[str]
    cols_to_copy_: List[str]
    params_: Dict[str, Any]
    plan_: Optional[TreatmentPlan]
    score_frame_: Optional[pandas.DataFrame]
    imputation_map_: Dict[str, Callable]
    last_fit_x_id_: Optional[str]
    cross_plan_: Optional[List[Dict[str, List[int]]]]
    cross_rows_: Optional[int]
    last_result_columns: Optional[List[str]]

    def __init__(
        self,
        *,
        var_list: Optional[Iterable[str]] = None,
        outcome_name: Optional[str] = None,
        outcome_target: Optional[Any] = None,
        cols_to_copy: Optional[Iterable[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        imputation_map: Optional[Dict[str, Any]] = None,
    ):
        """

        :param var_list: variables we intend to encode, empty means all
        :param outcome_name: column name of outcome
        :param outcome_target: outcome column value we consider in class or True
        :param cols_to_copy: columns to not process, but copy over
        :param params: control and configuration parameters
        :param imputation_map: per column imputation strategies or values
        """
        if var_list is None:
            var_list = []
        else:
            var_list = vtreat.util.unique_items_in_order(var_list)
        if cols_to_copy is None:
            cols_to_copy = []
        else:
            cols_to_copy = vtreat.util.unique_items_in_order(cols_to_copy)
        if (outcome_name is not None) and (outcome_name not in set(cols_to_copy)):
            cols_to_copy.append(outcome_name)
        confused = set(cols_to_copy).intersection(set(var_list))
        if len(confused) > 0:
            raise ValueError(
                "variables in treatment plan and non-treatment: " + ", ".join(confused)
            )
        if imputation_map is None:
            imputation_map = dict()
        self.outcome_name_ = outcome_name
        self.outcome_target_ = outcome_target
        self.var_list_ = [vi for vi in var_list if vi not in set(cols_to_copy)]
        self.cols_to_copy_ = cols_to_copy
        if params is not None:
            self.params_ = params.copy()
        else:
            self.params_ = dict()
        self.imputation_map_ = imputation_map.copy()
        self.plan_ = None
        self.score_frame_ = None
        self.cross_rows_ = None
        self.cross_plan_ = None
        self.last_fit_x_id_ = None
        self.last_result_columns = None
        self.result_restriction = None
        self.clear()

    def check_column_names(self, col_names: Iterable[str]) -> None:
        """
        Check that none of the column names we are working with are non-unique.
        Also check variable columns are all present (columns to copy and outcome allowed to be missing).

        :param col_names:
        :return: None, raises exception if there is a problem
        """

        col_names = [c for c in col_names]
        to_check = set(self.var_list_)
        if self.outcome_name_ is not None:
            to_check.add(self.outcome_name_)
        if self.cols_to_copy_ is not None:
            to_check.update(self.cols_to_copy_)
        seen = [c for c in col_names if c in to_check]
        if len(seen) != len(set(seen)):
            raise ValueError("duplicate column names in frame")
        missing = set(self.var_list_) - set(col_names)
        if len(missing) > 0:
            raise ValueError(f"missing required columns: {missing}")

    def clear(self) -> None:
        """reset state"""
        self.plan_ = None
        self.score_frame_ = None
        self.cross_rows_ = None
        self.cross_plan_ = None
        self.last_fit_x_id_ = None
        self.last_result_columns = None
        self.result_restriction = None

    def get_result_restriction(self):
        """accessor"""
        if self.result_restriction is None:
            return None
        return self.result_restriction.copy()

    def set_result_restriction(self, new_vars) -> None:
        """setter"""
        self.result_restriction = None
        if (new_vars is not None) and (len(new_vars) > 0):
            self.result_restriction = set(new_vars)

    @abc.abstractmethod
    def merge_params(self, p: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """merge in use parameters"""

    # sklearn pipeline step methods

    # https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X, y=None, **fit_params):
        """
        sklearn fit.

        :param X: explanatory variables
        :param y: (optional) dependent variable
        :param fit_params:
        :return: self (for method chaining)
        """

        self.fit_transform(X=X, y=y)
        return self

    # noinspection PyPep8Naming, PyUnusedLocal
    @abc.abstractmethod
    def fit_transform(self, X, y=None, **fit_params):
        """
        sklearn fit_transform, correct way to trigger cross methods.

        :param X: explanatory variables
        :param y: (optional) dependent variable
        :param fit_params:
        :return: transformed data
        """

    # noinspection PyPep8Naming
    @abc.abstractmethod
    def transform(self, X):
        """
        sklearn transform

        :param X: explanatory variables
        :return: transformed data
        """

    # https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_params(self, deep=True):
        """
        vtreat exposes a subset of controls as tunable parameters, users can choose this set
        by specifying the tunable_params list in object construction parameters

        :param deep: ignored
        :return: dict of tunable parameters
        """
        return {ti: self.params_[ti] for ti in self.params_["tunable_params"]}

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def set_params(self, **params):
        """
        vtreat exposes a subset of controls as tunable parameters, users can choose this set
        by specifying the tunable_params list in object construction parameters

        :param params:
        :return: self (for method chaining)
        """

        for (k, v) in params.items():
            if k in self.params_["tunable_params"]:
                self.params_[k] = v
        return self

    # extra methods to look more like sklearn objects

    # https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    # noinspection PyPep8Naming
    def fit_predict(self, X, y=None, **fit_params):
        """
        Alias for fit_transform()

        :param X: explanatory variables
        :param y: (optional) dependent variable
        :param fit_params:
        :return: transformed data
        """
        return self.fit_transform(X=X, y=y, **fit_params)

    # noinspection PyPep8Naming
    def predict(self, X):
        """
        Alias for transform.

        :param X: explanatory variables
        :return: transformed data
        """

        return self.transform(X)

    # noinspection PyPep8Naming
    def predict_proba(self, X):
        """
        Alias for transform.

        :param X: explanatory variables
        :return: transformed data
        """

        return self.transform(X)

    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/compose/_column_transformer.py

    def get_feature_names(self, input_features=None):
        """
        Get list of produced feature names.

        :param input_features: Optional, restrict to these features
        :return:
        """
        if self.score_frame_ is None:
            raise ValueError(
                "get_feature_names called on uninitialized vtreat transform"
            )
        if input_features is not None:
            input_features = [c for c in input_features]
        filter_to_recommended = False
        try:
            filter_to_recommended = self.params_["filter_to_recommended"]
        except KeyError:
            pass
        if filter_to_recommended:
            new_vars = [
                self.score_frame_["variable"][i]
                for i in range(self.score_frame_.shape[0])
                if self.score_frame_["has_range"][i]
                and self.score_frame_["recommended"][i]
                and (
                    input_features is None
                    or self.score_frame_["orig_variable"][i] in input_features
                )
            ]
        else:
            new_vars = [
                self.score_frame_["variable"][i]
                for i in range(self.score_frame_.shape[0])
                if self.score_frame_["has_range"][i]
                and (
                    input_features is None
                    or self.score_frame_["orig_variable"][i] in input_features
                )
            ]
        new_vars = new_vars + self.cols_to_copy_
        return new_vars

    def description_matrix(self) -> pandas.DataFrame:
        """
        Return description of transform as a data frame. Does not encode user steps. Not yet implemented for
        multinomial dependent variables.

        :return: description of transform.
        """

        assert self.plan_ is not None  # type hint
        assert self.score_frame_ is not None  # type hint
        xform_steps = [xfi for xfi in self.plan_.xforms]
        frames = [xfi.description_matrix() for xfi in xform_steps]
        res = pandas.concat(frames).reset_index(inplace=False, drop=True)
        # restrict down to non-constant variables
        scored_vars = set(self.score_frame_["variable"][self.score_frame_["has_range"]])
        usable = [v in scored_vars for v in res["variable"]]
        res = res.loc[usable, :].reset_index(inplace=False, drop=True)
        return res


def perform_transform(
    *, x: pandas.DataFrame, transform: VariableTreatment, params: Dict[str, Any]
) -> pandas.DataFrame:
    """
    Transform a data frame.

    :param x: data to be transformed.
    :param transform: transform
    :param params: control parameter dictionary
    :return: new data frame
    """
    assert transform.plan_ is not None  # type hint
    xform_steps = [xfi for xfi in transform.plan_.xforms]
    user_steps = [stp for stp in params["user_transforms"]]
    # restrict down to to results we are going to use
    if (transform.result_restriction is not None) and (
        len(transform.result_restriction) > 0
    ):
        xform_steps = [
            xfi
            for xfi in xform_steps
            if len(
                set(xfi.derived_column_names_).intersection(
                    transform.result_restriction
                )
            )
            > 0
        ]
        user_steps = [
            stp
            for stp in user_steps
            if len(set(stp.derived_vars_).intersection(transform.result_restriction))
            > 0
        ]
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
    copy_set = set(transform.plan_.cols_to_copy)
    to_copy = [ci for ci in x.columns if ci in copy_set]
    if len(to_copy) > 0:
        cp = x.loc[:, to_copy].copy()
        new_frames = [cp] + new_frames
    if len(new_frames) <= 0:
        raise ValueError("no columns transformed")
    res = pandas.concat(new_frames, axis=1, sort=False)
    res.reset_index(inplace=True, drop=True)
    return res


def limit_to_appropriate_columns(
    *, res: pandas.DataFrame, transform: VariableTreatment
) -> pandas.DataFrame:
    """
    Limit down to appropriate columns.

    :param res:
    :param transform:
    :return:
    """
    assert transform.plan_ is not None  # type hint
    assert transform.score_frame_ is not None  # type hint
    to_copy = set(transform.plan_.cols_to_copy)
    to_take = set(
        [
            ci
            for ci in transform.score_frame_["variable"][
                transform.score_frame_["has_range"]
            ]
        ]
    )
    if (transform.result_restriction is not None) and (
        len(transform.result_restriction) > 0
    ):
        to_take = to_take.intersection(transform.result_restriction)
    cols_to_keep = [ci for ci in res.columns if (ci in to_copy) or (ci in to_take)]
    if len(cols_to_keep) <= 0:
        raise ValueError("no columns retained")
    res = res[cols_to_keep].copy()
    res.reset_index(inplace=True, drop=True)
    return res
