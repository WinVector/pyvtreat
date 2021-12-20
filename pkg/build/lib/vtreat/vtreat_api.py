"""
Define user visible vtreat API.
"""

import warnings
from typing import Any, Dict, Iterable, Optional

import pandas
import numpy

import vtreat.vtreat_impl as vtreat_impl
import vtreat.util
import vtreat.cross_plan


def vtreat_parameters(user_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    build a vtreat parameters dictionary, adding in user choices

    :param user_params: dictionary of user choices
    :return: dictionary of user and default choices
    """

    params = {
        "use_hierarchical_estimate": True,
        "coders": {
            "clean_copy",
            "missing_indicator",
            "indicator_code",
            "impact_code",
            "deviation_code",
            "logit_code",
            "prevalence_code",
        },
        "filter_to_recommended": True,
        "indicator_min_fraction": 0.02,
        "cross_validation_plan": vtreat.cross_plan.KWayCrossPlanYStratified(),
        "cross_validation_k": 5,
        "user_transforms": [],
        "sparse_indicators": False,
        "missingness_imputation": numpy.mean,
        "check_for_duplicate_frames": True,
        "error_on_duplicate_frames": False,
        "retain_cross_plan": True,
        "tunable_params": ["indicator_min_fraction"],
    }
    pkeys = set(params.keys())
    if user_params is not None:
        for k in user_params.keys():
            if k not in pkeys:
                raise KeyError("parameter key " + str(k) + " not recognized")
            params[k] = user_params[k]
    if params["error_on_duplicate_frames"]:
        params["check_for_duplicate_frames"] = True
    for k in params["tunable_params"]:
        if k not in pkeys:
            raise KeyError("tunable_params key " + str(k) + " not recognized")
    return params


def unsupervised_parameters(
    user_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    build a vtreat parameters dictionary for unsupervised tasks, adding in user choices

    :param user_params: dictionary of user choices
    :return: dictionary of user and default choices
    """

    params = {
        "coders": {
            "clean_copy",
            "missing_indicator",
            "indicator_code",
            "prevalence_code",
        },
        "indicator_min_fraction": 0.0,
        "indicator_max_levels": 1000,
        "user_transforms": [],
        "sparse_indicators": False,
        "missingness_imputation": numpy.mean,
        "tunable_params": ["indicator_min_fraction", "indicator_max_levels"],
    }
    pkeys = set(params.keys())
    if user_params is not None:
        for k in user_params.keys():
            if k not in pkeys:
                raise KeyError("parameter key " + str(k) + " not recognized")
            params[k] = user_params[k]
    for k in params["tunable_params"]:
        if k not in pkeys:
            raise KeyError("tunable_params key " + str(k) + " not recognized")
    return params


class NumericOutcomeTreatment(vtreat_impl.VariableTreatment):
    """manage a treatment plan for a numeric outcome (regression)"""

    def __init__(
        self,
        *,
        var_list: Optional[Iterable[str]] = None,
        outcome_name: Optional[str] = None,
        cols_to_copy: Optional[Iterable[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        imputation_map: Optional[Dict[str, Any]] = None,
    ):
        """

         :param var_list: list or tuple of column names, if empty all non outcome and copy columns are used
         :param outcome_name: name of column containing dependent variable
         :param cols_to_copy: list or tuple of column names
         :param params: vtreat.vtreat_parameters()
         :param imputation_map: map of column names to custom missing imputation values or functions
        """
        params = self.merge_params(params)
        vtreat_impl.VariableTreatment.__init__(
            self,
            var_list=var_list,
            outcome_name=outcome_name,
            cols_to_copy=cols_to_copy,
            params=params,
            imputation_map=imputation_map,
        )

    def merge_params(self, p: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge user parameters, returns new parameters does not alter object.

        :param p:
        :return: merged parameters
        """

        return vtreat_parameters(p)

    # noinspection PyPep8Naming
    def transform(self, X):
        """
        Apply transform to data.

        :param X: data
        :return: transformed data
        """

        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if self.last_fit_x_id_ is None:
            raise ValueError("called transform on not yet fit treatment")
        if self.params_["check_for_duplicate_frames"] and (
            self.last_fit_x_id_ == vtreat.util.hash_data_frame(X)
        ):
            if self.params_["error_on_duplicate_frames"]:
                raise ValueError(
                    "possibly called transform on same data used to fit\n"
                    + "(this causes over-fit, please use fit_transform() instead)"
                )
            warnings.warn(
                "possibly called transform on same data used to fit\n"
                + "(this causes over-fit, please use fit_transform() instead)"
            )
        res = vtreat_impl.pre_prep_frame(
            X,
            col_list=self.plan_["num_list"] + self.plan_["cat_list"],
            cols_to_copy=self.cols_to_copy_,
            cat_cols=self.plan_["cat_list"],
        )
        res = vtreat_impl.perform_transform(x=res, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        res, res_columns = vtreat_impl.back_to_orig_type_data_frame(res, orig_type)
        self.last_result_columns = res_columns
        return res

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None, **fit_params):
        """
        fit_transform data, this is the way to fit with cross methods.

        :param X: explanatory values
        :param y: dependent values
        :param fit_params:
        :return: transformed data
        """

        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if y is None:
            if self.outcome_name_ is None:
                raise ValueError(".fit_transform(X) must have outcome_name set")
            y = numpy.asarray(X[self.outcome_name_])
        else:
            y = numpy.asarray(y)
            if (self.outcome_name_ is not None) and (self.outcome_name_ in X.columns):
                if not numpy.all(X[self.outcome_name_] == y):
                    raise ValueError(
                        ".fit_transform(X, y) called with y != X[outcome_name]"
                    )
        if not X.shape[0] == len(y):
            raise ValueError("X.shape[0] should equal len(y)")
        y = vtreat.util.safe_to_numeric_array(y)
        if vtreat.util.is_bad(y).sum() > 0:
            raise ValueError("y should not have any missing/NA/NaN values")
        if numpy.max(y) <= numpy.min(y):
            raise ValueError("y does not vary")
        cross_rows = None
        cross_plan = None
        if self.params_["retain_cross_plan"]:
            cross_rows = self.cross_rows_
            cross_plan = self.cross_plan_
        self.clear()
        self.last_fit_x_id_ = vtreat.util.hash_data_frame(X)
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        if isinstance(y, pandas.Series):
            y = y.reset_index(inplace=False, drop=True)
        # model for independent transforms
        self.plan_ = None
        self.score_frame_ = None
        self.plan_ = vtreat_impl.fit_numeric_outcome_treatment(
            X=X,
            y=y,
            var_list=self.var_list_,
            outcome_name=self.outcome_name_,
            cols_to_copy=self.cols_to_copy_,
            params=self.params_,
            imputation_map=self.imputation_map_,
        )
        cross_frame = vtreat_impl.perform_transform(
            x=X, transform=self, params=self.params_
        )
        if (cross_plan is None) or (cross_rows != X.shape[0]):
            if cross_plan is not None:
                warnings.warn(
                    "Number of rows different than previous fit with retain_cross_plan==True"
                )
            cross_plan = self.params_["cross_validation_plan"].split_plan(
                n_rows=X.shape[0],
                k_folds=self.params_["cross_validation_k"],
                data=X,
                y=y,
            )
            cross_rows = X.shape[0]
        # patch in cross-frame versions of complex columns such as impact
        vtreat_impl.cross_patch_refit_y_aware_cols(
            x=X, y=y, res=cross_frame, plan=self.plan_, cross_plan=cross_plan
        )
        vtreat_impl.cross_patch_user_y_aware_cols(
            x=cross_frame,
            y=y,
            res=cross_frame,
            params=self.params_,
            cross_plan=cross_plan,
        )
        # use cross_frame to compute variable effects
        self.score_frame_ = vtreat_impl.score_plan_variables(
            cross_frame=cross_frame,
            outcome=y,
            plan=self.plan_,
            params=self.params_,
            is_classification=False,
        )
        if ("filter_to_recommended" in self.params_.keys()) and self.params_[
            "filter_to_recommended"
        ]:
            self.set_result_restriction(
                set(
                    [
                        ci
                        for ci in self.score_frame_["variable"][
                            self.score_frame_["recommended"]
                        ]
                    ]
                )
            )
        cross_frame = vtreat_impl.limit_to_appropriate_columns(
            res=cross_frame, transform=self
        )
        cross_frame, res_columns = vtreat_impl.back_to_orig_type_data_frame(
            cross_frame, orig_type
        )
        self.last_result_columns = res_columns
        if self.params_["retain_cross_plan"]:
            self.cross_plan_ = cross_plan
            self.cross_rows_ = cross_rows
        else:
            self.cross_plan_ = None
            self.cross_rows_ = None
        return cross_frame


class BinomialOutcomeTreatment(vtreat_impl.VariableTreatment):
    """manage a treatment plan for a target outcome (binomial classification)"""

    def __init__(
        self,
        *,
        var_list: Optional[Iterable[str]] = None,
        outcome_name: Optional[str] = None,
        outcome_target=None,
        cols_to_copy: Optional[Iterable[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        imputation_map: Optional[Dict[str, Any]] = None,
    ):
        """

         :param var_list: list or tuple of column names, if empty all non outcome and copy columns are used
         :param outcome_name: name of column containing dependent variable
         :param outcome_target: value of outcome to consider "positive"
         :param cols_to_copy: list or tuple of column names
         :param params: vtreat.vtreat_parameters()
         :param imputation_map: map of column names to custom missing imputation values or functions
        """
        params = self.merge_params(params)
        vtreat_impl.VariableTreatment.__init__(
            self,
            var_list=var_list,
            outcome_name=outcome_name,
            outcome_target=outcome_target,
            cols_to_copy=cols_to_copy,
            params=params,
            imputation_map=imputation_map,
        )

    def merge_params(self, p: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge user parameters, returns new parameters does not alter object.

        :param p:
        :return: merged parameters
        """

        return vtreat_parameters(p)

    # noinspection PyPep8Naming
    def transform(self, X):
        """
        Apply transform to data.

        :param X: data
        :return: transformed data
        """

        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if self.last_fit_x_id_ is None:
            raise ValueError("called transform on not yet fit treatment")
        if self.params_["check_for_duplicate_frames"] and (
            self.last_fit_x_id_ == vtreat.util.hash_data_frame(X)
        ):
            if self.params_["error_on_duplicate_frames"]:
                raise ValueError(
                    "possibly called transform on same data used to fit\n"
                    + "(this causes over-fit, please use fit_transform() instead)"
                )
            warnings.warn(
                "possibly called transform on same data used to fit\n"
                + "(this causes over-fit, please use fit_transform() instead)"
            )
        X = vtreat_impl.pre_prep_frame(
            X,
            col_list=self.plan_["num_list"] + self.plan_["cat_list"],
            cols_to_copy=self.cols_to_copy_,
            cat_cols=self.plan_["cat_list"],
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        res, res_columns = vtreat_impl.back_to_orig_type_data_frame(res, orig_type)
        self.last_result_columns = res_columns
        return res

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None, **fit_params):
        """
        fit_transform data, this is the way to fit with cross methods.

        :param X: explanatory values
        :param y: dependent values
        :param fit_params:
        :return: transformed data
        """

        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if y is None:
            if self.outcome_name_ is None:
                raise ValueError(".fit_transform(X) must have outcome_name set")
            y = numpy.asarray(X[self.outcome_name_])
        else:
            y = numpy.asarray(y)
            if (self.outcome_name_ is not None) and (self.outcome_name_ in X.columns):
                if not numpy.all(X[self.outcome_name_] == y):
                    raise ValueError(
                        ".fit_transform(X, y) called with y != X[outcome_name]"
                    )
        if not X.shape[0] == len(y):
            raise ValueError("X.shape[0] should equal len(y)")
        y_mean = numpy.mean(y == self.outcome_target_)
        if y_mean <= 0 or y_mean >= 1:
            raise ValueError("y==outcome_target does not vary")
        cross_rows = None
        cross_plan = None
        if self.params_["retain_cross_plan"]:
            cross_rows = self.cross_rows_
            cross_plan = self.cross_plan_
        self.clear()
        self.last_fit_x_id_ = vtreat.util.hash_data_frame(X)
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        if isinstance(y, pandas.Series):
            y = y.reset_index(inplace=False, drop=True)
        # model for independent transforms
        self.plan_ = None
        self.score_frame_ = None
        self.plan_ = vtreat_impl.fit_binomial_outcome_treatment(
            X=X,
            y=y,
            outcome_target=self.outcome_target_,
            var_list=self.var_list_,
            outcome_name=self.outcome_name_,
            cols_to_copy=self.cols_to_copy_,
            params=self.params_,
            imputation_map=self.imputation_map_,
        )
        cross_frame = vtreat_impl.perform_transform(
            x=X, transform=self, params=self.params_
        )
        if (cross_plan is None) or (cross_rows != X.shape[0]):
            if cross_plan is not None:
                warnings.warn(
                    "Number of rows different than previous fit with retain_cross_plan==True"
                )
            cross_plan = self.params_["cross_validation_plan"].split_plan(
                n_rows=X.shape[0],
                k_folds=self.params_["cross_validation_k"],
                data=X,
                y=y,
            )
            cross_rows = X.shape[0]
        # patch in cross-frame versions of complex columns such as impact
        vtreat_impl.cross_patch_refit_y_aware_cols(
            x=X, y=y, res=cross_frame, plan=self.plan_, cross_plan=cross_plan
        )
        vtreat_impl.cross_patch_user_y_aware_cols(
            x=cross_frame,
            y=y,
            res=cross_frame,
            params=self.params_,
            cross_plan=cross_plan,
        )
        # use cross_frame to compute variable effects
        self.score_frame_ = vtreat_impl.score_plan_variables(
            cross_frame=cross_frame,
            outcome=numpy.asarray(
                numpy.asarray(y) == self.outcome_target_, dtype=float
            ),
            plan=self.plan_,
            params=self.params_,
            is_classification=True,
        )
        if ("filter_to_recommended" in self.params_.keys()) and self.params_[
            "filter_to_recommended"
        ]:
            self.set_result_restriction(
                set(
                    [
                        ci
                        for ci in self.score_frame_["variable"][
                            self.score_frame_["recommended"]
                        ]
                    ]
                )
            )
        cross_frame = vtreat_impl.limit_to_appropriate_columns(
            res=cross_frame, transform=self
        )
        cross_frame, res_columns = vtreat_impl.back_to_orig_type_data_frame(
            cross_frame, orig_type
        )
        self.last_result_columns = res_columns
        if self.params_["retain_cross_plan"]:
            self.cross_plan_ = cross_plan
            self.cross_rows_ = cross_rows
        else:
            self.cross_plan_ = None
            self.cross_rows_ = None
        return cross_frame


class MultinomialOutcomeTreatment(vtreat_impl.VariableTreatment):
    """manage a treatment plan for a set of outcomes (multinomial classification)"""

    def __init__(
        self,
        *,
        var_list: Optional[Iterable[str]] = None,
        outcome_name: Optional[str] = None,
        cols_to_copy: Optional[Iterable[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        imputation_map: Optional[Dict[str, Any]] = None,
    ):
        """

         :param var_list: list or tuple of column names, if empty all non outcome and copy columns are used
         :param outcome_name: name of column containing dependent variable
         :param cols_to_copy: list or tuple of column names
         :param params: vtreat.vtreat_parameters()
         :param imputation_map: map of column names to custom missing imputation values or functions
        """

        params = self.merge_params(params)
        vtreat_impl.VariableTreatment.__init__(
            self,
            var_list=var_list,
            outcome_name=outcome_name,
            cols_to_copy=cols_to_copy,
            params=params,
            imputation_map=imputation_map,
        )
        self.outcomes_ = None

    def merge_params(self, p: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge user parameters, returns new parameters does not alter object.

        :param p:
        :return: merged parameters
        """

        return vtreat_parameters(p)

    # noinspection PyPep8Naming
    def transform(self, X):
        """
        Apply transform to data.

        :param X: data
        :return: transformed data
        """

        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if self.last_fit_x_id_ is None:
            raise ValueError("called transform on not yet fit treatment")
        if self.params_["check_for_duplicate_frames"] and (
            self.last_fit_x_id_ == vtreat.util.hash_data_frame(X)
        ):
            if self.params_["error_on_duplicate_frames"]:
                raise ValueError(
                    "possibly called transform on same data used to fit\n"
                    + "(this causes over-fit, please use fit_transform() instead)"
                )
            warnings.warn(
                "possibly called transform on same data used to fit\n"
                + "(this causes over-fit, please use fit_transform() instead)"
            )
        X = vtreat_impl.pre_prep_frame(
            X,
            col_list=self.plan_["num_list"] + self.plan_["cat_list"],
            cols_to_copy=self.cols_to_copy_,
            cat_cols=self.plan_["cat_list"],
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        res, res_columns = vtreat_impl.back_to_orig_type_data_frame(res, orig_type)
        self.last_result_columns = res_columns
        return res

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None, **fit_params):
        """
        fit_transform data, this is the way to fit with cross methods.

        :param X: explanatory values
        :param y: dependent values
        :param fit_params:
        :return: transformed data
        """

        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if y is None:
            if self.outcome_name_ is None:
                raise ValueError(".fit_transform(X) must have outcome_name set")
            y = numpy.asarray(X[self.outcome_name_])
        else:
            y = numpy.asarray(y)
            if (self.outcome_name_ is not None) and (self.outcome_name_ in X.columns):
                if not numpy.all(X[self.outcome_name_] == y):
                    raise ValueError(
                        ".fit_transform(X, y) called with y != X[outcome_name]"
                    )
        if not X.shape[0] == len(y):
            raise ValueError("X.shape[0] should equal len(y)")
        if len(numpy.unique(y)) <= 1:
            raise ValueError("y must take on at least 2 values")
        cross_rows = None
        cross_plan = None
        if self.params_["retain_cross_plan"]:
            cross_rows = self.cross_rows_
            cross_plan = self.cross_plan_
        self.clear()
        self.last_fit_x_id_ = vtreat.util.hash_data_frame(X)
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        if isinstance(y, pandas.Series):
            y = y.reset_index(inplace=False, drop=True)
        # model for independent transforms
        self.plan_ = None
        self.score_frame_ = None
        self.outcomes_ = numpy.unique(y)
        self.plan_ = vtreat_impl.fit_multinomial_outcome_treatment(
            X=X,
            y=y,
            var_list=self.var_list_,
            outcome_name=self.outcome_name_,
            cols_to_copy=self.cols_to_copy_,
            params=self.params_,
            imputation_map=self.imputation_map_,
        )
        cross_frame = vtreat_impl.perform_transform(
            x=X, transform=self, params=self.params_
        )
        if (cross_plan is None) or (cross_rows != X.shape[0]):
            if cross_plan is not None:
                warnings.warn(
                    "Number of rows different than previous fit with retain_cross_plan==True"
                )
            cross_plan = self.params_["cross_validation_plan"].split_plan(
                n_rows=X.shape[0],
                k_folds=self.params_["cross_validation_k"],
                data=X,
                y=y,
            )
            cross_rows = X.shape[0]
        vtreat_impl.cross_patch_refit_y_aware_cols(
            x=X, y=y, res=cross_frame, plan=self.plan_, cross_plan=cross_plan
        )
        vtreat_impl.cross_patch_user_y_aware_cols(
            x=cross_frame,
            y=y,
            res=cross_frame,
            params=self.params_,
            cross_plan=cross_plan,
        )
        # use cross_frame to compute variable effects

        def si(oi):
            """score i-th outcome group"""
            sf = vtreat_impl.score_plan_variables(
                cross_frame=cross_frame,
                outcome=numpy.asarray(numpy.asarray(y) == oi, dtype=float),
                plan=self.plan_,
                params=self.params_,
                is_classification=True,
            )
            sf["outcome_target"] = oi
            return sf

        score_frames = [si(oi) for oi in self.outcomes_]
        self.score_frame_ = pandas.concat(score_frames, axis=0)
        self.score_frame_.reset_index(inplace=True, drop=True)
        if ("filter_to_recommended" in self.params_.keys()) and self.params_[
            "filter_to_recommended"
        ]:
            self.set_result_restriction(
                set(
                    [
                        ci
                        for ci in self.score_frame_["variable"][
                            self.score_frame_["recommended"]
                        ]
                    ]
                )
            )
        cross_frame = vtreat_impl.limit_to_appropriate_columns(
            res=cross_frame, transform=self
        )
        cross_frame, res_columns = vtreat_impl.back_to_orig_type_data_frame(
            cross_frame, orig_type
        )
        self.last_result_columns = res_columns
        if self.params_["retain_cross_plan"]:
            self.cross_plan_ = cross_plan
            self.cross_rows_ = cross_rows
        else:
            self.cross_plan_ = None
            self.cross_rows_ = None
        return cross_frame


class UnsupervisedTreatment(vtreat_impl.VariableTreatment):
    """manage an unsupervised treatment plan"""

    def __init__(
        self,
        *,
        var_list: Optional[Iterable[str]] = None,
        cols_to_copy: Optional[Iterable[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        imputation_map: Optional[Dict[str, Any]] = None,
    ):
        """

        :param var_list: list or tuple of column names, if empty all non copy columns are used
        :param cols_to_copy: list or tuple of column names
        :param params: vtreat.unsupervised_parameters()
        :param imputation_map: map of column names to custom missing imputation values or functions
        """
        params = self.merge_params(params)
        vtreat_impl.VariableTreatment.__init__(
            self,
            var_list=var_list,
            outcome_name=None,
            cols_to_copy=cols_to_copy,
            params=params,
            imputation_map=imputation_map,
        )

    def merge_params(self, p: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge user parameters, returns new parameters does not alter object.

        :param p:
        :return: merged parameters
        """

        return unsupervised_parameters(p)

    # noinspection PyPep8Naming
    def transform(self, X):
        """
        Apply transform to data.

        :param X: data
        :return: transformed data
        """

        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if self.last_fit_x_id_ is None:
            raise ValueError("called transform on not yet fit treatment")
        X = vtreat_impl.pre_prep_frame(
            X,
            col_list=self.plan_["num_list"] + self.plan_["cat_list"],
            cols_to_copy=self.cols_to_copy_,
            cat_cols=self.plan_["cat_list"],
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        res, res_columns = vtreat_impl.back_to_orig_type_data_frame(res, orig_type)
        self.last_result_columns = res_columns
        return res

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None, **fit_params):
        """
        fit_transform data.

        :param X: explanatory values
        :param y: dependent values
        :param fit_params:
        :return: transformed data
        """

        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if y is not None:
            raise ValueError("y should be None")
        self.clear()
        self.last_fit_x_id_ = vtreat.util.hash_data_frame(X)
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        self.plan_ = vtreat_impl.fit_unsupervised_treatment(
            X=X,
            var_list=self.var_list_,
            outcome_name=self.outcome_name_,
            cols_to_copy=self.cols_to_copy_,
            params=self.params_,
            imputation_map=self.imputation_map_,
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        self.score_frame_ = vtreat_impl.pseudo_score_plan_variables(
            cross_frame=res, plan=self.plan_, params=self.params_
        )
        if ("filter_to_recommended" in self.params_.keys()) and self.params_[
            "filter_to_recommended"
        ]:
            self.set_result_restriction(
                set(
                    [
                        ci
                        for ci in self.score_frame_["variable"][
                            self.score_frame_["recommended"]
                        ]
                    ]
                )
            )
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        res, res_columns = vtreat_impl.back_to_orig_type_data_frame(res, orig_type)
        self.last_result_columns = res_columns
        return res
