import warnings

import pandas
import numpy

import vtreat.vtreat_impl as vtreat_impl
import vtreat.util
import vtreat.cross_plan


def vtreat_parameters(user_params=None):
    """build a vtreat parameters dictionary, adding in user choices"""

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
        "indicator_min_fraction": 0.1,
        "cross_validation_plan": vtreat.cross_plan.KWayCrossPlanYStratified(),
        "cross_validation_k": 5,
        "user_transforms": [],
        "sparse_indicators": True,
        "missingness_imputation": numpy.mean,
        "check_for_duplicate_frames": True,
    }
    if user_params is not None:
        pkeys = set(params.keys())
        for k in user_params.keys():
            if k not in pkeys:
                raise KeyError("parameter key " + str(k) + " not recognized")
            params[k] = user_params[k]
    return params


def unsupervised_parameters(user_params=None):
    """build a vtreat parameters dictionary for unsupervised tasks, adding in user choices"""

    params = {
        "coders": {
            "clean_copy",
            "missing_indicator",
            "indicator_code",
            "prevalence_code",
        },
        "indicator_min_fraction": 0.0,
        "user_transforms": [],
        "sparse_indicators": True,
        "missingness_imputation": numpy.mean,
    }
    if user_params is not None:
        pkeys = set(params.keys())
        for k in user_params.keys():
            if k not in pkeys:
                raise KeyError("parameter key " + str(k) + " not recognized")
            params[k] = user_params[k]
    return params


class NumericOutcomeTreatment(vtreat_impl.VariableTreatment):
    """manage a treatment plan for a numeric outcome (regression)"""

    def __init__(
            self, *,
            var_list=None,
            outcome_name=None,
            cols_to_copy=None,
            params=None,
            imputation_map=None,
    ):
        """

         :param var_list: list or touple of column names
         :param outcome_name: name of column containing dependent variable
         :param cols_to_copy: list or touple of column names
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

    def merge_params(self, p):
        return vtreat_parameters(p)

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if y is None:
            if self.outcome_name_ is None:
                raise ValueError(".fit(X) must have outcome_name set")
            y = X[self.outcome_name_]
        else:
            if self.outcome_name_ is not None:
                if not numpy.all(X[self.outcome_name_] == y):
                    raise ValueError(".fit(X, y) called with y != X[outcome_name]")
        if not X.shape[0] == len(y):
            raise TypeError("X.shape[0] should equal len(y)")
        self._fit_transform_impl(X=X, y=y, do_transform=False)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if self.last_fit_x_id_ is None:
            raise ValueError("called transform on not yet fit treatment")
        if self.params_['check_for_duplicate_frames'] and (self.last_fit_x_id_ == id(X)):
            warnings.warn(
                "possibly called transform on same data used to fit\n" +
                "(this causes over-fit, please use _fit_transform_impl() instead)")
        res = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        res = vtreat_impl.perform_transform(x=res, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        res, res_columns = vtreat_impl.back_to_orig_type_data_frame(res, orig_type)
        self.last_result_columns = res_columns
        return res

    # noinspection PyPep8Naming
    def _fit_transform_impl(self, X, y=None, *, do_transform):
        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if y is None:
            if self.outcome_name_ is None:
                raise ValueError("._fit_transform_impl(X) must have outcome_name set")
            y = X[self.outcome_name_]
        else:
            if self.outcome_name_ is not None:
                if not numpy.all(X[self.outcome_name_] == y):
                    raise ValueError("._fit_transform_impl(X, y) called with y != X[outcome_name]")
        if not X.shape[0] == len(y):
            raise ValueError("X.shape[0] should equal len(y)")
        y = vtreat.util.safe_to_numeric_array(y)
        if vtreat.util.is_bad(y).sum() > 0:
            raise ValueError("y should not have any missing/NA/NaN values")
        if numpy.max(y) <= numpy.min(y):
            raise ValueError("y does not vary")
        self.clear()
        self.last_fit_x_id_ = id(X)
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
        if not do_transform:
            return None
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        # patch in cross-frame versions of complex columns such as impact
        self.cross_plan_ = self.params_["cross_validation_plan"].split_plan(
            n_rows=X.shape[0], k_folds=self.params_["cross_validation_k"], data=X, y=y
        )
        cross_frame = vtreat_impl.cross_patch_refit_y_aware_cols(
            x=X, y=y, res=res, plan=self.plan_, cross_plan=self.cross_plan_
        )
        cross_frame = vtreat_impl.cross_patch_user_y_aware_cols(
            x=cross_frame,
            y=y,
            res=res,
            params=self.params_,
            cross_plan=self.cross_plan_,
        )
        # use cross_frame to compute variable effects
        self.score_frame_ = vtreat_impl.score_plan_variables(
            cross_frame=cross_frame, outcome=y, plan=self.plan_, params=self.params_
        )
        cross_frame = vtreat_impl.limit_to_appropriate_columns(
            res=cross_frame, transform=self
        )
        cross_frame, res_columns = vtreat_impl.back_to_orig_type_data_frame(cross_frame, orig_type)
        self.last_result_columns = res_columns
        return cross_frame


class BinomialOutcomeTreatment(vtreat_impl.VariableTreatment):
    """manage a treatment plan for a target outcome (binomial classification)"""

    def __init__(
            self,
            *,
            var_list=None,
            outcome_name=None,
            outcome_target=True,
            cols_to_copy=None,
            params=None,
            imputation_map=None,
    ):
        """

         :param var_list: list or touple of column names
         :param outcome_name: name of column containing dependent variable
         :param outcome_target: value of outcome to consider "positive"
         :param cols_to_copy: list or touple of column names
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

    def merge_params(self, p):
        return vtreat_parameters(p)

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if y is None:
            if self.outcome_name_ is None:
                raise ValueError(".fit(X) must have outcome_name set")
            y = X[self.outcome_name_]
        else:
            if self.outcome_name_ is not None:
                if not numpy.all(X[self.outcome_name_] == y):
                    raise ValueError(".fit(X, y) called with y != X[outcome_name]")
        if not X.shape[0] == len(y):
            raise ValueError("X.shape[0] should equal len(y)")
        self._fit_transform_impl(X=X, y=y, do_transform=False)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if self.last_fit_x_id_ is None:
            raise ValueError("called transform on not yet fit treatment")
        if self.params_['check_for_duplicate_frames'] and (self.last_fit_x_id_ == id(X)):
            warnings.warn(
                "possibly called transform on same data used to fit\n" +
                "(this causes over-fit, please use _fit_transform_impl() instead)")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        res, res_columns = vtreat_impl.back_to_orig_type_data_frame(res, orig_type)
        self.last_result_columns = res_columns
        return res

    # noinspection PyPep8Naming
    def _fit_transform_impl(self, X, y=None, *, do_transform):
        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if y is None:
            if self.outcome_name_ is None:
                raise ValueError("._fit_transform_impl(X) must have outcome_name set")
            y = X[self.outcome_name_]
        else:
            if self.outcome_name_ is not None:
                if not numpy.all(X[self.outcome_name_] == y):
                    raise ValueError("._fit_transform_impl(X, y) called with y != X[outcome_name]")
        if not X.shape[0] == len(y):
            raise ValueError("X.shape[0] should equal len(y)")
        y_mean = numpy.mean(y == self.outcome_target_)
        if y_mean <= 0 or y_mean >= 1:
            raise ValueError("y==outcome_target does not vary")
        self.clear()
        self.last_fit_x_id_ = id(X)
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
        if not do_transform:
            return None
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        # patch in cross-frame versions of complex columns such as impact
        self.cross_plan_ = self.params_["cross_validation_plan"].split_plan(
            n_rows=X.shape[0], k_folds=self.params_["cross_validation_k"], data=X, y=y
        )
        cross_frame = vtreat_impl.cross_patch_refit_y_aware_cols(
            x=X, y=y, res=res, plan=self.plan_, cross_plan=self.cross_plan_
        )
        cross_frame = vtreat_impl.cross_patch_user_y_aware_cols(
            x=cross_frame,
            y=y,
            res=res,
            params=self.params_,
            cross_plan=self.cross_plan_,
        )
        # use cross_frame to compute variable effects
        self.score_frame_ = vtreat_impl.score_plan_variables(
            cross_frame=cross_frame,
            outcome=numpy.asarray(
                numpy.asarray(y) == self.outcome_target_, dtype=float
            ),
            plan=self.plan_,
            params=self.params_,
        )
        cross_frame = vtreat_impl.limit_to_appropriate_columns(
            res=cross_frame, transform=self
        )
        cross_frame, res_columns = vtreat_impl.back_to_orig_type_data_frame(cross_frame, orig_type)
        self.last_result_columns = res_columns
        return cross_frame


class MultinomialOutcomeTreatment(vtreat_impl.VariableTreatment):
    """manage a treatment plan for a set of outcomes (multinomial classification)"""

    def __init__(
            self,
            *,
            var_list=None,
            outcome_name=None,
            cols_to_copy=None,
            params=None,
            imputation_map=None,
    ):
        """

         :param var_list: list or touple of column names
         :param outcome_name: name of column containing dependent variable
         :param cols_to_copy: list or touple of column names
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

    def merge_params(self, p):
        return vtreat_parameters(p)

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if y is None:
            if self.outcome_name_ is None:
                raise ValueError(".fit(X) must have outcome_name set")
            y = X[self.outcome_name_]
        else:
            if self.outcome_name_ is not None:
                if not numpy.all(X[self.outcome_name_] == y):
                    raise ValueError(".fit(X, y) called with y != X[outcome_name]")
        if not X.shape[0] == len(y):
            raise ValueError("X.shape[0] should equal len(y)")
        self._fit_transform_impl(X=X, y=y, do_transform=False)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if self.last_fit_x_id_ is None:
            raise ValueError("called transform on not yet fit treatment")
        if self.params_['check_for_duplicate_frames'] and (self.last_fit_x_id_ == id(X)):
            warnings.warn(
                "possibly called transform on same data used to fit\n" +
                "(this causes over-fit, please use _fit_transform_impl() instead)")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        res, res_columns = vtreat_impl.back_to_orig_type_data_frame(res, orig_type)
        self.last_result_columns = res_columns
        return res

    # noinspection PyPep8Naming
    def _fit_transform_impl(self, X, y=None, *, do_transform):
        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if y is None:
            if self.outcome_name_ is None:
                raise ValueError("._fit_transform_impl(X) must have outcome_name set")
            y = X[self.outcome_name_]
        else:
            if self.outcome_name_ is not None:
                if not numpy.all(X[self.outcome_name_] == y):
                    raise ValueError("._fit_transform_impl(X, y) called with y != X[outcome_name]")
        if not X.shape[0] == len(y):
            raise ValueError("X.shape[0] should equal len(y)")
        if len(numpy.unique(y)) <= 1:
            raise ValueError("y must take on at least 2 values")
        self.clear()
        self.last_fit_x_id_ = id(X)
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
        if not do_transform:
            return None
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        # patch in cross-frame versions of complex columns such as impact
        self.cross_plan_ = self.params_["cross_validation_plan"].split_plan(
            n_rows=X.shape[0], k_folds=self.params_["cross_validation_k"], data=X, y=y
        )
        cross_frame = vtreat_impl.cross_patch_refit_y_aware_cols(
            x=X, y=y, res=res, plan=self.plan_, cross_plan=self.cross_plan_
        )
        cross_frame = vtreat_impl.cross_patch_user_y_aware_cols(
            x=cross_frame,
            y=y,
            res=res,
            params=self.params_,
            cross_plan=self.cross_plan_,
        )

        # use cross_frame to compute variable effects

        def si(oi):
            sf = vtreat_impl.score_plan_variables(
                cross_frame=cross_frame,
                outcome=numpy.asarray(numpy.asarray(y) == oi, dtype=float),
                plan=self.plan_,
                params=self.params_,
            )
            sf["outcome_target"] = oi
            return sf

        score_frames = [si(oi) for oi in self.outcomes_]
        self.score_frame_ = pandas.concat(score_frames, axis=0)
        self.score_frame_.reset_index(inplace=True, drop=True)
        cross_frame = vtreat_impl.limit_to_appropriate_columns(
            res=cross_frame, transform=self
        )
        cross_frame, res_columns = vtreat_impl.back_to_orig_type_data_frame(cross_frame, orig_type)
        self.last_result_columns = res_columns
        return cross_frame


class UnsupervisedTreatment(vtreat_impl.VariableTreatment):
    """manage an unsupervised treatment plan"""

    def __init__(self,
                 *,
                 var_list=None,
                 cols_to_copy=None,
                 params=None,
                 imputation_map=None):
        """

        :param var_list: list or touple of column names
        :param cols_to_copy: list or touple of column names
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

    def merge_params(self, p):
        return unsupervised_parameters(p)

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if y is not None:
            raise ValueError("y should be None")
        self._fit_transform_impl(X, do_transform=False)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if self.last_fit_x_id_ is None:
            raise ValueError("called transform on not yet fit treatment")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        res, res_columns = vtreat_impl.back_to_orig_type_data_frame(res, orig_type)
        self.last_result_columns = res_columns
        return res

    # noinspection PyPep8Naming
    def _fit_transform_impl(self, X, y=None, *, do_transform):
        X, orig_type = vtreat_impl.ready_data_frame(X)
        self.check_column_names(X.columns)
        if y is not None:
            raise ValueError("y should be None")
        self.clear()
        self.last_fit_x_id_ = id(X)
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
        if not do_transform:
            return None
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        self.score_frame_ = vtreat_impl.pseudo_score_plan_variables(
            cross_frame=res, plan=self.plan_, params=self.params_
        )
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        res, res_columns = vtreat_impl.back_to_orig_type_data_frame(res, orig_type)
        self.last_result_columns = res_columns
        return res
