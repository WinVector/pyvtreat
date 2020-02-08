import warnings
import pprint

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


class VariableTreatment:
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
        self.cross_plan_ = None
        self.last_fit_x_id_ = None
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
        self.cross_plan_ = None
        self.last_fit_x_id_ = None

    def merge_params(self, p):
        return vtreat_parameters(p)

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

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        raise NotImplementedError("base class method called")

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None):
        raise NotImplementedError("base class method called")

    def get_feature_names(self, input_features=None):
        if self.score_frame_ is None:
            raise ValueError("get_feature_names called on uninitialized vtreat transform")
        new_vars = [self.score_frame_['variable'][i] for i in range(self.score_frame_.shape[0])
                    if self.score_frame_['has_range'][i]
                    and (input_features is None or self.score_frame_['variable'][i] in input_features)]
        new_vars = new_vars + self.cols_to_copy_
        return new_vars

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_params(self, deep=False):
        """
        vtreat doesn't expose parameters so outside code doesn't attempt to optimize over them
        """
        return {}

    def set_params(self, **params):
        """
        vtreat doesn't expose parameters so outside code doesn't attempt to optimize over them
        """
        pass

    # noinspection PyPep8Naming
    def inverse_transform(self, X):
        raise TypeError("vtreat does not support inverse_transform")

    # noinspection PyPep8Naming
    def transform(self, X):
        raise NotImplementedError("base class method called")


class NumericOutcomeTreatment(VariableTreatment):
    """manage a treatment plan for a numeric outcome (regression)"""

    def __init__(
            self, *,
            var_list=None,
            outcome_name,
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
        VariableTreatment.__init__(
            self,
            var_list=var_list,
            outcome_name=outcome_name,
            cols_to_copy=cols_to_copy,
            params=params,
            imputation_map=imputation_map,
        )

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("X should be a Pandas DataFrame")
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
        self.fit_transform(X=X, y=y)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("X should be a Pandas DataFrame")
        self.check_column_names(X.columns)
        if self.last_fit_x_id_ is None:
            raise ValueError("called transform on not yet fit treatment")
        if self.params_['check_for_duplicate_frames'] and (self.last_fit_x_id_ == id(X)):
            warnings.warn(
                "possibly called transform on same data used to fit (this causes over-fit, please use fit_transform() instead)")
        res = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        res = vtreat_impl.perform_transform(x=res, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        return res

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None):
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("X should be a Pandas DataFrame")
        self.check_column_names(X.columns)
        if y is None:
            if self.outcome_name_ is None:
                raise ValueError(".fit_transform(X) must have outcome_name set")
            y = X[self.outcome_name_]
        else:
            if self.outcome_name_ is not None:
                if not numpy.all(X[self.outcome_name_] == y):
                    raise ValueError(".fit_transform(X, y) called with y != X[outcome_name]")
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
        return cross_frame


class BinomialOutcomeTreatment(VariableTreatment):
    """manage a treatment plan for a target outcome (binomial classification)"""

    def __init__(
            self,
            *,
            var_list=None,
            outcome_name,
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
        VariableTreatment.__init__(
            self,
            var_list=var_list,
            outcome_name=outcome_name,
            outcome_target=outcome_target,
            cols_to_copy=cols_to_copy,
            params=params,
            imputation_map=imputation_map,
        )

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("X should be a Pandas DataFrame")
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
        self.fit_transform(X=X, y=y)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("X should be a Pandas DataFrame")
        self.check_column_names(X.columns)
        if self.last_fit_x_id_ is None:
            raise ValueError("called transform on not yet fit treatment")
        if self.params_['check_for_duplicate_frames'] and (self.last_fit_x_id_ == id(X)):
            warnings.warn(
                "possibly called transform on same data used to fit (this causes over-fit, please use fit_transform() instead)")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        return res

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None):
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("X should be a Pandas DataFrame")
        self.check_column_names(X.columns)
        if y is None:
            if self.outcome_name_ is None:
                raise ValueError(".fit_transform(X) must have outcome_name set")
            y = X[self.outcome_name_]
        else:
            if self.outcome_name_ is not None:
                if not numpy.all(X[self.outcome_name_] == y):
                    raise ValueError(".fit_transform(X, y) called with y != X[outcome_name]")
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
        return cross_frame


class MultinomialOutcomeTreatment(VariableTreatment):
    """manage a treatment plan for a set of outcomes (multinomial classification)"""

    def __init__(
            self,
            *,
            var_list=None,
            outcome_name,
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
        VariableTreatment.__init__(
            self,
            var_list=var_list,
            outcome_name=outcome_name,
            cols_to_copy=cols_to_copy,
            params=params,
            imputation_map=imputation_map,
        )
        self.outcomes_ = None

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("X should be a Pandas DataFrame")
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
        self.fit_transform(X=X, y=y)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("X should be a Pandas DataFrame")
        self.check_column_names(X.columns)
        if self.last_fit_x_id_ is None:
            raise ValueError("called transform on not yet fit treatment")
        if self.params_['check_for_duplicate_frames'] and (self.last_fit_x_id_ == id(X)):
            warnings.warn(
                "possibly called transform on same data used to fit (this causes over-fit, please use fit_transform() instead)")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        return res

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None):
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("X should be a Pandas DataFrame")
        self.check_column_names(X.columns)
        if y is None:
            if self.outcome_name_ is None:
                raise ValueError(".fit_transform(X) must have outcome_name set")
            y = X[self.outcome_name_]
        else:
            if self.outcome_name_ is not None:
                if not numpy.all(X[self.outcome_name_] == y):
                    raise ValueError(".fit_transform(X, y) called with y != X[outcome_name]")
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
        return cross_frame


class UnsupervisedTreatment(VariableTreatment):
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
        VariableTreatment.__init__(
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
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("X should be a Pandas DataFrame")
        self.check_column_names(X.columns)
        if y is not None:
            raise ValueError("y should be None")
        self.fit_transform(X)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("X should be a Pandas DataFrame")
        self.check_column_names(X.columns)
        if self.last_fit_x_id_ is None:
            raise ValueError("called transform on not yet fit treatment")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        return res

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None):
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("X should be a Pandas DataFrame")
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
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        self.score_frame_ = vtreat_impl.pseudo_score_plan_variables(
            cross_frame=res, plan=self.plan_, params=self.params_
        )
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        return res
