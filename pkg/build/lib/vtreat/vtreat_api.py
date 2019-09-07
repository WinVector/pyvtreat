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
        "cross_validation_plan": vtreat.cross_plan.KWayCrossPlan(),
        "cross_validation_k": 5,
        "user_transforms": [],
        "sparse_indicators": True,
    }
    if user_params is not None:
        pkeys = set(params.keys())
        for k in user_params.keys():
            if k not in pkeys:
                raise Exception("paramater key " + str(k) + " not recognized")
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
    }
    if user_params is not None:
        pkeys = set(params.keys())
        for k in user_params.keys():
            if k not in pkeys:
                raise Exception("paramater key " + str(k) + " not recognized")
            params[k] = user_params[k]
    return params


class VariableTreatment:
    def __init__(
        self, *, var_list=None, outcome_name=None, cols_to_copy=None, params=None
    ):
        if var_list is None:
            var_list = []
        if cols_to_copy is None:
            cols_to_copy = []
        if outcome_name is not None and outcome_name not in set(cols_to_copy):
            cols_to_copy = cols_to_copy + [outcome_name]
        self.outcome_name_ = outcome_name
        self.var_list_ = [vi for vi in var_list if vi not in set(cols_to_copy)]
        self.cols_to_copy_ = cols_to_copy.copy()
        self.params_ = params.copy()
        self.plan_ = None
        self.score_frame_ = None
        self.cross_plan_ = None
        self.n_training_rows_ = None


class NumericOutcomeTreatment(VariableTreatment):
    """manage a treatment plan for a numeric outcome (regression)"""

    def __init__(self, *, var_list=None, outcome_name=None, cols_to_copy=None, params=None):
        """

         :param var_list: list or touple of column names
         :param outcome_name: name of column containing dependent variable
         :param cols_to_copy: list or touple of column names
         :param params: vtreat.vtreat_parameters()
        """
        params = vtreat_parameters(params)
        VariableTreatment.__init__(
            self,
            var_list=var_list,
            outcome_name=outcome_name,
            cols_to_copy=cols_to_copy,
            params=params,
        )

    # noinspection PyPep8Naming
    def fit(self, X, y):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is None:
            y = X[self.outcome_name_]
        if not X.shape[0] == len(y):
            raise Exception("X.shape[0] should equal len(y)")
        self.fit_transform(X=X, y=y)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        res = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        res = vtreat_impl.perform_transform(x=res, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        return res

    # noinspection PyPep8Naming
    def fit_transform(self, X, y):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is None:
            y = X[self.outcome_name_]
        if not X.shape[0] == len(y):
            raise Exception("X.shape[0] should equal len(y)")
        y = numpy.asarray(y, dtype=numpy.float64)
        if vtreat.util.is_bad(y).sum() > 0:
            raise Exception("y should not have any missing/NA/NaN values")
        if numpy.max(y) <= numpy.min(y):
            raise Exception("y does not vary")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        if isinstance(y, pandas.Series):
            y = y.reset_index(inplace=False, drop=True)
        # model for independent transforms
        self.plan_ = None
        self.score_frame_ = None
        self.n_training_rows_ = X.shape[0]
        self.plan_ = vtreat_impl.fit_numeric_outcome_treatment(
            X=X,
            y=y,
            var_list=self.var_list_,
            outcome_name=self.outcome_name_,
            cols_to_copy=self.cols_to_copy_,
            params=self.params_,
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
        outcome_name=None,
        outcome_target,
        cols_to_copy=None,
        params=None
    ):
        """

         :param var_list: list or touple of column names
         :param outcome_name: name of column containing dependent variable
         :param outcome_target: value of outcome to consider "positive"
         :param cols_to_copy: list or touple of column names
         :param params: vtreat.vtreat_parameters()
        """
        params = vtreat_parameters(params)
        VariableTreatment.__init__(
            self,
            var_list=var_list,
            outcome_name=outcome_name,
            cols_to_copy=cols_to_copy,
            params=params,
        )
        self.outcome_target_ = outcome_target

    # noinspection PyPep8Naming
    def fit(self, X, y):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is None:
            y = X[self.outcome_name_]
        if not X.shape[0] == len(y):
            raise Exception("X.shape[0] should equal len(y)")
        self.fit_transform(X=X, y=y)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        return res

    # noinspection PyPep8Naming
    def fit_transform(self, X, y):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is None:
            y = X[self.outcome_name_]
        if not X.shape[0] == len(y):
            raise Exception("X.shape[0] should equal len(y)")
        y_mean = numpy.mean(y == self.outcome_target_)
        if y_mean <= 0 or y_mean >= 1:
            raise Exception("y==outcome_target does not vary")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        if isinstance(y, pandas.Series):
            y = y.reset_index(inplace=False, drop=True)
        # model for independent transforms
        self.plan_ = None
        self.score_frame_ = None
        self.n_training_rows_ = X.shape[0]
        self.plan_ = vtreat_impl.fit_binomial_outcome_treatment(
            X=X,
            y=y,
            outcome_target=self.outcome_target_,
            var_list=self.var_list_,
            outcome_name=self.outcome_name_,
            cols_to_copy=self.cols_to_copy_,
            params=self.params_,
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
                numpy.asarray(y) == self.outcome_target_, dtype=numpy.float64
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

    def __init__(self, *, var_list=None, outcome_name=None, cols_to_copy=None, params=None):
        """

         :param var_list: list or touple of column names
         :param outcome_name: name of column containing dependent variable
         :param cols_to_copy: list or touple of column names
         :param params: vtreat.vtreat_parameters()
        """

        params = vtreat_parameters(params)
        VariableTreatment.__init__(
            self,
            var_list=var_list,
            outcome_name=outcome_name,
            cols_to_copy=cols_to_copy,
            params=params,
        )
        self.outcomes_ = None

    # noinspection PyPep8Naming
    def fit(self, X, y):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is None:
            y = X[self.outcome_name_]
        if not X.shape[0] == len(y):
            raise Exception("X.shape[0] should equal len(y)")
        self.fit_transform(X=X, y=y)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        return res

    # noinspection PyPep8Naming
    def fit_transform(self, X, y):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is None:
            y = X[self.outcome_name_]
        if not X.shape[0] == len(y):
            raise Exception("X.shape[0] should equal len(y)")
        if len(numpy.unique(y)) <= 1:
            raise Exception("y must take on at least 2 values")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        if isinstance(y, pandas.Series):
            y = y.reset_index(inplace=False, drop=True)
        # model for independent transforms
        self.plan_ = None
        self.score_frame_ = None
        self.n_training_rows_ = X.shape[0]
        self.outcomes_ = numpy.unique(y)
        self.plan_ = vtreat_impl.fit_multinomial_outcome_treatment(
            X=X,
            y=y,
            var_list=self.var_list_,
            outcome_name=self.outcome_name_,
            cols_to_copy=self.cols_to_copy_,
            params=self.params_,
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
                outcome=numpy.asarray(numpy.asarray(y) == oi, dtype=numpy.float64),
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

    def __init__(self, *, var_list=None, cols_to_copy=None, params=None):
        """

        :param var_list: list or touple of column names
        :param cols_to_copy: list or touple of column names
        :param params: vtreat.unsupervised_parameters()
        """
        params = unsupervised_parameters(params)
        VariableTreatment.__init__(
            self,
            var_list=var_list,
            outcome_name=None,
            cols_to_copy=cols_to_copy,
            params=params,
        )

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is not None:
            raise Exception("y should be None")
        self.fit_transform(X)
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        return res

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None):
        if y is not None:
            raise Exception("y should be None")
        X = vtreat_impl.pre_prep_frame(
            X, col_list=self.var_list_, cols_to_copy=self.cols_to_copy_
        )
        self.n_training_rows_ = X.shape[0]
        self.plan_ = vtreat_impl.fit_unsupervised_treatment(
            X=X,
            var_list=self.var_list_,
            outcome_name=self.outcome_name_,
            cols_to_copy=self.cols_to_copy_,
            params=self.params_,
        )
        res = vtreat_impl.perform_transform(x=X, transform=self, params=self.params_)
        self.score_frame_ = vtreat_impl.pseudo_score_plan_variables(
            cross_frame=res, plan=self.plan_, params=self.params_
        )
        res = vtreat_impl.limit_to_appropriate_columns(res=res, transform=self)
        return res
