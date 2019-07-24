

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:40:41 2019

@author: johnmount
"""


import pandas
import numpy


import vtreat.vtreat_impl as vtreat_impl

# had been getting Future warnings on seemining correct (no missing values) use of
# Pandas indexing from vtreat.vtreat_impl.cross_patch_refit_y_aware_cols
# 
#/Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages/pandas/core/series.py:942: FutureWarning: 
#Passing list-likes to .loc or [] with any missing label will raise
#KeyError in the future, you can use .reindex() as an alternative.
#
#See the documentation here:
#https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike
#  return self.loc[key]
#/Users/johnmount/anaconda3/envs/aiAcademy/lib/python3.7/site-packages/pandas/core/indexing.py:1494: FutureWarning: 
#Passing list-likes to .loc or [] with any missing label will raise
#KeyError in the future, you can use .reindex() as an alternative.
#
#See the documentation here:
#https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike
#  return self._getitem_tuple(key)
#
# working around with:
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class numeric_outcome_treatment():
    """manage a treatment plan for a numeric outcome (regression)"""
    def __init__(self, 
                 *,
                 varlist=[],
                 outcomename,
                 cols_to_copy=[],
                 params = {}):
        if not outcomename in set(cols_to_copy):
            cols_to_copy = cols_to_copy + [outcomename]
        self.varlist_ = varlist.copy()
        self.outcomename_ = outcomename
        self.cols_to_copy_ = cols_to_copy.copy()
        self.params_ = params.copy()
        self.plan_ = None
        self.score_frame_ = None

    def fit(self, X, y=None, 
            *, 
            sample_weight=None):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is None:
            y = X[self.outcomename_]
        if not X.shape[0]==len(y):
            raise Exception("X.shape[0] should equal len(y)")
        if not sample_weight is None:
            raise Exception("doesn't accept sample_weight yest yet")
        self.plan_ = None
        self.score_frame_ = None
        self.fit_transform(
                X=X,
                y=y,
                sample_weight=sample_weight)
        return self

    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        return(vtreat_impl.perform_transform(
                X = X,
                plan = self.plan_))
    
    def fit_transform(self, X, y=None, 
                      *, 
                      sample_weight=None):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is None:
            y = X[self.outcomename_]
        if not X.shape[0]==len(y):
            raise Exception("X.shape[0] should equal len(y)")
        y = numpy.asarray(y, dtype=numpy.float64)
        if numpy.isnan(y).sum()>0:
            raise Exception("y should not have any missing/NA/NaN values")
        if not sample_weight is None:
            raise Exception("doesn't accept sample_weight yest yet")
        # model for independent transforms
        self.plan_ = None
        self.score_frame_ = None
        self.plan_ = vtreat_impl.fit_numeric_outcome_treatment(
                X = X, 
                y = y, 
                sample_weight = sample_weight,
                varlist = self.varlist_, 
                outcomename = self.outcomename_,
                cols_to_copy = self.cols_to_copy_,
                plan = self.plan_
                )
        res = self.transform(X)
        # patch in cross-frame versions of complex columns such as impact
        cross_frame = vtreat_impl.cross_patch_refit_y_aware_cols(
                X = X, y = y, sample_weight = sample_weight,
                res = res,
                plan = self.plan_
                )
        # use cross_frame to compute variable effects
        self.score_frame_ = vtreat_impl.score_plan_variables(
                cross_frame = cross_frame,
                outcome = y,
                plan = self.plan_)
        return(cross_frame)
  
    def get_params(self, deep=True):
        return(self.params_.copy())
    
    def set_params(self, **params):
        self.plan = None
        for a in params:
            self.params_[a] = params[a]


class binomial_outcome_treatment():
    """manage a treatment plan for a target outcome (binomial classification)"""
    def __init__(self, 
                 *,
                 varlist=[],
                 outcomename,
                 outcometarget,
                 cols_to_copy=[],
                 params = {}):
        if not outcomename in set(cols_to_copy):
            cols_to_copy = cols_to_copy + [outcomename]
        self.varlist_ = varlist.copy()
        self.outcomename_ = outcomename
        self.outcometarget_ = outcometarget
        self.cols_to_copy_ = cols_to_copy.copy()
        self.params_ = params.copy()
        self.plan_ = None
        self.score_frame_ = None

    def fit(self, X, y=None, 
            *, 
            sample_weight=None):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is None:
            y = X[self.outcomename_]
        if not X.shape[0]==len(y):
            raise Exception("X.shape[0] should equal len(y)")
        if not sample_weight is None:
            raise Exception("doesn't accept sample_weight yest yet")
        self.plan_ = None
        self.score_frame_ = None
        self.fit_transform(
                X=X,
                y=y,
                sample_weight=sample_weight)
        return self

    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        return(vtreat_impl.perform_transform(
                X = X,
                plan = self.plan_))
    
    def fit_transform(self, X, y=None, 
                      *, 
                      sample_weight=None):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is None:
            y = X[self.outcomename_]
        if not X.shape[0]==len(y):
            raise Exception("X.shape[0] should equal len(y)")
        if numpy.isnan(y).sum()>0:
            raise Exception("y should not have any missing/NA/NaN values")
        if not sample_weight is None:
            raise Exception("doesn't accept sample_weight yest yet")
        # model for independent transforms
        self.plan_ = None
        self.score_frame_ = None
        self.plan_ = vtreat_impl.fit_binomial_outcome_treatment(
                X = X, 
                y = y, 
                outcometarget = self.outcometarget_,
                sample_weight = sample_weight,
                varlist = self.varlist_, 
                outcomename = self.outcomename_,
                cols_to_copy = self.cols_to_copy_,
                plan = self.plan_
                )
        res = self.transform(X)
        # patch in cross-frame versions of complex columns such as impact
        cross_frame = vtreat_impl.cross_patch_refit_y_aware_cols(
                X = X, y = y, sample_weight = sample_weight,
                res = res,
                plan = self.plan_
                )
        # use cross_frame to compute variable effects
        self.score_frame_ = vtreat_impl.score_plan_variables(
                cross_frame = cross_frame,
                outcome = y,
                plan = self.plan_)
        return(cross_frame)
  
    def get_params(self, deep=True):
        return(self.params_.copy())
    
    def set_params(self, **params):
        self.plan = None
        for a in params:
            self.params_[a] = params[a]


class multinomial_outcome_treatment():
    """manage a treatment plan for a set of outcomes (multinomial classification)"""
    def __init__(self, 
                 *,
                 varlist=[],
                 outcomename,
                 cols_to_copy=[],
                 params = {}):
        if not outcomename in set(cols_to_copy):
            cols_to_copy = cols_to_copy + [outcomename]
        self.varlist_ = varlist.copy()
        self.outcomename_ = outcomename
        self.cols_to_copy_ = cols_to_copy.copy()
        self.params_ = params.copy()
        self.plan_ = None
        self.score_frame_ = None

    def fit(self, X, y=None, 
            *, 
            sample_weight=None):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is None:
            y = X[self.outcomename_]
        if not X.shape[0]==len(y):
            raise Exception("X.shape[0] should equal len(y)")
        if not sample_weight is None:
            raise Exception("doesn't accept sample_weight yest yet")
        self.plan_ = None
        self.score_frame_ = None
        self.fit_transform(
                X=X,
                y=y,
                sample_weight=sample_weight)
        return self

    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        return(vtreat_impl.perform_transform(
                X = X,
                plan = self.plan_))
    
    def fit_transform(self, X, y=None, 
                      *, 
                      sample_weight=None):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if y is None:
            y = X[self.outcomename_]
        if not X.shape[0]==len(y):
            raise Exception("X.shape[0] should equal len(y)")
        if numpy.isnan(y).sum()>0:
            raise Exception("y should not have any missing/NA/NaN values")
        if not sample_weight is None:
            raise Exception("doesn't accept sample_weight yest yet")
        # model for independent transforms
        self.plan_ = None
        self.score_frame_ = None
        self.plan_ = vtreat_impl.fit_multinomial_outcome_treatment(
                X = X, 
                y = y, 
                sample_weight = sample_weight,
                varlist = self.varlist_, 
                outcomename = self.outcomename_,
                cols_to_copy = self.cols_to_copy_,
                plan = self.plan_
                )
        res = self.transform(X)
        # patch in cross-frame versions of complex columns such as impact
        cross_frame = vtreat_impl.cross_patch_refit_y_aware_cols(
                X = X, y = y, sample_weight = sample_weight,
                res = res,
                plan = self.plan_
                )
        # use cross_frame to compute variable effects
        self.score_frame_ = vtreat_impl.score_plan_variables(
                cross_frame = cross_frame,
                outcome = y,
                plan = self.plan_)
        return(cross_frame)
  
    def get_params(self, deep=True):
        return(self.params_.copy())
    
    def set_params(self, **params):
        self.plan = None
        for a in params:
            self.params_[a] = params[a]


class unsupervised_treatment():
    """manage an unsupervised treatment plan"""
    def __init__(self, 
                 *,
                 varlist=[],
                 outcomename,
                 cols_to_copy=[],
                 params = {}):
        if not outcomename in set(cols_to_copy):
            cols_to_copy = cols_to_copy + [outcomename]
        self.varlist_ = varlist.copy()
        self.outcomename_ = outcomename
        self.cols_to_copy_ = cols_to_copy.copy()
        self.params_ = params.copy()
        self.plan_ = None

    def fit(self, X, y=None, 
            *, 
            sample_weight=None):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        if not y is None:
            raise Exception("y should be None")
        if not sample_weight is None:
            raise Exception("doesn't accept sample_weight yest yet")
        # model for independent transforms
        self.plan_ = None
        self.plan_ = vtreat_impl.fit_unsupervised_treatment(
                X = X, 
                sample_weight = sample_weight,
                varlist = self.varlist_, 
                outcomename = self.outcomename_,
                cols_to_copy = self.cols_to_copy_,
                plan = self.plan_
                )
        return self

    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        return(vtreat_impl.perform_transform(
                X = X,
                plan = self.plan_))
    
    def fit_transform(self, X, y=None, 
                      *, 
                      sample_weight=None):
        
        if not y is None:
            raise Exception("y should be None")
        self.fit(X = X, sample_weight = sample_weight)
        return(self.transform(X))
  
    def get_params(self, deep=True):
        return(self.params_.copy())
    
    def set_params(self, **params):
        self.plan = None
        for a in params:
            self.params_[a] = params[a]


