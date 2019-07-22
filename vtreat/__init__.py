

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:40:41 2019

@author: johnmount
"""


import pandas
import numpy


import vtreat.vtreat_impl as vtreat_impl


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
        return(vtreat_impl.transform_numeric_outcome_treatment(
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
        cross_frame = vtreat_impl.fit_numeric_outcome_treatment_cross_patch(
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
    """manage a treatment plan for a numeric outcome (classification)"""
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
        return(vtreat_impl.transform_numeric_outcome_treatment(
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
        y = numpy.asarray(y==self.outcometarget_, dtype=numpy.float64)
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
                sample_weight = sample_weight,
                varlist = self.varlist_, 
                outcomename = self.outcomename_,
                cols_to_copy = self.cols_to_copy_,
                plan = self.plan_
                )
        print("before transform")
        res = self.transform(X)
        # patch in cross-frame versions of complex columns such as impact
        print("before patch")
        cross_frame = vtreat_impl.fit_numeric_outcome_treatment_cross_patch(
                X = X, y = y, sample_weight = sample_weight,
                res = res,
                plan = self.plan_
                )
        print("before sf")
        # use cross_frame to compute variable effects
        self.score_frame_ = vtreat_impl.score_plan_variables(
                cross_frame = cross_frame,
                outcome = y,
                plan = self.plan_)
        print("before return")
        return(cross_frame)
  
    def get_params(self, deep=True):
        return(self.params_.copy())
    
    def set_params(self, **params):
        self.plan = None
        for a in params:
            self.params_[a] = params[a]


