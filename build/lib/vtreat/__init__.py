

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:40:41 2019

@author: johnmount
"""


import pandas

import vtreat.vtreat_impl as vtreat_impl


class numeric_outcome_treatment():
    """build a treatment plan for a numeric outcome (regression)"""
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
        self.plan_ = vtreat_impl.fit_numeric_outcome_treatment_(
                X = X, y = y, sample_weight = sample_weight,
                varlist = self.varlist_, 
                outcomename = self.outcomename_,
                cols_to_copy = self.cols_to_copy_,
                plan = self.plan_
                )
        return self

    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        return(vtreat_impl.transform_numeric_outcome_treatment_(
                X = X,
                plan = self.plan_))
    
    def fit_transform(self, X, y=None, 
                      *, 
                      sample_weight=None):
        self.fit(X, y, sample_weight=sample_weight)
        return(self.transform(X))
    
    def get_params(self, deep=True):
        return(self.params_.copy())
    
    def set_params(self, **params):
        self.plan = None
        for a in params:
            self.params_[a] = params[a]


