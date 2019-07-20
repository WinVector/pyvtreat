

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:40:41 2019

@author: johnmount
"""


import pandas





class numeric_outcome_treatment():
    """build a treatment plan for a numeric outcome (regression)"""
    def __init__(self, 
                 *,
                 varlist=None,
                 outcomename=None,
                 params = {}):
        self.varlist_ = None
        self.outcomename_ = None
        if not varlist is None:
            self.varlist_ = varlist.copy()
        if not outcomename is None:
            self.outcomename_ = outcomename
        self.params_ = params.copy()
        self.plan_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        self.plan_ = None
        return self

    def transform(self, X):
        if not isinstance(X, pandas.DataFrame):
            raise Exception("X should be a Pandas DataFrame")
        pass
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return(self.transform(X))
    
    def get_params(self, deep=True):
        return(self.params_.copy())
    
    def set_params(self, **params):
        self.plan = None
        for a in params:
            self.params_[a] = params[a]


