

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:07:57 2019

@author: johnmount
"""

import numpy
import pandas

def characterize_numeric_(x):
    """compute na count, min,max,mean of a numeric vector"""
    not_nan = numpy.logical_not(numpy.isnan(x))
    n_not_nan = sum(not_nan)
    n = len(x)
    if n_not_nan<=0:
        return({"n":n,
                "n_not_nan":n_not_nan, 
                "min":None, 
                "mean":None, 
                "max":None,
                "varies":False,
                "has_range":False
                })
    x = x[not_nan]
    mn = numpy.min(x)
    mx = numpy.max(x)
    return({"n":n,
            "n_not_nan":n_not_nan, 
            "min":mn, 
            "mean":numpy.mean(x), 
            "max":mx,
            "varies":(mx>mn) or ((n_not_nan>0) and (n_not_nan<n)),
            "has_range":(mx>mn)
            })


class var_transform():
    """build a treatment plan for a numeric outcome (regression)"""
    def __init__(self, 
                 incoming_column_name,
                 dervied_column_names):
        self.incoming_column_name_ = incoming_column_name
        self.dervied_column_names_ = dervied_column_names.copy()
    
    def transform(self, data_frame):
        return(None)


class clean_numeric(var_transform):
    def __init__(self, 
                 incoming_column_name,
                 replacement_value):
        var_transform.__init__(self, 
                               incoming_column_name, 
                               [incoming_column_name])
        self.replacement_value_ = replacement_value
    
    def transform(self, data_frame):
        col = numpy.asarray(data_frame[self.incoming_column_name_].copy())
        na_posns = numpy.isnan(col)
        col[na_posns] = self.replacement_value_
        res = pandas.DataFrame({self.dervied_column_names_[0]:col})
        return(res)



class indicate_missing(var_transform):
    def __init__(self, 
                 incoming_column_name,
                 dervied_column_names):
        var_transform.__init__(self, 
                               incoming_column_name, 
                               [dervied_column_names])
    
    def transform(self, data_frame):
        col = numpy.isnan(data_frame[self.incoming_column_name_].copy())
        res = pandas.DataFrame({self.dervied_column_names_[0]:col})
        return(res.astype(float))


def fit_numeric_outcome_treatment_(
        *,
        X, y,
        sample_weight,
        varlist, 
        outcomename,
        cols_to_copy,
        plan):
    if (varlist is None) or (len(varlist)<=0):
        varlist = [ co for co in X.columns ]
    copy_set = set(cols_to_copy)
    varlist = [ co for co in varlist if ((not (co in copy_set)) and 
                                         (numpy.issubdtype(X[co].dtype, numpy.number))) ]
    xforms = []
    for vi in varlist:
        summaryi = characterize_numeric_(X[vi])
        if summaryi["varies"]:
            if summaryi["has_range"]:
                xforms = xforms + [ clean_numeric(incoming_column_name = vi, 
                                                  replacement_value = summaryi["mean"]) ]
            if summaryi["n_not_nan"]<summaryi["n"]:
                xforms = xforms + [ indicate_missing(incoming_column_name = vi, 
                                                     dervied_column_names = vi + "_is_bad") ]  
    return({
            "outcomename":outcomename,
            "cols_to_copy":cols_to_copy,
            "xforms":xforms
            })
    


def transform_numeric_outcome_treatment_(
        *,
        X,
        plan):
    X = X.reset_index(inplace=False, drop=True)
    new_frames = [ xfi.transform(X) for xfi in plan["xforms"] ]
    cp = X.loc[:, plan["cols_to_copy"] ].copy()
    return(pandas.concat([cp] + new_frames, axis=1))
    #return({"plan":plan, "X":X, "new_frames":new_frames})


    