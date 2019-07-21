

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:07:57 2019

@author: johnmount
"""

import numpy
import pandas
import statistics

def characterize_numeric_(x):
    """compute na count, min,max,mean of a numeric vector"""
    x = numpy.asarray(x).astype(float)
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
        col = numpy.asarray(data_frame[self.incoming_column_name_].copy()).astype(float)
        na_posns = numpy.isnan(col)
        col[na_posns] = self.replacement_value_
        res = pandas.DataFrame({self.dervied_column_names_[0]:col})
        return(res)



class indicate_missing(var_transform):
    def __init__(self, 
                 incoming_column_name,
                 dervied_column_name):
        var_transform.__init__(self, 
                               incoming_column_name, 
                               [dervied_column_name])
    
    def transform(self, data_frame):
        col = data_frame[self.incoming_column_name_].isnull()
        res = pandas.DataFrame({self.dervied_column_names_[0]:col})
        return(res.astype(float))


def can_convert_v_to_numeric_(x):
    """check if non-empty vector can convert to numeric"""
    try:
        x[0] + 0
        return(True)
    except:
        return(False)



class impact_code(var_transform):
    def __init__(self, 
                 incoming_column_name,
                 dervied_column_name,
                 code_book):
        var_transform.__init__(self, 
                               incoming_column_name, 
                               [dervied_column_name])
        self.code_book_ = code_book
    
    def transform(self, data_frame):
        res = data_frame[[self.incoming_column_name_]].join(
                self.code_book_,
                on = [ self.incoming_column_name_ ],
                how = 'left',
                sort = False) # ordered by left table rows
        res = res[[self.dervied_column_names_[0]]]
        res.loc[res[self.dervied_column_names_[0]].isnull(), 
                self.dervied_column_names_[0]] = 0
        return(res)


def fit_regression_impact_code(incoming_column_name, x, y):
    try:
        sf = pandas.DataFrame({"x":x,
                               "y":y})
        na_posns = sf["x"].isnull()
        sf.loc[na_posns, "x"] = "_NA_"
        sf["_group_mean"] = sf.groupby("x")["y"].transform("mean")
        sf["_gm"] = numpy.mean(sf[["y"]])[0]
        sf["_nest"] = sf["_group_mean"] - sf["_gm"] # naive condtional mean est
        # continue on to get heirarchical est with estimated variances
        # http://www.win-vector.com/blog/2017/09/partial-pooling-for-lower-variance-variable-encoding/
        means = sf.groupby("x")["y"].mean()
        sf["_vb"] = statistics.variance(means)
        sf["_one"] = 1
        sf["_ni"] = sf.groupby("x")["_one"].transform("sum")
        sf["_vw"] = numpy.mean((sf["y"] - sf["_group_mean"])**2) # a bit inflated
        sf["_hest"] = (sf["_ni"]*sf["_group_mean"]/sf["_vw"] + sf["_gm"]/sf["_vb"])/(sf["_ni"]/sf["_vw"] + 1/sf["_vb"]) - sf["_gm"]
        sf = sf.loc[:, ["x", "_hest"]].copy()
        newcol = incoming_column_name + "_impact_code"
        sf.columns = [ incoming_column_name, newcol ]
        sf = sf.groupby(incoming_column_name)[newcol].mean()
        return(impact_code(incoming_column_name, 
                           incoming_column_name + "_impact_code",
                           code_book = sf))
    except:
        return(None)


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
    varlist = [ co for co in varlist if (not (co in copy_set)) ]    
    xforms = []
    n = X.shape[0]
    all_null = []
    for vi in varlist:
        n_null = sum(X[vi].isnull())
        if n_null>=n:
            all_null = all_null + [vi]
        if (n_null>0) and (n_null<n):
            xforms = xforms + [ indicate_missing(incoming_column_name = vi, 
                                                dervied_column_name = vi + "_is_bad") ]
    varlist = [ co for co in varlist if (not (co in set(all_null))) ]
    numlist = [ co for co in varlist if can_convert_v_to_numeric_(X[co]) ]
    catlist = [ co for co in varlist if not co in set(numlist) ]
    for vi in numlist:
        summaryi = characterize_numeric_(X[vi])
        if summaryi["varies"] and summaryi["has_range"]:
            xforms = xforms + [ clean_numeric(incoming_column_name = vi, 
                                              replacement_value = summaryi["mean"]) ]
    for vi in catlist:
        xforms = xforms + [ fit_regression_impact_code(incoming_column_name = vi, 
                                                       x = numpy.asarray(X[vi]), 
                                                       y = y) ]
    xforms = [ xf for xf in xforms if xf is not None ]
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


    