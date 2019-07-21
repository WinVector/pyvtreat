

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:40:41 2019

@author: johnmount
"""




import numpy



def k_way_cross_plan(n_rows, 
                     k_folds, 
                     *, 
                     data_frame=None, 
                     y=None):
    """randomly split range(n_rows) into k_folds disjoint groups"""
    if n_rows<2:
        raise Exception('n_rows should be at least 2')
    if k_folds>=n_rows:
        k_folds = n_rows-1
    # first assign groups modulo k (ensuring at least one in each group)
    grp = [i % k_folds for i in range(n_rows)]
    # now shuffle
    numpy.random.shuffle(grp)
    plan = [ 
            { "train"  : [i for i in range(n_rows) if grp[i] != j],
               "app" : [i for i in range(n_rows) if grp[i] == j] } for j in range(k_folds) 
            ]
    return(plan)


