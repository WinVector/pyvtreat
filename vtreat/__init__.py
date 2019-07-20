


import numpy
import pandas



def k_way_cross_plan(n, k):
    """randomly split range(n) into k disjoint groups"""
    grp = [i % k for i in range(n)]
    numpy.random.shuffle(grp)
    plan = [ { "train"  : [i for i in range(n) if grp[i] != j],
               "test" : [i for i in range(n) if grp[i] == j] } for j in range(k) ]
    return(plan)


