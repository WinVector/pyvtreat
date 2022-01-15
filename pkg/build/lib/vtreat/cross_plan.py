"""Basic cross validation methods"""

import abc
from typing import Dict, List, Optional
import warnings

import numpy
import numpy.random

import pandas


class CrossValidationPlan(abc.ABC):
    """Data splitting plan"""

    def __init__(self):
        pass

    @abc.abstractmethod
    def split_plan(
        self,
        *,
        n_rows: Optional[int] = None,
        k_folds: Optional[int] = None,
        data=None,
        y=None,
    ) -> List[Dict[str, List[int]]]:
        """
        Build a cross validation plan for the given parameters.

        :param n_rows: (optional) number of input rows
        :param k_folds: (optional) number of folds we want
        :param data: (optional) explanatory variables
        :param y: (optional) dependent variable
        :return: cross validation plan (list of dictionaries)
        """


def _k_way_cross_plan_y_stratified(
    *, n_rows: int, k_folds: int, y
) -> List[Dict[str, List[int]]]:
    """

    :param n_rows: number of input rows
    :param k_folds: number of cross folds desired
    :param y: values to stratify on
    :return: cross validation plan (list of dictionaries)
    """
    """randomly split range(n_rows) into k_folds disjoint groups, attempting an even y-distribution"""
    if k_folds < 2:
        k_folds = 2
    if k_folds > n_rows:
        k_folds = n_rows
    if n_rows <= 2 or k_folds <= 1:
        # degenerate overlap cases
        warnings.warn("degenerate cross plan, not disjoint", UserWarning)
        plan = [
            {"train": [i for i in range(n_rows)], "app": [i for i in range(n_rows)]}
        ]
        return plan
    # first sort by y plus a random key
    if y is None:
        y = numpy.zeros(n_rows)
    else:
        y = numpy.asarray(y)
    assert len(y) == n_rows
    d = pandas.DataFrame(
        {
            "y": y,
            "i": [i for i in range(n_rows)],
            "r": numpy.random.uniform(size=n_rows),
        }
    )
    d.sort_values(by=["y", "r"], inplace=True)
    d.reset_index(inplace=True, drop=True)
    # assign y-blocks to lose fine details of y
    fold_size = n_rows / k_folds
    d["block"] = [numpy.floor(i / fold_size) for i in range(n_rows)]
    d.sort_values(by=["block", "r"], inplace=True)
    d.reset_index(inplace=True, drop=True)
    # now assign groups modulo k (ensuring at least one in each group)
    d["grp"] = [i % k_folds for i in range(n_rows)]
    d.sort_values(by=["i"], inplace=True)
    d.reset_index(inplace=True, drop=True)
    grp = numpy.asarray(d["grp"])
    plan = [
        {
            "train": [i for i in range(n_rows) if grp[i] != j],
            "app": [i for i in range(n_rows) if grp[i] == j],
        }
        for j in range(k_folds)
    ]
    return plan


class KWayCrossPlanYStratified(CrossValidationPlan):
    """K-way cross validation plan, attempting an even y-distribution"""

    def __init__(self):
        CrossValidationPlan.__init__(self)

    def split_plan(
        self,
        *,
        n_rows: Optional[int] = None,
        k_folds: Optional[int] = None,
        data=None,
        y=None,
    ) -> List[Dict[str, List[int]]]:
        """

        :param n_rows: required, number of rows
        :param k_folds: required, number of cross-folds
        :param data: not used
        :param y: required, outcomes to stratify on
        :return:
        """
        if n_rows is None:
            raise ValueError("n_rows must not be None")
        if k_folds is None:
            raise ValueError("k_folds must not be None")
        if y is None:
            y = numpy.zeros(n_rows)
        assert len(y) == n_rows
        return _k_way_cross_plan_y_stratified(n_rows=n_rows, k_folds=k_folds, y=y)
