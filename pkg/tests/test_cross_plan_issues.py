
import pandas as pd
import pytest

import vtreat.cross_plan


def test_cross_plan_issues_corner_cases():
    with pytest.warns(UserWarning):
        vtreat.cross_plan._k_way_cross_plan_y_stratified(n_rows=1, k_folds=0, y=None)
    with pytest.raises(ValueError):
        vtreat.cross_plan.KWayCrossPlanYStratified().split_plan()
    with pytest.raises(ValueError):
        vtreat.cross_plan.KWayCrossPlanYStratified().split_plan(n_rows=10)
    with pytest.raises(ValueError):
        vtreat.cross_plan.KWayCrossPlanYStratified().split_plan(k_folds=10)


def test_one_way_holdout():  # not recommended
    plan = vtreat.cross_plan.KWayCrossPlanYStratified().split_plan(n_rows=5, k_folds=5)
    assert len(plan) == 5


def test_one_way_holdout():
    plan = vtreat.cross_plan._k_way_cross_plan_y_stratified(n_rows=5, k_folds=3, y=None)
    assert len(plan) == 3
