
import pytest
import pandas
import vtreat
import warnings

def test_id_col_check():
    d = pandas.DataFrame({'x': ['a', 'b', 'c'], 'y': ['a', 'b', 'b']})

    transform = vtreat.UnsupervisedTreatment(
        var_list=['x', 'y']
    )

    with pytest.warns(Warning):
        transform.fit_transform(d)