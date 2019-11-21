
import pytest
import pandas
import vtreat

def test_col_dups_1():
    d = pandas.DataFrame({'x': [1], 'x2': [2], 'y': [3]})
    d.columns = ['x', 'x', 'y']

    transform = vtreat.UnsupervisedTreatment(
        var_list=['x'],
        cols_to_copy=['y']
    )

    with pytest.raises(ValueError):
        transform.fit_transform(d, d["y"])
