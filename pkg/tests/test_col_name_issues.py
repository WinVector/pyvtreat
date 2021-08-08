import pytest
import pandas
import vtreat


def test_col_dups_1():
    d = pandas.DataFrame({"x": [1], "x2": [2], "y": [3]})
    d.columns = ["x", "x", "y"]

    transform = vtreat.UnsupervisedTreatment(var_list=["x"], cols_to_copy=["y"])

    with pytest.raises(ValueError):
        transform.fit_transform(d, d["y"])


def test_xgboost_col_name_issue_1():
    # https://stackoverflow.com/questions/48645846/pythons-xgoost-valueerrorfeature-names-may-not-contain-or
    # ValueError('feature_names may not contain [, ] or <')
    d = pandas.DataFrame({"x": ["[", "]", "<", ">", ">"]})

    transform = vtreat.UnsupervisedTreatment(var_list=["x"])
    d_transformed = transform.fit_transform(d, None)
    cols = d_transformed.columns
    for col in cols:
        assert not any(c in col for c in "[]<>")
    assert len(set(cols)) == len(cols)


def test_xgboost_col_name_issue_2():
    # https://stackoverflow.com/questions/48645846/pythons-xgoost-valueerrorfeature-names-may-not-contain-or
    # ValueError('feature_names may not contain [, ] or <')
    d = pandas.DataFrame({"x": ["[", "]", "<", "_lt_", "_lt_"]})

    transform = vtreat.UnsupervisedTreatment(var_list=["x"])
    d_transformed = transform.fit_transform(d, None)
    cols = d_transformed.columns
    for col in cols:
        assert not any(c in col for c in "[]<>")
    assert len(set(cols)) == len(cols)
