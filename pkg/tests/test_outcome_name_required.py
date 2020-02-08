
import numpy

import pytest
import pandas

import vtreat  # https://github.com/WinVector/pyvtreat

def test_outcome_name_required():

    numpy.random.seed(235)
    d = pandas.DataFrame(
        {"x": ['1', '2', '3', '4', '5', '6'], "y": [1, 2, 3, 4, 5, 6]}
    )

    with pytest.raises(Exception):
        vtreat.NumericOutcomeTreatment()

    transform = vtreat.NumericOutcomeTreatment(outcome_name=None)
    transform.fit_transform(d, d["y"])
    with pytest.raises(Exception):
        transform.fit_transform(d)

    with pytest.raises(Exception):
        vtreat.BinomialOutcomeTreatment(outcome_target=3)

    transform = vtreat.BinomialOutcomeTreatment(outcome_name=None, outcome_target=3)
    transform.fit_transform(d, d["y"])
    with pytest.raises(Exception):
        transform.fit_transform(d)

    with pytest.raises(Exception):
        vtreat.vtreat_api.MultinomialOutcomeTreatment()

    transform = vtreat.vtreat_api.MultinomialOutcomeTreatment(outcome_name=None)
    transform.fit_transform(d, d["y"])
    with pytest.raises(Exception):
        transform.fit_transform(d)
