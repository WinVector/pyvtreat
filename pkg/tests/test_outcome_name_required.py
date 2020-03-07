
import numpy

import pytest
import pandas

import vtreat  # https://github.com/WinVector/pyvtreat


def test_outcome_name_required():

    numpy.random.seed(235)
    d = pandas.DataFrame(
        {"x": ['1', '2', '3', '4', '5', '6'], "y": [1, 2, 3, 4, 5, 6]}
    )

    transform = vtreat.NumericOutcomeTreatment()
    transform.fit_transform(d, d["y"])
    with pytest.raises(Exception):
        transform.fit_transform(d)

    transform = vtreat.BinomialOutcomeTreatment(outcome_target=3)
    transform.fit_transform(d, d["y"])
    with pytest.raises(Exception):
        transform.fit_transform(d)

    transform = vtreat.vtreat_api.MultinomialOutcomeTreatment()
    transform.fit_transform(d, d["y"])
    with pytest.raises(Exception):
        transform.fit_transform(d)
