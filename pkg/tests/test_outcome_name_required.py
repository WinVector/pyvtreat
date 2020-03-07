
import numpy

import pytest
import pandas

import vtreat  # https://github.com/WinVector/pyvtreat


def test_outcome_name_required():

    numpy.random.seed(235)
    d = pandas.DataFrame(
        {"x": ['1', '1', '1', '2', '2', '2']})
    y = [1, 2, 3, 4, 5, 6]

    transform = vtreat.NumericOutcomeTreatment(
        params=vtreat.vtreat_parameters({'filter_to_recommended': False})
    )
    transform.fit_transform(d, y)
    with pytest.raises(Exception):
        transform.fit_transform(d)

    transform = vtreat.BinomialOutcomeTreatment(
        params=vtreat.vtreat_parameters({'filter_to_recommended': False}),
        outcome_target=3)
    transform.fit_transform(d, y)
    with pytest.raises(Exception):
        transform.fit_transform(d)

    transform = vtreat.vtreat_api.MultinomialOutcomeTreatment(
        params=vtreat.vtreat_parameters({'filter_to_recommended': False})
    )
    transform.fit_transform(d, y)
    with pytest.raises(Exception):
        transform.fit_transform(d)
