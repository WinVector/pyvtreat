import numpy.random
import pandas
import numpy
import vtreat  # https://github.com/WinVector/pyvtreat
import vtreat.util


def test_unexpected_nan():
    # confirm NaN processing correct, even when none seenin training data
    numpy.random.seed(235)
    d = pandas.DataFrame(
        {"x": [1, 2, 3, 4, 5, 6], "y": [1, 2, 3, 4, 5, 6]}
    )

    transform = vtreat.NumericOutcomeTreatment(
        outcome_name="y",
        params=vtreat.vtreat_parameters({"filter_to_recommended": False}),
    )

    d_treated = transform.fit_transform(d, d["y"])
    assert transform.score_frame_.shape[0] == 1
    assert 'x' in set(transform.score_frame_['variable'])

    d_app = pandas.DataFrame(
        {"x": [1, 2, numpy.NAN, 4, None, 6]}
    )
    assert numpy.any(numpy.isnan(d_app['x']))
    d_app_treated = transform.transform(d_app)
    assert not numpy.any(numpy.isnan(d_app_treated['x']))
