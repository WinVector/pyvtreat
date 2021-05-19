import numpy.random
import pandas
import numpy
import vtreat  # https://github.com/WinVector/pyvtreat
import vtreat.util


def test_nan_inf():
    numpy.random.seed(235)
    d = pandas.DataFrame(
        {"x": [1.0, numpy.nan, numpy.inf, -numpy.inf, None, 0], "y": [1, 2, 3, 4, 5, 6]}
    )

    transform = vtreat.NumericOutcomeTreatment(
        outcome_name="y",
        params=vtreat.vtreat_parameters({"filter_to_recommended": False}),
    )

    d_treated = transform.fit_transform(d, d["y"])

    for c in d_treated.columns:
        assert vtreat.util.can_convert_v_to_numeric(d_treated[c])
        assert numpy.sum(vtreat.util.is_bad(d_treated[c])) == 0

    expect = pandas.DataFrame(
        {
            "x": [1.0, 0.5, 0.5, 0.5, 0.5, 0],
            "x_is_bad": [0, 1, 1, 1, 1, 0],
            "y": [1, 2, 3, 4, 5, 6],
        }
    )

    for c in expect.columns:
        ec = numpy.asarray(expect[c])
        ed = numpy.asarray(d_treated[c])
        assert numpy.max(numpy.abs(ec - ed)) < 1.0e-6
