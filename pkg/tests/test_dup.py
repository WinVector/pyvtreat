import pytest
import warnings

import pandas
import numpy
import numpy.random
import vtreat
import vtreat.util


def test_dup():
    numpy.random.seed(46546)

    def make_data(nrows):
        d = pandas.DataFrame({"x": [0.1 * i for i in range(nrows)]})
        d["y"] = d["x"] + numpy.sin(d["x"]) + 0.1 * numpy.random.normal(size=d.shape[0])
        d["xc"] = ["level_" + str(5 * numpy.round(yi / 5, 1)) for yi in d["y"]]
        d["x2"] = numpy.random.normal(size=d.shape[0])
        d.loc[d["xc"] == "level_-1.0", "xc"] = numpy.nan  # introduce a nan level
        d["yc"] = d["y"] > 0.5
        return d

    d = make_data(500)

    transform = vtreat.BinomialOutcomeTreatment(
        outcome_name="yc",  # outcome variable
        outcome_target=True,  # outcome of interest
        cols_to_copy=["y"],  # columns to "carry along" but not treat as input variables
    )

    d_prepared = transform.fit_transform(d, d["yc"])

    with pytest.warns(UserWarning):
        d_prepared_wrong = transform.transform(d)

    transform = vtreat.BinomialOutcomeTreatment(
        outcome_name="yc",  # outcome variable
        outcome_target=True,  # outcome of interest
        cols_to_copy=["y"],  # columns to "carry along" but not treat as input variables
        params={"error_on_duplicate_frames": True},
    )

    d_prepared = transform.fit_transform(d, d["yc"])

    with pytest.raises(ValueError):
        d_prepared_wrong = transform.transform(d)

    # no warning or error

    # https://docs.pytest.org/en/7.0.x/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dtest = make_data(450)

    dtest_prepared = transform.transform(dtest)
