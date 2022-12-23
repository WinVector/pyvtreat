import numpy.random
import pandas
import vtreat  # https://github.com/WinVector/pyvtreat

import pytest
import warnings


def test_unsupervised():
    numpy.random.seed(235)
    zip = ["z" + str(i + 1).zfill(5) for i in range(15)]
    d = pandas.DataFrame({"zip": numpy.random.choice(zip, size=1000)})
    d["const"] = 1
    d["const2"] = "b"
    d["const3"] = None

    transform = vtreat.UnsupervisedTreatment(
        params=vtreat.unsupervised_parameters({"indicator_min_fraction": 0.01})
    )

    d_treated = transform.fit_transform(d)

    for c in d_treated.columns:
        assert vtreat.util.can_convert_v_to_numeric(d_treated[c])
        assert numpy.sum(vtreat.util.is_bad(d_treated[c])) == 0

    sf = transform.score_frame_
    assert set(sf["orig_variable"]) == {"zip"}

    # https://stackoverflow.com/a/45671804/6901725
    # https://docs.pytest.org/en/7.0.x/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d_treated_2 = transform.transform(d)
    assert d_treated.equals(d_treated_2)
    fn = transform.get_feature_names()
    assert set(sf["variable"]) == set(fn)
