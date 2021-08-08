import vtreat.util
import pandas
import numpy

import pytest


def test_can_convert_v_to_numeric():
    assert vtreat.util.can_convert_v_to_numeric(0) == True
    assert vtreat.util.can_convert_v_to_numeric(1.0) == True
    assert vtreat.util.can_convert_v_to_numeric("hi") == False
    assert vtreat.util.can_convert_v_to_numeric(numpy.asarray([1, 2])) == True
    assert (
        vtreat.util.can_convert_v_to_numeric(pandas.DataFrame({"x": [1, 2]})["x"])
        == True
    )
    assert (
        vtreat.util.can_convert_v_to_numeric(
            pandas.DataFrame({"x": [1, numpy.nan]})["x"]
        )
        == True
    )
    assert (
        vtreat.util.can_convert_v_to_numeric(pandas.DataFrame({"x": ["a", "b"]})["x"])
        == False
    )
    assert (
        vtreat.util.can_convert_v_to_numeric(
            pandas.DataFrame({"x": ["a", numpy.nan]})["x"]
        )
        == False
    )


def test_convert_raises():
    # check some corner cases involving -0, '1', and so on
    res1 = vtreat.util.safe_to_numeric_array([1, None])
    expect1 = numpy.asarray([1.0, numpy.NaN])
    assert numpy.array_equal(res1, expect1, equal_nan=True)

    with pytest.raises(ValueError):
        vtreat.util.safe_to_numeric_array([1.0, "b"])

    assert vtreat.util.can_convert_v_to_numeric([1, 2])
    assert vtreat.util.can_convert_v_to_numeric([1, None])
    assert vtreat.util.can_convert_v_to_numeric([1.0])
    assert not vtreat.util.can_convert_v_to_numeric([1.0, "b"])
    assert not vtreat.util.can_convert_v_to_numeric(["1", 2])
    assert vtreat.util.can_convert_v_to_numeric([])
    assert vtreat.util.can_convert_v_to_numeric([None])

    zeros = [0, 0.0, 1 / numpy.inf, 1 / (-numpy.inf)]
    assert vtreat.util.can_convert_v_to_numeric(zeros)
    res2 = vtreat.util.safe_to_numeric_array(zeros)
    expect2 = numpy.asarray([0.0, 0.0, 0.0, 1 / (-numpy.inf)])
    assert numpy.array_equal(res2, expect2, equal_nan=True)
    assert numpy.array_equal(res2.astype(str), expect2.astype(str))
    wrong2 = numpy.asarray([0.0, 0.0, 0.0, 0.0])
    assert not numpy.array_equal(res2.astype(str), wrong2.astype(str))
