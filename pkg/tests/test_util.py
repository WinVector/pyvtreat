
import vtreat.util
import pandas
import numpy


def test_can_convert_v_to_numeric():
    assert vtreat.util.can_convert_v_to_numeric(0) == True
    assert vtreat.util.can_convert_v_to_numeric(1.0) == True
    assert vtreat.util.can_convert_v_to_numeric("hi") == False
    assert vtreat.util.can_convert_v_to_numeric(numpy.asarray([1, 2])) == True
    assert vtreat.util.can_convert_v_to_numeric(pandas.DataFrame({'x': [1, 2]})['x']) == True
    assert vtreat.util.can_convert_v_to_numeric(pandas.DataFrame({'x': [1, numpy.nan]})['x']) == True
    assert vtreat.util.can_convert_v_to_numeric(pandas.DataFrame({'x': ['a', 'b']})['x']) == False
    assert vtreat.util.can_convert_v_to_numeric(pandas.DataFrame({'x': ['a', numpy.nan]})['x']) == False

