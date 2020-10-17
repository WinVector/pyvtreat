import vtreat.util
import pandas
import numpy


def test_range():
    # https://github.com/WinVector/pyvtreat/blob/master/Examples/Bugs/asarray_issue.md
    # https://github.com/WinVector/pyvtreat/issues/7
    numpy.random.seed(2019)
    arr = numpy.random.randint(2, size=10)
    sparr = pandas.arrays.SparseArray(arr, fill_value=0)
    assert vtreat.util.numeric_has_range(arr)
    assert vtreat.util.numeric_has_range(sparr)
