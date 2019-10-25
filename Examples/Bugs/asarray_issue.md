
Issue from: https://github.com/WinVector/pyvtreat/issues/7#issuecomment-546502465


```python
import numpy
import pandas
```


```python
print(numpy.__version__)
```

    1.16.4



```python
print(pandas.__version__)
```

    0.25.0



```python
numpy.random.seed(2019)
arr = numpy.random.randint(2, size=10)
print(arr)
```

    [0 0 1 1 0 0 0 0 1 1]



```python

sparr = pandas.SparseArray(arr, fill_value=0)
print(sparr)
```

    [0, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    Fill: 0
    IntIndex
    Indices: array([2, 3, 8, 9], dtype=int32)
    



```python
np_arr = numpy.asarray(sparr)
print(np_arr)


```

    [0 0 1 1 0 0 0 0 1 1]

