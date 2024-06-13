
import numpy as np
import pandas as pd

from vtreat.partial_pooling_estimator import pooled_effect_estimate, standard_effect_estimate

def test_standard_effect_estimate():
    d = pd.DataFrame({
        'location_id': ['a', 'a', 'a', 'b', 'b', 'c'],
        'observation': [  1,   2,   3,   4,   5,   6],
    })
    r = standard_effect_estimate(d)
    expect = pd.DataFrame({
        'location_id': ['a', 'b', 'c'],
        'mean': [2.0, 4.5, 6.0],
        'var': [1.0, 0.5, np.nan],
        'size': [3, 2, 1],
        'estimate': [2.0, 4.5, 6.0],
        'grand_mean': [3.5, 3.5, 3.5],
        'impact': [-1.5, 1.0, 2.5],
    })
    assert r.equals(expect)


def test_pooled_effect_estimate():
    d = pd.DataFrame({
        'location_id': ['a', 'a', 'a', 'b', 'b', 'c'],
        'observation': [  1,   2,   3,   4,   5,   6],
    })
    r = pooled_effect_estimate(d)
    expect = pd.DataFrame({
        'location_id': ['a', 'b', 'c'],
        'mean': [2.0, 4.5, 6.0],
        'var': [1.0, 0.5, np.nan],
        'size': [3, 2, 1],
        'estimate': [2.161608, 4.362170, 5.342105],
        'grand_mean': [3.5, 3.5, 3.5],
        'impact': [-1.338392, 0.862170, 1.842105],
    })
    assert r.shape == expect.shape
    assert np.all(r.columns == expect.columns)
    assert np.all(r['location_id'] == expect['location_id'])
    for c in expect.columns:
        if c != 'location_id':
            assert np.max(np.abs(r[c] - expect[c])) < 1e-5
