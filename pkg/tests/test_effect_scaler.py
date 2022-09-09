
import numpy as np
import pandas as pd
import vtreat.effect_scaler


def test_effect_scaler_pd_1():
    d = pd.DataFrame({'x': [0.5, -0.5], 'y': [1, -1]})
    es = vtreat.effect_scaler.EffectScaler()
    es.fit(d.loc[:, ['x']], d['y'])
    res = es.transform(d.loc[:, ['x']])
    assert isinstance(res, pd.DataFrame)
    assert np.all(np.array(res['x']) == np.array([1.0, -1.0]))
    assert list(res.columns) == list(d.loc[:, ['x']].columns)


def test_effect_scaler_np_1():
    x = np.array([[0.5], [-0.5]])
    y = np.array([1, -1])
    es = vtreat.effect_scaler.EffectScaler()
    es.fit(x, y)
    res = es.transform(x)
    assert isinstance(res, pd.DataFrame)
    assert np.all(np.array(res[0]) == np.array([1.0, -1.0]))
