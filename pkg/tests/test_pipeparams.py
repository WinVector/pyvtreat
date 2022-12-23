#%% md

# From [pyvtreat issue 12](https://github.com/WinVector/pyvtreat/issues/12)

#%%

import pytest
import warnings

import pandas as pd
import numpy as np
import numpy.random
import vtreat
import vtreat.util
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def test_pipeparams():
    numpy.random.seed(2019)

    def make_data(nrows):
        d = pd.DataFrame({"x": 5 * numpy.random.normal(size=nrows)})
        d["y"] = numpy.sin(d["x"]) + 0.1 * numpy.random.normal(size=nrows)
        d.loc[numpy.arange(3, 10), "x"] = numpy.nan  # introduce a nan level
        d["xc"] = ["level_" + str(5 * numpy.round(yi / 5, 1)) for yi in d["y"]]
        d["x2"] = np.random.normal(size=nrows)
        d.loc[d["xc"] == "level_-1.0", "xc"] = numpy.nan  # introduce a nan level
        d["yc"] = d["y"] > 0.5
        return d

    df = make_data(500)

    df = df.drop(columns=["y"])

    transform = vtreat.BinomialOutcomeTreatment(
        outcome_target=True,
        params=vtreat.vtreat_parameters({"sparse_indicators": False}),
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", transform),
            ("classifier", LogisticRegression(solver="lbfgs")),
        ]
    )

    X, y = df, df.pop("yc")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf.fit(X_train, y_train)

    #%%

    t_params = transform.get_params()
    assert t_params["indicator_min_fraction"] is not None
    assert transform.get_params()["indicator_min_fraction"] != 0

    #%%

    p_params = clf.get_params()
    assert p_params["preprocessor__indicator_min_fraction"] is not None

    #%%

    clf.set_params(preprocessor__indicator_min_fraction=0)
    assert transform.get_params()["indicator_min_fraction"] == 0

    # no warning or error
    # https://docs.pytest.org/en/7.0.x/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clf.fit(X_train, y_train)
