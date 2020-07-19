import pandas
import numpy
import numpy.random
import vtreat
import vtreat.util


def test_classification():
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
    vars = [c for c in d.columns if c not in set(['y', 'yc'])]
    d_test = make_data(100)

    transform = vtreat.BinomialOutcomeTreatment(
        outcome_name="yc",  # outcome variable
        outcome_target=True,  # outcome of interest
        cols_to_copy=["y"],  # columns to "carry along" but not treat as input variables
        params=vtreat.vtreat_parameters({
            'filter_to_recommended': False
        })
    )
    d_prepared = transform.fit_transform(d[vars], d["yc"])

    # show vars are under control
    assert transform.get_result_restriction() is None
    assert 'x2' in set(d_prepared.columns)

    transform.set_result_restriction(['xc_logit_code', 'x2'])
    dt_prepared = transform.transform(d_test)
    assert set(dt_prepared.columns) == set(['y', 'yc', 'x2', 'xc_logit_code'])

    transform = vtreat.BinomialOutcomeTreatment(
        outcome_name="yc",  # outcome variable
        outcome_target=True,  # outcome of interest
        cols_to_copy=["y"],  # columns to "carry along" but not treat as input variables
        params=vtreat.vtreat_parameters({
            'filter_to_recommended': True
        })
    )
    d_prepared = transform.fit_transform(d[vars], d["yc"])

    assert transform.get_result_restriction() is not None
    assert 'x2' not in transform.get_result_restriction()
    assert 'x2' not in set(d_prepared.columns)

    transform.set_result_restriction(['xc_logit_code', 'x2'])
    dt_prepared = transform.transform(d_test)
    assert set(dt_prepared.columns) == set(['y', 'yc', 'x2', 'xc_logit_code'])
