import pandas
import numpy
import numpy.random
import vtreat
import vtreat.util


def test_classification_numpy():
    numpy.random.seed(46546)

    def make_data(nrows):
        d = pandas.DataFrame({"x": [0.1 * i for i in range(500)]})
        d["y"] = d["x"] + numpy.sin(d["x"]) + 0.1 * numpy.random.normal(size=d.shape[0])
        d["xc"] = ["level_" + str(5 * numpy.round(yi / 5, 1)) for yi in d["y"]]
        d["x2"] = numpy.random.normal(size=d.shape[0])
        d.loc[d["xc"] == "level_-1.0", "xc"] = numpy.nan  # introduce a nan level
        d["yc"] = d["y"] > 0.5
        return d

    d = make_data(5000)

    transform = vtreat.BinomialOutcomeTreatment(
        outcome_name="yc",  # outcome variable
        outcome_target=True,  # outcome of interest
        cols_to_copy=["y"],  # columns to "carry along" but not treat as input variables
    )

    d_prepared = transform.fit_transform(d, d["yc"])

    for c in d_prepared.columns:
        assert vtreat.util.can_convert_v_to_numeric(d_prepared[c])
        assert sum(vtreat.util.is_bad(d_prepared[c])) == 0

    dtest = make_data(450)

    dtest_prepared = transform.transform(dtest)

    for c in dtest_prepared.columns:
        assert vtreat.util.can_convert_v_to_numeric(dtest_prepared[c])
        assert sum(vtreat.util.is_bad(dtest_prepared[c])) == 0

    sf = transform.score_frame_

    xrow = sf.loc[
        numpy.logical_and(sf.variable == "x", sf.treatment == "clean_copy"), :
    ]
    xrow.reset_index(inplace=True, drop=True)

    assert xrow.recommended[0]


def test_classification_numpy():
    numpy.random.seed(46546)

    def make_data(nrows):
        d = pandas.DataFrame({"x": [0.1 * i for i in range(500)]})
        d["y"] = d["x"] + numpy.sin(d["x"]) + 0.1 * numpy.random.normal(size=d.shape[0])
        d["xc"] = ["level_" + str(5 * numpy.round(yi / 5, 1)) for yi in d["y"]]
        d["x2"] = numpy.random.normal(size=d.shape[0])
        d.loc[d["xc"] == "level_-1.0", "xc"] = numpy.nan  # introduce a nan level
        d["yc"] = d["y"] > 0.5
        return d

    d = make_data(5000)
    vars = [v for v in d.columns if v not in ['y', 'c']]
    d_n = numpy.asarray(d[vars])

    transform = vtreat.BinomialOutcomeTreatment(
        outcome_target=True,  # outcome of interest
    )

    d_prepared = transform.fit_transform(d_n, numpy.asarray(d["yc"]))

    for c in d_prepared.columns:
        assert vtreat.util.can_convert_v_to_numeric(d_prepared[c])
        assert sum(vtreat.util.is_bad(d_prepared[c])) == 0

    dtest = make_data(450)

    dtest_prepared = transform.transform(numpy.asarray(dtest[vars]))

    for c in dtest_prepared.columns:
        assert vtreat.util.can_convert_v_to_numeric(dtest_prepared[c])
        assert sum(vtreat.util.is_bad(dtest_prepared[c])) == 0

    sf = transform.score_frame_
