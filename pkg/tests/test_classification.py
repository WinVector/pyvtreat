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

    transform = vtreat.BinomialOutcomeTreatment(
        outcome_name="yc",  # outcome variable
        outcome_target=True,  # outcome of interest
        cols_to_copy=["y"],  # columns to "carry along" but not treat as input variables
    )

    # show y-column doesn't get copied in, and can tolerate copy columns not being around
    vars = [c for c in d.columns if c not in set(['y', 'yc'])]
    d_prepared = transform.fit_transform(d[vars], d["yc"])
    assert 'yc' not in d_prepared.columns

    # design again

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
        d = pandas.DataFrame({"x": [0.1 * i for i in range(nrows)]})
        d["y"] = d["x"] + numpy.sin(d["x"]) + 0.1 * numpy.random.normal(size=d.shape[0])
        d["xc"] = ["level_" + str(5 * numpy.round(yi / 5, 1)) for yi in d["y"]]
        d["x2"] = numpy.random.normal(size=d.shape[0])
        d.loc[d["xc"] == "level_-1.0", "xc"] = numpy.nan  # introduce a nan level
        d["yc"] = d["y"] > 0.5
        return d

    d = make_data(500)
    vars = [v for v in d.columns if v not in ['y', 'c']]
    d_n = numpy.asarray(d[vars])

    transform = vtreat.BinomialOutcomeTreatment(
        outcome_target=True,  # outcome of interest
    )

    d_prepared = transform.fit_transform(d_n, numpy.asarray(d["yc"]))
    assert isinstance(d_prepared, numpy.ndarray)
    d_prepared_columns = transform.last_result_columns
    sf = transform.score_frame_

    assert len(set(d_prepared_columns) - set(sf.variable)) == 0

    dtest = make_data(450)

    dtest_prepared = transform.transform(numpy.asarray(dtest[vars]))
    assert isinstance(dtest_prepared, numpy.ndarray)
    dtest_prepared_columns = transform.last_result_columns

    assert len(set(dtest_prepared_columns) - set(sf.variable)) == 0


def test_classification_type_free():
    # confirm incoming column type does not matter during apply
    numpy.random.seed(46546)

    def make_data(nrows):
        d = pandas.DataFrame({"x": numpy.random.normal(size=nrows)})
        d["y"] = d["x"] + numpy.random.normal(size=nrows)
        d["xcn"] = numpy.round(d['x']/5, 1)*5
        d["yc"] = d["y"] > 0
        return d

    d = make_data(100)
    d_head = d.loc[range(10), :].copy()
    d_train = d.copy()
    d_train['xcn'] = d_train['xcn'].astype(str)
    vars = ['x', 'xcn']

    transform = vtreat.BinomialOutcomeTreatment(
        outcome_target=True,  # outcome of interest
        outcome_name='yc',
        cols_to_copy=['y']
    )

    d_prepared = transform.fit_transform(d_train, numpy.asarray(d["yc"]))
    d_train_head = d_train.loc[range(10), :].copy()

    t1 = transform.transform(d_train_head)
    t2 = transform.transform(d_head)
    assert t1.equals(t2)
