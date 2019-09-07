import pandas
import numpy
import numpy.random
import vtreat
import vtreat.util


def test_classification():
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

    dtest = make_data(450)

    dtest_prepared = transform.transform(dtest)

    sf = transform.score_frame_

    xrow = sf.loc[
        numpy.logical_and(sf.variable == "x", sf.treatment == "clean_copy"), :
    ]
    xrow.reset_index(inplace=True, drop=True)

    assert xrow.recommended[0]
