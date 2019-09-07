import numpy
import numpy.random
import pandas
import vtreat


def test_multinomial():
    numpy.random.seed(2019)

    n_rows = 1000
    y_levels = ["a", "b", "c"]
    d = pandas.DataFrame({"y": numpy.random.choice(y_levels, size=n_rows)})
    # signal variables, correlated with y-levels
    for i in range(2):
        vmap = {yl: numpy.random.normal(size=1)[0] for yl in y_levels}
        d["var_n_" + str(i)] = [
            vmap[li] + numpy.random.normal(size=1)[0] for li in d["y"]
        ]
    for i in range(2):
        col = numpy.random.choice(y_levels, size=n_rows)
        col = [
            col[i] if numpy.random.uniform(size=1)[0] <= 0.8 else d["y"][i]
            for i in range(n_rows)
        ]
        d["var_c_" + str(i)] = col
    # noise variables, uncorrelated with y-levels
    for i in range(2):
        d["noise_n_" + str(i)] = [
            numpy.random.normal(size=1)[0] + numpy.random.normal(size=1)[0]
            for li in d["y"]
        ]
    for i in range(2):
        d["noise_c_" + str(i)] = numpy.random.choice(y_levels, size=n_rows)

    treatment = vtreat.MultinomialOutcomeTreatment(outcome_name="y")
    cross_frame = treatment.fit_transform(d, d["y"])

    for c in cross_frame.columns:
        if not c == 'y':
            assert vtreat.util.can_convert_v_to_numeric(cross_frame[c])
            assert sum(vtreat.util.is_bad(cross_frame[c])) == 0

    sf = treatment.score_frame_

    targets = sf.loc[
        numpy.logical_and(
            [v in ["var_n_0", "var_n_1"] for v in sf.variable],
            sf.treatment == "clean_copy",
        ),
        :,
    ]
    assert all(targets.recommended)
