import numpy
import numpy.random
import pandas
import vtreat  # https://github.com/WinVector/pyvtreat


def test_unsupervised():
    n_rows = 10000
    n_levels = 10
    n_cat = 10
    n_numeric = 10
    numpy.random.seed(235)
    zip = ["z" + str(i + 1).zfill(5) for i in range(n_levels)]
    d = pandas.DataFrame({"const": numpy.zeros(n_rows) + 1})
    d["const2"] = "b"
    for i in range(n_cat):
        d[f"zip_{i}"] = numpy.random.choice(zip, size=n_rows)
    for i in range(n_numeric):
        d[f"num_{i}"] = numpy.random.uniform(size=n_rows)

    transform = vtreat.UnsupervisedTreatment(
        params=vtreat.unsupervised_parameters({"indicator_min_fraction": 0.01})
    )

    ## https://docs.python.org/3/library/profile.html
    # import cProfile
    # cProfile.run('d_treated = transform.fit_transform(d)')
    d_treated = transform.fit_transform(d)

    for c in d_treated.columns:
        assert vtreat.util.can_convert_v_to_numeric(d_treated[c])
        assert numpy.sum(vtreat.util.is_bad(d_treated[c])) == 0

    sf = transform.score_frame_
