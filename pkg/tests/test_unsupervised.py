import numpy.random
import pandas
import vtreat  # https://github.com/WinVector/pyvtreat


def test_unsupervised():
    numpy.random.seed(235)
    zip = ["z" + str(i + 1).zfill(5) for i in range(15)]
    d = pandas.DataFrame({"zip": numpy.random.choice(zip, size=1000)})
    d["const"] = 1
    d["const2"] = "b"

    transform = vtreat.UnsupervisedTreatment(
        params=vtreat.unsupervised_parameters({"indicator_min_fraction": 0.01})
    )

    d_treated = transform.fit_transform(d)

    for c in d_treated.columns:
        assert vtreat.util.can_convert_v_to_numeric(d_treated[c])
        assert numpy.sum(vtreat.util.is_bad(d_treated[c])) == 0

    sf = transform.score_frame_

    d_treated_2 = transform.transform(d)

