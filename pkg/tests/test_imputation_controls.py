import pandas
import numpy
import vtreat  # https://github.com/WinVector/pyvtreat
import vtreat.util


def test_imputation_controls():

    d = pandas.DataFrame({"x": [0, 1, 1000, None], "y": [0, 0, 1, 1],})

    transform = vtreat.UnsupervisedTreatment(cols_to_copy=["y"],)
    d_treated = transform.fit_transform(d)
    expect = pandas.DataFrame(
        {
            "y": [0, 0, 1, 1],
            "x_is_bad": [0.0, 0.0, 0.0, 1.0],
            "x": [0.0, 1.0, 1000.0, 333.6666666667],
        }
    )
    vtreat.util.check_matching_numeric_frames(res=d_treated, expect=expect)

    transform = vtreat.UnsupervisedTreatment(
        cols_to_copy=["y"],
        params=vtreat.unsupervised_parameters(
            {"missingness_imputation": numpy.median,}
        ),
    )
    d_treated = transform.fit_transform(d)
    expect = pandas.DataFrame(
        {
            "y": [0, 0, 1, 1],
            "x_is_bad": [0.0, 0.0, 0.0, 1.0],
            "x": [0.0, 1.0, 1000.0, 1.0],
        }
    )
    vtreat.util.check_matching_numeric_frames(res=d_treated, expect=expect)

    transform = vtreat.UnsupervisedTreatment(
        cols_to_copy=["y"],
        params=vtreat.unsupervised_parameters({"missingness_imputation": numpy.min,}),
    )
    d_treated = transform.fit_transform(d)
    expect = pandas.DataFrame(
        {
            "y": [0, 0, 1, 1],
            "x_is_bad": [0.0, 0.0, 0.0, 1.0],
            "x": [0.0, 1.0, 1000.0, 0.0],
        }
    )
    vtreat.util.check_matching_numeric_frames(res=d_treated, expect=expect)

    transform = vtreat.UnsupervisedTreatment(
        cols_to_copy=["y"],
        params=vtreat.unsupervised_parameters({"missingness_imputation": 7,}),
        imputation_map={"y": numpy.median},
    )
    d_treated = transform.fit_transform(d)
    expect = pandas.DataFrame(
        {
            "y": [0, 0, 1, 1],
            "x_is_bad": [0.0, 0.0, 0.0, 1.0],
            "x": [0.0, 1.0, 1000.0, 7.0],
        }
    )
    vtreat.util.check_matching_numeric_frames(res=d_treated, expect=expect)

    transform = vtreat.UnsupervisedTreatment(
        cols_to_copy=["y"],
        params=vtreat.unsupervised_parameters({"missingness_imputation": 7,}),
        imputation_map={"x": numpy.median},
    )
    d_treated = transform.fit_transform(d)
    expect = pandas.DataFrame(
        {
            "y": [0, 0, 1, 1],
            "x_is_bad": [0.0, 0.0, 0.0, 1.0],
            "x": [0.0, 1.0, 1000.0, 1.0],
        }
    )
    vtreat.util.check_matching_numeric_frames(res=d_treated, expect=expect)

    transform = vtreat.UnsupervisedTreatment(
        cols_to_copy=["y"],
        params=vtreat.unsupervised_parameters({"missingness_imputation": numpy.mean,}),
        imputation_map={"x": 12},
    )
    d_treated = transform.fit_transform(d)
    expect = pandas.DataFrame(
        {
            "y": [0, 0, 1, 1],
            "x_is_bad": [0.0, 0.0, 0.0, 1.0],
            "x": [0.0, 1.0, 1000.0, 12.0],
        }
    )
    vtreat.util.check_matching_numeric_frames(res=d_treated, expect=expect)
