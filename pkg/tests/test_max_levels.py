import pandas
import vtreat
import vtreat.util


def test_max_levels_1():
    d = pandas.DataFrame(
        {
            # id-like variables are not encoded
            "x": [str(i) for i in range(1000)]
            + ["500"]
        }
    )
    treatment_plan = vtreat.UnsupervisedTreatment(
        var_list=["x"],
        params=vtreat.unsupervised_parameters({"indicator_max_levels": 100}),
    )
    treatment_plan.fit_transform(d)
    sf = treatment_plan.score_frame_
    inds = sf.loc[sf["treatment"] == "indicator_code", :].reset_index(
        inplace=False, drop=True
    )
    assert inds.shape[0] == 100
    assert "x_lev_500" in inds["variable"].values
    assert "x_lev_200" not in inds["variable"].values
