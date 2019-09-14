import vtreat
import pandas
import numpy


def test_r1_issue():
    plan = vtreat.NumericOutcomeTreatment(
        outcome_name="y",
        params=vtreat.vtreat_parameters({"filter_to_recommended": False}),
    )

    # from https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
    df = pandas.DataFrame(
        numpy.random.randn(5, 3),
        index=["a", "c", "e", "f", "h"],
        columns=["one", "two", "three"],
    )
    df["four"] = "foo"
    df["five"] = df["one"] > 0
    df2 = df.reindex(["a", "b", "c", "d", "e", "f", "g", "h"])
    df2.reset_index(inplace=True, drop=True)
    df2["y"] = range(df2.shape[0])
    df2.loc[3, "four"] = "blog"
    df2["const"] = 1

    vtreat.util.is_bad(df2["five"])
    prepped = plan.fit_transform(df2, df2["y"])  # used to raise an exception

    for c in prepped.columns:
        assert vtreat.util.can_convert_v_to_numeric(prepped[c])
        assert sum(vtreat.util.is_bad(prepped[c])) == 0
