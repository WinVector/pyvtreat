import numpy as np
import pandas as pd


# define the standard estimator
def standard_effect_estimate(observations: pd.DataFrame) -> pd.DataFrame:
    """
    Get the standard estimate of the effect at locations from observations.
    Please see: https://github.com/WinVector/Examples/blob/main/PartialPooling/PartialPooling.ipynb

    :param observations: the observations data frame
    :return: standard estimate of effect or mean by location
    """
    assert isinstance(observations, pd.DataFrame)
    means = (
        observations.loc[:, ["location_id", "observation"]]
        .reset_index(drop=True, inplace=False)
        .groupby(["location_id"])
        .mean()
        .reset_index(drop=False, inplace=False)
    )
    means.sort_values(["location_id"], inplace=True, ignore_index=True)
    means.rename(columns={"observation": "estimate"}, inplace=True)
    means["grand_mean"] = np.mean(observations["observation"])
    means["impact"] = means["estimate"] - means["grand_mean"]
    means.sort_values(["location_id"], inplace=True, ignore_index=True)
    return means


# define the pooled estimator
def pooled_effect_estimate(observations: pd.DataFrame) -> pd.DataFrame:
    """
    Get the pooled estimate of effect.
    See: https://github.com/WinVector/Examples/blob/main/PartialPooling/PartialPooling.ipynb .

    :param observations: the observations data frame, with columns location_id and observation
    :return: pooled estimates
    """
    assert isinstance(observations, pd.DataFrame)
    observations = observations.loc[:, ["location_id", "observation"]].reset_index(
        inplace=False, drop=True
    )
    # get the standard estimates
    estimated_centers = standard_effect_estimate(observations=observations)
    if estimated_centers.shape[0] <= 1:
        # no pooling possible
        return estimated_centers
    # get counts per group
    obs_count_frame = (
        pd.DataFrame({"location_id": observations["location_id"], "count": 1})
        .groupby(["location_id"])
        .sum()
        .reset_index(drop=False, inplace=False)
        .sort_values(["location_id"], inplace=False, ignore_index=True)
    )
    n_j = obs_count_frame["count"].values
    # get the observed variance for each item at for each location
    combined = observations.merge(
        estimated_centers,
        on=["location_id"],
        how="left",
    ).merge(
        obs_count_frame,
        on=["location_id"],
        how="left",
    )
    combined.sort_values(["location_id"], inplace=True, ignore_index=True)
    per_location_observation_var = np.sum(
        (combined["observation"] - combined["estimate"]) ** 2
    ) / (combined.shape[0] - len(set(combined["location_id"])))
    # get the observed variance between locations
    between_location_var = np.var(estimated_centers["estimate"], ddof=1)
    # get v, the pooling coefficient
    if between_location_var <= 0:
        v = 0
    elif per_location_observation_var <= 0:
        v = 1
    else:
        # as between_location_var > 0 and per_location_observation_var > 0 here
        # v will be in the range 0 to 1
        v = 1 / (1 + per_location_observation_var / (n_j * between_location_var))
    # our estimate of the overall shared effect
    # note we are using the mixing proportions suggested by the variance reduction ideas
    # simpler weightings include:
    #   combined["obs_weight"] = 1   # weights all observations equally
    #   combined["obs_weight"] = 1 / combined["count"]  # weights all locations equally
    # below, weights larger observations groups more, but with a diminishing return
    # this is an ad-hoc heuristic to try to reduce square error when the number of
    # observations per location has a lot of variation
    combined["obs_weight"] = 1
    if (between_location_var > 0) and (per_location_observation_var > 0):
        combined["obs_weight"] = 1 / (
            1
            + per_location_observation_var / (combined["count"] * between_location_var)
        )
    # this quantity can be improved using knowledge of the variances
    grand_mean = np.sum(combined["observation"] * combined["obs_weight"]) / np.sum(
        combined["obs_weight"]
    )
    # build the pooled estimate
    pooled_estimate = v * estimated_centers["estimate"] + (1 - v) * grand_mean
    return pd.DataFrame(
        {
            "location_id": estimated_centers["location_id"],
            "estimate": pooled_estimate,
            "grand_mean": grand_mean,
            "impact": pooled_estimate - grand_mean,
        }
    )
