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
        .agg(['mean', 'var', 'size'])
        .reset_index(drop=False, inplace=False)
    )
    cols = [' '.join(col).strip() for col in means.columns]
    means.columns = [c.removeprefix('observation ') for c in cols]
    means.sort_values(["location_id"], inplace=True, ignore_index=True)
    means['estimate'] = means['mean']
    means["grand_mean"] = np.mean(observations["observation"])
    means["impact"] = means["estimate"]
    means["impact"] = (
        means["impact"] 
        - np.sum(means['size'] * means['impact']) / np.sum(means['size']))
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
    n_j = estimated_centers["size"]
    per_location_observation_var = estimated_centers['var'].copy()
    per_location_observation_var[pd.isnull(per_location_observation_var)] = 0
    # inflate per-loc a bit
    per_location_observation_var = (
        (n_j * per_location_observation_var + np.var(observations['observation'])) 
        / (n_j + 1))
    # get the observed variance between locations
    between_location_var = np.var(estimated_centers["estimate"], ddof=1)
    # get v, the pooling coefficient
    if between_location_var <= 0:
        v = 0 * per_location_observation_var
    else:
        # as between_location_var > 0 and per_location_observation_var > 0 here
        # v will be in the range 0 to 1
        v = 1 / (1 + per_location_observation_var / (n_j * between_location_var))
    v[n_j <= 1] = 0  # no information in size one items
    v[pd.isnull(v)] = 0
    # build the pooled estimate
    pooled_estimate = v * estimated_centers["estimate"] + (1 - v) * estimated_centers["grand_mean"]
    estimated_centers["estimate"] = pooled_estimate
    estimated_centers["impact"] = estimated_centers["estimate"] 
    estimated_centers["impact"] = (
        estimated_centers["impact"] 
        - np.sum(estimated_centers['size'] * estimated_centers['impact']) / np.sum(estimated_centers['size']))
    return estimated_centers
