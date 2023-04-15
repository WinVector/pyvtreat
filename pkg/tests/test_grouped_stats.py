
import numpy as np
import pandas as pd
from vtreat.util import grouped_by_x_statistics, pooled_impact_estimate
from vtreat.vtreat_impl import fit_binomial_impact_code, fit_regression_impact_code


def test_grouped_stats():
    # from: https://github.com/WinVector/Examples/blob/main/PartialPooling/PartialPooling.ipynb
    # set state of pseudo random number generator for repeatability
    rng = np.random.default_rng(2023)
    # set parameters to specific values
    example_between_locations_sd = 3.0
    example_per_observations_sd = 10.0
    n_locations = 10
    n_obs_per_location = 3

    def generate_example_centers() -> pd.DataFrame:
        """generate the unobserved location values"""
        example_location_value_mean = rng.normal(loc=0, scale=15, size=1)
        example_locations = pd.DataFrame({
            "location_id": range(n_locations),
            "effect": rng.normal(
            loc=example_location_value_mean, 
            scale=example_between_locations_sd, 
            size=n_locations)
        })
        return example_locations
    
    example_locations = generate_example_centers()

    def generate_observations(example_locations: pd.DataFrame)-> pd.DataFrame:
        """
        generate observed data

        :param example_locations: the (unobserved) ground truth to generate from
        :return: observed data
        """
        assert isinstance(example_locations, pd.DataFrame)
        observations = []
        for j in range(example_locations.shape[0]):
            obs_j = pd.DataFrame({
                "location_id": j,
                "observation": rng.normal(
                loc=example_locations.effect[j], 
                scale=example_per_observations_sd, 
                size=n_obs_per_location),
            })
            observations.append(obs_j)
        observations = pd.concat(observations, ignore_index=True)
        return observations

    observations = generate_observations(example_locations)
    grouped_stats = grouped_by_x_statistics(observations["location_id"], observations["observation"])
    assert isinstance(grouped_stats, pd.DataFrame)
    pooled_stats = pooled_impact_estimate(observations["location_id"], observations["observation"])
    assert isinstance(pooled_stats, pd.DataFrame)
    assert grouped_stats.shape[0] == pooled_stats.shape[0]
    assert np.all(pooled_stats['x'] == grouped_stats['x'])
    xform = fit_binomial_impact_code(
        incoming_column_name="test",
        x=observations["location_id"],
        y=observations["observation"] >= 0,
        extra_args={"outcome_target": True, "var_suffix": "_v"},
        params={"use_hierarchical_estimate": True},
    )
    xfc = xform.code_book_
    assert isinstance(xfc, pd.DataFrame)
    v = "test_logit_code_v"
    assert np.max(xfc[v]) > np.min(xfc[v])
    xform = fit_regression_impact_code(
        incoming_column_name="test",
        x=observations["location_id"],
        y=observations["observation"],
        extra_args={"var_suffix": "_v"},
        params={"use_hierarchical_estimate": True},
    )
    xfc = xform.code_book_
    assert isinstance(xfc, pd.DataFrame)
    v = "test_impact_code"
    assert np.max(xfc[v]) > np.min(xfc[v])

