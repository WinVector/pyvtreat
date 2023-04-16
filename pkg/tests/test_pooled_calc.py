

import numpy as np
import pandas as pd
from vtreat.partial_pooling_estimator import standard_effect_estimate, pooled_effect_estimate


def test_pooled_calc():
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

    # estimate the quality of the standard estimator
    def get_sq_error_loss_std_estimate(
            *, 
            location_id: int, example_locations: pd.DataFrame, observations: pd.DataFrame) -> float:
        """
        Show the square-error of estimate of the effect at location location_id from observations

        :parm location_id: which location to calculate for
        :param example_locations: the (unobserved) ground truth to compare to
        :param observations: the observations data frame
        :return: estimated square error of this estimation procedure
        """
        # get the standard estimates
        observed_means = standard_effect_estimate(observations=observations)
        # get the unobservable true effect for comparison
        true_effect = example_locations.loc[example_locations["location_id"] == location_id, "effect"].values[0]
        # calculate the square error of these estimates
        estimated_effect = observed_means.loc[observed_means["location_id"] == location_id, "estimate"].values[0]
        square_error = (estimated_effect - true_effect)**2
        return square_error

    # evaluate the pooled estimator quality
    def get_sq_error_loss_pooled_estimate(
            *, 
            location_id: int, example_locations: pd.DataFrame, observations: pd.DataFrame) -> float:
        """
        Show the square error of partial pooled out estimates of the effect at location location_id from observations

        :parm location_id: which location to calculate for
        :param example_locations: the (unobserved) ground truth to compare to
        :param observations: the observations data frame
        :return: estimated square error of this estimation procedure
        """
        # get the estimates
        pooled_estimates = pooled_effect_estimate(observations=observations)
        # get the unobservable true effect for comparison
        true_effect = example_locations.loc[example_locations["location_id"] == location_id, "effect"].values[0]
        # calculate the square error of these estimates
        estimated_effect = pooled_estimates.loc[pooled_estimates["location_id"] == location_id, "estimate"].values[0]
        square_error = (estimated_effect - true_effect)**2
        return square_error
    
    # run the experiment for the standard estimator
    std_est_loss = get_sq_error_loss_std_estimate(
        location_id=0, 
        example_locations=example_locations, 
        observations=observations)
    # run the experiment for the pooled estimator
    pooled_est_loss = get_sq_error_loss_pooled_estimate(
        location_id=0, 
        example_locations=example_locations, 
        observations=observations)
    assert pooled_est_loss < std_est_loss


def test_pooled_calc_2():
    # more obs takes us closer to true expectation
    rng = np.random.default_rng(2023)
    d = pd.DataFrame({
        "location_id": ["a"] * 100 + ["b"] * 5,
    })
    d["observation"] = 1.0 + rng.normal(size=d.shape[0])
    pooled_estimates = pooled_effect_estimate(observations=d)
    ests = {lid: est for lid, est in zip(pooled_estimates["location_id"], pooled_estimates["estimate"])}
    assert np.abs(ests['a'] - 1.0) < np.abs(ests['b'] - 1.0)
