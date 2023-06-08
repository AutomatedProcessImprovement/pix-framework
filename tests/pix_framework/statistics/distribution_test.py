import random

import numpy as np
import pytest
import scipy.stats as st
from pix_framework.statistics.distribution import (
    DistributionType, DurationDistribution, get_best_fitting_distribution)


def test_infer_distribution_fixed():
    data = [150] * 1000
    distribution = get_best_fitting_distribution(data)
    assert distribution.type == DistributionType.FIXED
    assert distribution.mean == 150.0


def test_infer_distribution_fixed_with_noise():
    data = [149] * 100 + [150] * 1000 + [151] * 100 + [200] * 5
    distribution = get_best_fitting_distribution(data)
    assert distribution.type == DistributionType.FIXED
    assert distribution.mean == 150.0


def test_infer_distribution_not_fixed():
    data = [147] * 28 + [150] * 26 + [151] * 32 + [240] * 14
    distribution = get_best_fitting_distribution(data)
    assert distribution.type != DistributionType.FIXED


def test_infer_distribution_normal():
    distribution = st.norm(loc=100, scale=20)
    data = distribution.rvs(size=5000)
    distribution = get_best_fitting_distribution(data)
    assert distribution.type == DistributionType.NORMAL
    _assert_distribution_params(distribution, data)


def test_infer_distribution_exponential():
    distribution = st.expon(loc=2, scale=700)
    data = distribution.rvs(size=1000)
    distribution = get_best_fitting_distribution(data)
    assert distribution.type in [
        DistributionType.EXPONENTIAL,
        DistributionType.GAMMA,
        DistributionType.LOG_NORMAL,
    ]
    _assert_distribution_params(distribution, data)


def test_infer_distribution_uniform():
    distribution = st.uniform(loc=600, scale=120)
    data = distribution.rvs(size=1000)
    distribution = get_best_fitting_distribution(data)
    assert distribution.type == DistributionType.UNIFORM
    _assert_distribution_params(distribution, data)


def test_infer_distribution_log_normal():
    distribution = st.lognorm(s=0.5, loc=600, scale=300)
    data = distribution.rvs(size=1000)
    distribution = get_best_fitting_distribution(data)
    assert distribution.type in [
        DistributionType.LOG_NORMAL,
        DistributionType.EXPONENTIAL,
        DistributionType.GAMMA,
    ]
    _assert_distribution_params(distribution, data)


def test_infer_distribution_gamma():
    distribution = st.gamma(a=0.7, loc=600, scale=300)
    data = distribution.rvs(size=1000)
    distribution = get_best_fitting_distribution(data)
    assert distribution.type in [
        DistributionType.GAMMA,
        DistributionType.LOG_NORMAL,
        DistributionType.EXPONENTIAL,
    ]
    _assert_distribution_params(distribution, data)


def _assert_distribution_params(distribution, data):
    assert distribution.mean == np.mean(data)
    assert distribution.var == np.var(data)
    assert distribution.std == np.std(data)
    assert distribution.min == np.min(data)
    assert distribution.max == np.max(data)


def test_scale_distributions():
    distributions = [
        DurationDistribution(name="fix", mean=60, var=0, std=0, minimum=60, maximum=60),
        DurationDistribution(
            name="norm", mean=1200, var=36, std=6, minimum=1000, maximum=1400
        ),
        DurationDistribution(
            name="expon", mean=3600, var=100, std=10, minimum=1200, maximum=7200
        ),
        DurationDistribution(
            name="uniform", mean=3600, var=4000000, std=2000, minimum=0, maximum=7200
        ),
        DurationDistribution(
            name="lognorm", mean=120, var=100, std=10, minimum=100, maximum=190
        ),
        DurationDistribution(
            name="gamma", mean=1200, var=144, std=12, minimum=800, maximum=1400
        ),
    ]
    for distribution in distributions:
        alpha = random.randrange(1, 500) / 100
        scaled = distribution.scale_distribution(alpha)
        assert scaled.type == distribution.type
        assert scaled.mean == distribution.mean * alpha
        assert scaled.var == distribution.var * alpha * alpha
        assert scaled.std == distribution.std * alpha
        assert scaled.min == distribution.min * alpha
        assert scaled.max == distribution.max * alpha

def get_prosimos_dict(distribution_name, distribution_params):
    params = []
    for param in distribution_params:
        params.append({"value": param})
    
    return {
        "distribution_name": distribution_name,
        "distribution_params": params
    }

duration_dict = [
    (get_prosimos_dict("fix", [10]), DurationDistribution(DistributionType.FIXED, mean=10.0)),
    (get_prosimos_dict("expon", [10, 100, 500]), DurationDistribution(DistributionType.EXPONENTIAL, mean=10.0, minimum=100.0, maximum=500.0)),
    (get_prosimos_dict("uniform", [100, 200]), DurationDistribution(DistributionType.UNIFORM, minimum=100.0, maximum=200.0)),
    (get_prosimos_dict("norm", [10, 20, 100, 200]), DurationDistribution(DistributionType.NORMAL, mean=10.0, std=20.0, minimum=100.0, maximum=200.0)),
    (get_prosimos_dict("lognorm", [10, 20, 100, 200]), DurationDistribution(DistributionType.LOG_NORMAL, mean=10.0, var=20.0, minimum=100.0, maximum=200.0)),
    (get_prosimos_dict("gamma", [10, 20, 100, 200]), DurationDistribution(DistributionType.GAMMA, mean=10.0, var=20.0, minimum=100.0, maximum=200.0)),
]

@pytest.mark.parametrize(
    "duration_dict, expected_distribution",
    duration_dict
)

def test_deserialization(duration_dict: dict, expected_distribution: DurationDistribution):
    # act: perform the deserialization
    actual_result = DurationDistribution.from_dict(duration_dict)

    # assert: check whether class fields are equal by comparing strings
    assert str(actual_result) == str(expected_distribution)

@pytest.mark.parametrize(
    "expected_prosimos_dict, input_distribution",
    duration_dict
)

def test_serialization(expected_prosimos_dict: dict, input_distribution: DurationDistribution):
    # act: convert class instance to dictionary used as an input for Prosimos
    actual_result = input_distribution.to_prosimos_distribution()

    # assert: compare received dictionaries
    assert actual_result == expected_prosimos_dict
