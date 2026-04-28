"""tests/test_speed.py — Basic correctness checks for dfc_speed."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dfc_speed import (
    compute_dfc_speed,
    compute_dfc_speed_batch,
    speed_summary,
    permutation_test_speed,
    bootstrap_mean_ci,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def fc_seq(rng):
    """(T=20, N=10, N=10) random symmetric FC sequence."""
    T, N = 20, 10
    mats = []
    for _ in range(T):
        m = rng.standard_normal((N, N))
        m = (m + m.T) / 2
        np.fill_diagonal(m, 1.0)
        mats.append(m)
    return np.stack(mats)


def test_speed_shape(fc_seq):
    speed = compute_dfc_speed(fc_seq)
    assert speed.shape == (fc_seq.shape[0] - 1,)


def test_speed_nonnegative(fc_seq):
    speed = compute_dfc_speed(fc_seq)
    assert np.all(speed >= 0)


def test_speed_cosine(fc_seq):
    speed = compute_dfc_speed(fc_seq, metric="cosine")
    assert speed.shape == (fc_seq.shape[0] - 1,)
    assert np.all(speed >= 0)


def test_batch_shape(rng):
    S, T, N = 5, 20, 10
    fc_batch = rng.standard_normal((S, T, N, N))
    fc_batch = (fc_batch + fc_batch.transpose(0, 1, 3, 2)) / 2
    speeds = compute_dfc_speed_batch(fc_batch)
    assert speeds.shape == (S, T - 1)


def test_speed_summary(fc_seq):
    speed = compute_dfc_speed(fc_seq)
    s = speed_summary(speed)
    assert all(k in s for k in ["mean", "std", "median", "iqr"])
    assert s["std"] >= 0


def test_permutation_test(rng, fc_seq):
    # two groups of 6 subjects each
    group_a = [compute_dfc_speed(fc_seq + 0.01 * rng.standard_normal(fc_seq.shape))
               for _ in range(6)]
    group_b = [compute_dfc_speed(fc_seq + 0.50 * rng.standard_normal(fc_seq.shape))
               for _ in range(6)]
    diff, p = permutation_test_speed(group_a, group_b, n_permutations=500)
    assert isinstance(diff, float)
    assert 0.0 <= p <= 1.0


def test_bootstrap_ci(rng, fc_seq):
    speed_list = [compute_dfc_speed(fc_seq + 0.1 * rng.standard_normal(fc_seq.shape))
                  for _ in range(8)]
    m, lo, hi = bootstrap_mean_ci(speed_list, n_boot=500)
    assert lo <= m <= hi
