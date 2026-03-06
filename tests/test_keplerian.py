from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxoplanet.orbits.keplerian import Body, Central, System

from lightcurve_strategies import bodies, centrals, systems


class TestCentralsDefaults:
    """Verify that default strategies produce valid Central objects."""

    @given(c=centrals())
    @settings(max_examples=30)
    def test_returns_central(self, c):
        assert isinstance(c, Central)

    @given(c=centrals())
    @settings(max_examples=30)
    def test_mass_in_range(self, c):
        assert 0.1 <= float(c.mass) <= 10.0

    @given(c=centrals())
    @settings(max_examples=30)
    def test_radius_in_range(self, c):
        assert 0.1 <= float(c.radius) <= 10.0

    @given(c=centrals())
    @settings(max_examples=30)
    def test_density_positive(self, c):
        assert float(c.density) > 0.0


class TestCentralsOverrides:
    """Verify that user-supplied strategies override defaults."""

    @given(c=centrals(mass=st.just(1.0)))
    @settings(max_examples=10)
    def test_fixed_mass(self, c):
        assert float(c.mass) == pytest.approx(1.0)

    @given(c=centrals(mass=st.just(2.0), radius=st.just(3.0)))
    @settings(max_examples=10)
    def test_fixed_both(self, c):
        assert float(c.mass) == pytest.approx(2.0)
        assert float(c.radius) == pytest.approx(3.0)


class TestBodiesDefaults:
    """Verify that default strategies produce valid Body objects."""

    @given(b=bodies())
    @settings(max_examples=30)
    def test_returns_body(self, b):
        assert isinstance(b, Body)

    @given(b=bodies())
    @settings(max_examples=30)
    def test_period_in_range(self, b):
        assert 0.5 <= float(b.period) <= 365.0

    @given(b=bodies())
    @settings(max_examples=30)
    def test_impact_param_in_range(self, b):
        assert 0.0 <= float(b.impact_param) <= 1.0

    @given(b=bodies())
    @settings(max_examples=30)
    def test_circular_by_default(self, b):
        assert b.eccentricity is None

    @given(b=bodies())
    @settings(max_examples=30)
    def test_time_transit_bounded_by_period(self, b):
        assert 0.0 <= float(b.time_transit) <= float(b.period)


class TestBodiesOverrides:
    """Verify that user-supplied strategies override defaults."""

    @given(
        b=bodies(
            eccentricity=st.just(0.3),
            omega_peri=st.just(1.0),
        )
    )
    @settings(max_examples=10)
    def test_eccentricity_pair(self, b):
        assert float(b.eccentricity) == pytest.approx(0.3)
        assert float(b.omega_peri) == pytest.approx(1.0)

    @given(b=bodies(eccentricity=st.just(0.2)))
    @settings(max_examples=10)
    def test_eccentricity_auto_triggers_omega_peri(self, b):
        assert float(b.eccentricity) == pytest.approx(0.2)
        # omega_peri should have been drawn from default strategy
        assert 0.0 <= float(b.omega_peri) <= 6.283185307179586

    @given(b=bodies(mass=st.just(0.001), radius=st.just(0.1)))
    @settings(max_examples=10)
    def test_mass_and_radius(self, b):
        assert float(b.mass) == pytest.approx(0.001)
        assert float(b.radius) == pytest.approx(0.1)


class TestSystemsDefaults:
    """Verify that default strategies produce valid System objects."""

    @given(s=systems())
    @settings(max_examples=30)
    def test_returns_system(self, s):
        assert isinstance(s, System)

    @given(s=systems())
    @settings(max_examples=30)
    def test_has_central(self, s):
        assert isinstance(s.central, Central)

    @given(s=systems())
    @settings(max_examples=30)
    def test_body_count_in_range(self, s):
        assert 0 <= len(s.bodies) <= 5


class TestSystemsOverrides:
    """Verify that user-supplied strategies override defaults."""

    @given(s=systems(central=st.just(Central(mass=1.0, radius=1.0))))
    @settings(max_examples=10)
    def test_custom_central(self, s):
        assert float(s.central.mass) == pytest.approx(1.0)
        assert float(s.central.radius) == pytest.approx(1.0)

    @given(
        s=systems(
            body=bodies(period=st.just(10.0)),
            min_bodies=1,
            max_bodies=3,
        )
    )
    @settings(max_examples=10)
    def test_custom_body_strategy(self, s):
        assert 1 <= len(s.bodies) <= 3
        for b in s.bodies:
            assert float(b.period) == pytest.approx(10.0)

    @given(s=systems(min_bodies=2, max_bodies=4))
    @settings(max_examples=10)
    def test_min_max_body_counts(self, s):
        assert 2 <= len(s.bodies) <= 4

    @given(s=systems(min_bodies=0, max_bodies=0))
    @settings(max_examples=5)
    def test_zero_bodies(self, s):
        assert len(s.bodies) == 0
