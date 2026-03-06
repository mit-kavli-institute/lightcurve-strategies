from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxoplanet.orbits import TransitOrbit

from lightcurve_strategies import transit_orbits


class TestTransitOrbitsDefaults:
    """Verify that default strategies produce valid TransitOrbit objects."""

    @given(orbit=transit_orbits())
    @settings(max_examples=30)
    def test_returns_transit_orbit(self, orbit):
        assert isinstance(orbit, TransitOrbit)

    @given(orbit=transit_orbits())
    @settings(max_examples=30)
    def test_period_in_range(self, orbit):
        assert 0.5 <= float(orbit.period) <= 365.0

    @given(orbit=transit_orbits())
    @settings(max_examples=30)
    def test_duration_positive_and_bounded(self, orbit):
        dur = float(orbit.duration)
        per = float(orbit.period)
        assert 0.001 <= dur <= min(0.5 * per, 2.0)

    @given(orbit=transit_orbits())
    @settings(max_examples=30)
    def test_impact_param_in_range(self, orbit):
        assert 0.0 <= float(orbit.impact_param) <= 1.0

    @given(orbit=transit_orbits())
    @settings(max_examples=30)
    def test_radius_ratio_in_range(self, orbit):
        assert 0.001 <= float(orbit.radius_ratio) <= 0.3

    @given(orbit=transit_orbits())
    @settings(max_examples=30)
    def test_time_transit_in_range(self, orbit):
        assert 0.0 <= float(orbit.time_transit) <= float(orbit.period)

    @given(orbit=transit_orbits())
    @settings(max_examples=30)
    def test_speed_positive(self, orbit):
        assert float(orbit.speed) > 0.0


class TestTransitOrbitsOverrides:
    """Verify that user-supplied strategies override defaults."""

    @given(orbit=transit_orbits(period=st.just(10.0)))
    @settings(max_examples=10)
    def test_fixed_period(self, orbit):
        assert float(orbit.period) == pytest.approx(10.0)

    @given(
        orbit=transit_orbits(
            period=st.just(10.0),
            duration=st.just(0.2),
        )
    )
    @settings(max_examples=10)
    def test_fixed_period_and_duration(self, orbit):
        assert float(orbit.period) == pytest.approx(10.0)
        assert float(orbit.duration) == pytest.approx(0.2)

    @given(
        orbit=transit_orbits(
            period=st.just(5.0),
            duration=st.just(0.1),
            time_transit=st.just(1.0),
            impact_param=st.just(0.3),
            radius_ratio=st.just(0.05),
        )
    )
    @settings(max_examples=5)
    def test_all_fixed(self, orbit):
        assert float(orbit.period) == pytest.approx(5.0)
        assert float(orbit.duration) == pytest.approx(0.1)
        assert float(orbit.time_transit) == pytest.approx(1.0)
        assert float(orbit.impact_param) == pytest.approx(0.3)
        assert float(orbit.radius_ratio) == pytest.approx(0.05)


class TestTransitOrbitsAstropyQuantities:
    """Verify that astropy Quantity inputs are handled correctly."""

    astropy_units = pytest.importorskip("astropy.units")

    @given(
        orbit=transit_orbits(
            period=st.just(pytest.importorskip("astropy.units").Quantity(24.0, "h")),
        )
    )
    @settings(max_examples=5)
    def test_quantity_period_converted(self, orbit):
        # 24 hours == 1 day
        assert float(orbit.period) == pytest.approx(1.0)

    @given(
        orbit=transit_orbits(
            period=st.just(pytest.importorskip("astropy.units").Quantity(2.0, "d")),
            duration=st.just(
                pytest.importorskip("astropy.units").Quantity(120.0, "min")
            ),
        )
    )
    @settings(max_examples=5)
    def test_quantity_duration_converted(self, orbit):
        # 120 minutes == 0.08333... days
        assert float(orbit.period) == pytest.approx(2.0)
        assert float(orbit.duration) == pytest.approx(120.0 / 1440.0)

    @given(
        orbit=transit_orbits(
            impact_param=st.just(0.5),
            radius_ratio=st.just(0.1),
        )
    )
    @settings(max_examples=5)
    def test_plain_floats_still_work_alongside_quantities(self, orbit):
        assert float(orbit.impact_param) == pytest.approx(0.5)
        assert float(orbit.radius_ratio) == pytest.approx(0.1)
