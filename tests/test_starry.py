from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm

from lightcurve_strategies import bodies, surface_systems, surfaces


class TestSurfacesDefaults:
    """Verify that default strategies produce valid Surface objects."""

    @given(s=surfaces())
    @settings(max_examples=30)
    def test_returns_surface(self, s):
        assert isinstance(s, Surface)


class TestSurfaceSystemsDefaults:
    """Verify that default strategies produce valid SurfaceSystem objects."""

    @given(ss=surface_systems())
    @settings(max_examples=30)
    def test_returns_surface_system(self, ss):
        assert isinstance(ss, SurfaceSystem)

    @given(ss=surface_systems())
    @settings(max_examples=30)
    def test_has_central_surface(self, ss):
        assert isinstance(ss.central_surface, Surface)

    @given(ss=surface_systems())
    @settings(max_examples=30)
    def test_body_count_in_range(self, ss):
        assert 0 <= len(ss.bodies) <= 5

    @given(ss=surface_systems())
    @settings(max_examples=30)
    def test_body_surfaces_length_matches_bodies(self, ss):
        assert len(ss.body_surfaces) == len(ss.bodies)


class TestSurfacesYlm:
    """Verify that the ``y`` parameter is accepted and wired through."""

    @given(
        s=surfaces(
            y=st.just(Ylm.from_dense(np.array([1.0, 0.01, -0.02, 0.03]))),
        )
    )
    @settings(max_examples=5)
    def test_y_parameter_accepted(self, s):
        assert isinstance(s, Surface)
        assert s.ydeg > 0

    @given(s=surfaces())
    @settings(max_examples=5)
    def test_y_none_default(self, s):
        assert isinstance(s, Surface)
        assert s.ydeg == 0


class TestSurfaceSystemsOverrides:
    """Verify that user-supplied strategies override defaults."""

    @given(
        ss=surface_systems(
            body=st.tuples(bodies(period=st.just(5.0)), surfaces()),
            min_bodies=1,
            max_bodies=3,
        )
    )
    @settings(max_examples=10)
    def test_custom_body_surface_strategy(self, ss):
        assert 1 <= len(ss.bodies) <= 3
        for b in ss.bodies:
            assert float(b.period) == 5.0

    @given(ss=surface_systems(min_bodies=2, max_bodies=4))
    @settings(max_examples=10)
    def test_min_max_body_counts(self, ss):
        assert 2 <= len(ss.bodies) <= 4
