from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxoplanet.starry.orbit import SurfaceSystem

from lightcurve_strategies import (
    LightCurveData,
    bodies,
    centrals,
    combined_noise,
    light_curves,
    matern32_kernel,
    red_noise,
    sq_exp_kernel,
    surface_systems,
    surfaces,
    white_noise,
)

TIME = np.linspace(-0.2, 0.2, 50)


# ---- helpers for deterministic systems ------------------------------------

_SYSTEM = surface_systems(
    central=centrals(mass=st.just(1.0), radius=st.just(1.0)),
    central_surface=surfaces(u=st.just((0.1, 0.3))),
    body=st.tuples(
        bodies(
            period=st.just(3.0),
            radius=st.just(0.1),
            time_transit=st.just(0.0),
            impact_param=st.just(0.3),
        ),
        surfaces(),
    ),
    min_bodies=1,
    max_bodies=1,
)


# ===========================================================================
# Defaults (no noise)
# ===========================================================================


class TestLightCurvesDefaults:
    @given(data=light_curves(time=TIME, system=_SYSTEM))
    @settings(max_examples=5)
    def test_returns_light_curve_data(self, data):
        assert isinstance(data, LightCurveData)

    @given(data=light_curves(time=TIME, system=_SYSTEM))
    @settings(max_examples=5)
    def test_correct_shape(self, data):
        assert data.flux.shape == (len(TIME),)
        assert data.flux_with_noise.shape == (len(TIME),)
        assert data.noise.shape == (len(TIME),)

    @given(data=light_curves(time=TIME, system=_SYSTEM))
    @settings(max_examples=5)
    def test_no_noise_by_default(self, data):
        np.testing.assert_array_equal(data.noise, 0.0)
        np.testing.assert_array_equal(data.flux_with_noise, data.flux)

    @given(data=light_curves(time=TIME, system=_SYSTEM))
    @settings(max_examples=5)
    def test_system_is_surface_system(self, data):
        assert isinstance(data.system, SurfaceSystem)


# ===========================================================================
# White noise
# ===========================================================================


class TestLightCurvesWhiteNoise:
    @given(
        data=light_curves(
            time=TIME,
            system=_SYSTEM,
            noise=st.just(white_noise(1e-3)),
        )
    )
    @settings(max_examples=5)
    def test_nonzero_noise(self, data):
        assert not np.allclose(data.noise, 0.0)

    @given(
        data=light_curves(
            time=TIME,
            system=_SYSTEM,
            noise=st.just(white_noise(1e-3)),
        )
    )
    @settings(max_examples=5)
    def test_flux_plus_noise(self, data):
        np.testing.assert_allclose(
            data.flux_with_noise, data.flux + data.noise, atol=1e-15
        )

    @given(
        data=light_curves(
            time=np.linspace(0, 10, 5000),
            system=_SYSTEM,
            noise=st.just(white_noise(0.01)),
        )
    )
    @settings(max_examples=3)
    def test_noise_std_close_to_scale(self, data):
        assert abs(np.std(data.noise) - 0.01) < 0.003


# ===========================================================================
# Red noise
# ===========================================================================


class TestLightCurvesRedNoise:
    @given(
        data=light_curves(
            time=np.linspace(0, 10, 200),
            system=_SYSTEM,
            noise=st.just(red_noise(sq_exp_kernel(1e-3, 1.0), jitter=1e-10)),
        )
    )
    @settings(max_examples=3)
    def test_correct_shape(self, data):
        assert data.noise.shape == (200,)

    @given(
        data=light_curves(
            time=np.linspace(0, 10, 200),
            system=_SYSTEM,
            noise=st.just(red_noise(sq_exp_kernel(1e-3, 1.0), jitter=1e-10)),
        )
    )
    @settings(max_examples=3)
    def test_correlated(self, data):
        # Lag-1 autocorrelation should be positive for correlated noise
        n = data.noise
        autocorr = np.corrcoef(n[:-1], n[1:])[0, 1]
        assert autocorr > 0.0


# ===========================================================================
# Combined noise
# ===========================================================================


class TestLightCurvesCombinedNoise:
    @given(
        data_white=light_curves(
            time=np.linspace(0, 10, 1000),
            system=_SYSTEM,
            noise=st.just(white_noise(1e-3)),
            seed=st.just(42),
        ),
        data_combined=light_curves(
            time=np.linspace(0, 10, 1000),
            system=_SYSTEM,
            noise=st.just(
                combined_noise(
                    white_noise(1e-3),
                    white_noise(1e-3),
                )
            ),
            seed=st.just(42),
        ),
    )
    @settings(max_examples=3)
    def test_combined_larger_std(self, data_white, data_combined):
        assert np.std(data_combined.noise) > np.std(data_white.noise)


# ===========================================================================
# Determinism
# ===========================================================================


class TestLightCurvesDeterminism:
    @given(
        data1=light_curves(
            time=TIME,
            system=_SYSTEM,
            noise=st.just(white_noise(1e-3)),
            seed=st.just(12345),
        ),
        data2=light_curves(
            time=TIME,
            system=_SYSTEM,
            noise=st.just(white_noise(1e-3)),
            seed=st.just(12345),
        ),
    )
    @settings(max_examples=3)
    def test_same_seed_same_noise(self, data1, data2):
        np.testing.assert_array_equal(data1.noise, data2.noise)


# ===========================================================================
# System override
# ===========================================================================


class TestLightCurvesSystemOverride:
    @given(
        data=light_curves(
            time=TIME,
            system=surface_systems(
                central=centrals(mass=st.just(2.0), radius=st.just(2.0)),
                min_bodies=0,
                max_bodies=0,
            ),
        )
    )
    @settings(max_examples=3)
    def test_custom_system(self, data):
        assert float(data.system.central.mass) == 2.0
        assert float(data.system.central.radius) == 2.0
        assert len(data.system.bodies) == 0


# ===========================================================================
# Kernel functions
# ===========================================================================


class TestKernelFunctions:
    def test_sq_exp_at_zero(self):
        k = sq_exp_kernel(amplitude=2.0, length_scale=1.0)
        dt = np.zeros((3, 3))
        result = k(dt)
        np.testing.assert_allclose(result, 4.0)

    def test_sq_exp_symmetric(self):
        k = sq_exp_kernel(amplitude=1.0, length_scale=1.0)
        t = np.linspace(0, 5, 20)
        dt = t[:, None] - t[None, :]
        cov = k(dt)
        np.testing.assert_allclose(cov, cov.T, atol=1e-15)

    def test_sq_exp_psd(self):
        k = sq_exp_kernel(amplitude=1.0, length_scale=1.0)
        t = np.linspace(0, 5, 20)
        dt = t[:, None] - t[None, :]
        cov = k(dt)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)

    def test_matern32_at_zero(self):
        k = matern32_kernel(amplitude=3.0, length_scale=1.0)
        dt = np.zeros((3, 3))
        result = k(dt)
        np.testing.assert_allclose(result, 9.0)

    def test_matern32_symmetric(self):
        k = matern32_kernel(amplitude=1.0, length_scale=2.0)
        t = np.linspace(0, 5, 20)
        dt = t[:, None] - t[None, :]
        cov = k(dt)
        np.testing.assert_allclose(cov, cov.T, atol=1e-15)

    def test_matern32_psd(self):
        k = matern32_kernel(amplitude=1.0, length_scale=2.0)
        t = np.linspace(0, 5, 20)
        dt = t[:, None] - t[None, :]
        cov = k(dt)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)
