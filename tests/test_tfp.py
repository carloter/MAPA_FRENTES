"""Tests para fronts/tfp.py y utils asociados."""

import numpy as np
import pytest

from mapa_frentes.utils.geo import haversine, grid_spacing, spherical_gradient
from mapa_frentes.utils.smoothing import smooth_field


class TestHaversine:
    def test_same_point(self):
        d = haversine(40.0, -8.0, 40.0, -8.0)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        # Santiago de Compostela a Madrid: ~480-520 km
        d = haversine(42.88, -8.54, 40.42, -3.70)
        assert 470e3 < d < 530e3

    def test_symmetric(self):
        d1 = haversine(40.0, -8.0, 50.0, 0.0)
        d2 = haversine(50.0, 0.0, 40.0, -8.0)
        assert d1 == pytest.approx(d2, rel=1e-10)


class TestGridSpacing:
    def test_dx_varies_with_latitude(self):
        lats = np.array([0.0, 30.0, 60.0])
        lons = np.array([0.0, 1.0, 2.0])
        dx, dy = grid_spacing(lats, lons)
        # dx en el ecuador debe ser mayor que a 60N
        assert dx[0, 0] > dx[2, 0]

    def test_dy_is_constant(self):
        lats = np.linspace(30, 60, 121)
        lons = np.linspace(-60, 30, 361)
        dx, dy = grid_spacing(lats, lons)
        # dy no depende de la longitud
        assert dy[0, 0] == pytest.approx(dy[0, -1], rel=1e-6)
        # dy no depende mucho de la latitud (es constante)
        assert dy[0, 0] == pytest.approx(dy[-1, 0], rel=1e-6)


class TestSphericalGradient:
    def test_uniform_field_zero_gradient(self):
        lats = np.linspace(30, 60, 50)
        lons = np.linspace(-20, 20, 80)
        field = np.ones((50, 80)) * 300.0
        gx, gy = spherical_gradient(field, lats, lons)
        assert np.allclose(gx, 0, atol=1e-15)
        assert np.allclose(gy, 0, atol=1e-15)

    def test_meridional_gradient_direction(self):
        lats = np.linspace(30, 60, 50)
        lons = np.linspace(-20, 20, 80)
        lon2d, lat2d = np.meshgrid(lons, lats)
        # Campo que aumenta hacia el norte
        field = lat2d * 1.0
        gx, gy = spherical_gradient(field, lats, lons)
        # El gradiente zonal deberia ser ~0
        assert np.allclose(gx, 0, atol=1e-10)
        # El gradiente meridional deberia ser positivo (N es positivo)
        assert np.all(gy[1:-1, :] > 0)  # ignorar bordes


class TestSmoothField:
    def test_smooth_preserves_mean(self):
        data = np.random.randn(50, 80)
        smoothed = smooth_field(data, sigma=2.0)
        # El suavizado gaussiano preserva la media (aprox)
        assert np.mean(data) == pytest.approx(np.mean(smoothed), abs=0.1)

    def test_smooth_reduces_variance(self):
        data = np.random.randn(50, 80)
        smoothed = smooth_field(data, sigma=3.0)
        assert np.var(smoothed) < np.var(data)

    def test_smooth_handles_nan(self):
        data = np.ones((20, 20))
        data[5, 5] = np.nan
        smoothed = smooth_field(data, sigma=1.0)
        assert not np.any(np.isnan(smoothed))


def _compute_tfp_field(theta_w, lats, lons, sigma):
    """Version local de compute_tfp_field sin dependencia de xarray/metpy."""
    theta_smooth = smooth_field(theta_w, sigma=sigma)
    gx, gy = spherical_gradient(theta_smooth, lats, lons)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag_safe = np.where(grad_mag > 1e-12, grad_mag, 1e-12)
    ux = gx / grad_mag_safe
    uy = gy / grad_mag_safe
    gmag_x, gmag_y = spherical_gradient(grad_mag, lats, lons)
    tfp = -(gmag_x * ux + gmag_y * uy)
    return tfp, grad_mag


class TestTFPField:
    def test_front_detection_synthetic(self, synthetic_theta_w):
        """Verifica que el TFP detecta un frente en un campo sintetico."""
        theta_w, lats, lons = synthetic_theta_w
        tfp, grad_mag = _compute_tfp_field(theta_w, lats, lons, sigma=4.0)

        # El TFP deberia tener cambios de signo cerca de lat=45
        assert tfp.shape == theta_w.shape
        assert grad_mag.shape == theta_w.shape

        # Debe haber zero-crossings (cambios de signo)
        sign_changes = np.abs(np.diff(np.sign(tfp), axis=0))
        assert np.sum(sign_changes > 0) > 0

    def test_gradient_magnitude_positive(self, synthetic_theta_w):
        theta_w, lats, lons = synthetic_theta_w
        _, grad_mag = _compute_tfp_field(theta_w, lats, lons, sigma=4.0)
        assert np.all(grad_mag >= 0)
