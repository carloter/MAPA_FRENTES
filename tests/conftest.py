"""Fixtures compartidas para tests."""

import numpy as np
import pytest

from mapa_frentes.config import load_config, AppConfig
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType


@pytest.fixture
def cfg():
    """Configuracion por defecto."""
    return AppConfig()


@pytest.fixture
def sample_front():
    """Un frente de ejemplo con 5 puntos."""
    return Front(
        front_type=FrontType.COLD,
        lats=np.array([40.0, 41.0, 42.0, 43.0, 44.0]),
        lons=np.array([-10.0, -9.0, -8.0, -7.0, -6.0]),
        id="test_cold_01",
    )


@pytest.fixture
def sample_collection(sample_front):
    """Coleccion con un frente frio y uno calido."""
    warm = Front(
        front_type=FrontType.WARM,
        lats=np.array([38.0, 39.0, 40.0, 41.0]),
        lons=np.array([-5.0, -4.0, -3.0, -2.0]),
        id="test_warm_01",
    )
    collection = FrontCollection(
        valid_time="2025-05-01T00:00",
        model_run="2025050100",
        description="Test collection",
    )
    collection.add(sample_front)
    collection.add(warm)
    return collection


@pytest.fixture
def synthetic_theta_w():
    """Campo sintetico de theta_w con un frente zonal claro.

    Gradiente fuerte de sur a norte para simular un frente.
    """
    lats = np.linspace(25, 65, 161)
    lons = np.linspace(-60, 30, 361)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Campo base: gradiente meridional
    theta_w = 300 - 0.5 * (lat2d - 45)

    # Anadir un frente fuerte alrededor de lat=45
    theta_w += 5.0 * np.tanh((lat2d - 45.0) / 2.0)

    return theta_w, lats, lons
