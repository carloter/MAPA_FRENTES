"""Procesamiento de isobaras: suavizado MSLP y niveles."""

import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr

from mapa_frentes.config import AppConfig


def smooth_mslp(msl: xr.DataArray, sigma: float) -> np.ndarray:
    """Aplica suavizado gaussiano a la presion a nivel del mar.

    Args:
        msl: Presion a nivel del mar en hPa.
        sigma: Desviacion estandar del filtro gaussiano (en puntos de grid).

    Returns:
        Array 2D suavizado.
    """
    data = msl.values.copy()
    # Rellenar NaN con interpolacion antes de suavizar
    if np.any(np.isnan(data)):
        from scipy.interpolate import griddata
        mask = np.isnan(data)
        y, x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        data[mask] = griddata(
            (y[~mask], x[~mask]), data[~mask], (y[mask], x[mask]),
            method="nearest"
        )
    return gaussian_filter(data, sigma=sigma)


def compute_isobar_levels(msl_smooth: np.ndarray, interval: int = 4) -> np.ndarray:
    """Calcula los niveles de isobaras (multiplos del intervalo).

    Args:
        msl_smooth: Array 2D de MSLP suavizada en hPa.
        interval: Intervalo entre isobaras en hPa.

    Returns:
        Array de niveles de isobaras.
    """
    vmin = np.floor(msl_smooth.min() / interval) * interval
    vmax = np.ceil(msl_smooth.max() / interval) * interval
    return np.arange(vmin, vmax + interval, interval)
