"""Wrappers para suavizado gaussiano."""

import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_field(data: np.ndarray, sigma: float) -> np.ndarray:
    """Aplica suavizado gaussiano a un campo 2D.

    Maneja NaN rellenandolos antes de suavizar.

    Args:
        data: Array 2D.
        sigma: Desviacion estandar del filtro en puntos de grid.

    Returns:
        Array 2D suavizado.
    """
    result = data.copy()
    if np.any(np.isnan(result)):
        # Rellenar NaN con valor medio
        mean_val = np.nanmean(result)
        result[np.isnan(result)] = mean_val
    return gaussian_filter(result, sigma=sigma)
