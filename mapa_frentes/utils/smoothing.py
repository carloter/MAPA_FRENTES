"""Wrappers para suavizado: gaussiano y n-pass 5-point (Sansom & Catto 2024)."""

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
        mean_val = np.nanmean(result)
        result[np.isnan(result)] = mean_val
    return gaussian_filter(result, sigma=sigma)


def smooth_field_npass(data: np.ndarray, n_passes: int) -> np.ndarray:
    """Suavizado iterativo con estencil de 5 puntos (Sansom & Catto 2024).

    Aplica n_passes del promedio de 5 puntos (cruz):
        F_new[j,i] = (F[j,i] + F[j-1,i] + F[j+1,i] + F[j,i-1] + F[j,i+1]) / 5

    En los bordes, np.pad(mode='edge') replica el valor del borde, lo que
    equivale a un estencil reducido (3 puntos en aristas, 2 en esquinas).

    Para ERA-Interim (0.75 deg): n=8.  Para ERA5/IFS (0.25 deg): n=96.
    Ref: Sansom & Catto (2024), Sect. 3.2; Berry et al. (2011b).

    Args:
        data: Array 2D (nlat, nlon).
        n_passes: Numero de pasadas del filtro.

    Returns:
        Array 2D suavizado.
    """
    result = data.copy()
    if np.any(np.isnan(result)):
        mean_val = np.nanmean(result)
        result[np.isnan(result)] = mean_val

    for _ in range(n_passes):
        padded = np.pad(result, 1, mode='edge')
        result = (
            padded[1:-1, 1:-1]    # centro
            + padded[0:-2, 1:-1]  # norte (j-1)
            + padded[2:,   1:-1]  # sur   (j+1)
            + padded[1:-1, 0:-2]  # oeste (i-1)
            + padded[1:-1, 2:]    # este  (i+1)
        ) / 5.0

    return result
