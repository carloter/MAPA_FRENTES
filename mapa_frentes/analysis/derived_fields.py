"""Campos meteorologicos derivados del IFS para fondo de mapa.

Calcula campos utiles para el trazado manual/hibrido de frentes:
- Theta-e 850 hPa (temperatura potencial equivalente)
- Gradiente de Theta-e 850
- Espesor 1000-500 hPa (thickness)
- Adveccion de temperatura 850 hPa
- Velocidad del viento 850 hPa
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from mapa_frentes.utils.geo import spherical_gradient
from mapa_frentes.utils.smoothing import smooth_field


# Constantes termodinamicas
RD = 287.05      # J/(kg·K) - constante gas seco
CP = 1004.0      # J/(kg·K) - calor especifico a presion constante
LV = 2.501e6     # J/kg     - calor latente de vaporizacion
G = 9.80665      # m/s²     - gravedad


@dataclass
class DerivedField:
    """Resultado de un campo derivado."""
    data: np.ndarray     # Array 2D con los valores
    label: str           # Nombre para mostrar
    units: str           # Unidades
    cmap: str            # Colormap matplotlib
    center_zero: bool = False  # Si centrar el colormap en 0 (divergente)


# Registro de campos disponibles
AVAILABLE_FIELDS = {
    "theta_e_850": "θe 850 hPa",
    "grad_theta_e_850": "|∇θe| 850 hPa",
    "thickness_1000_500": "Espesor 1000-500",
    "temp_advection_850": "Adv. T 850 hPa",
    "wind_speed_850": "Viento 850 hPa",
}


def compute_derived_field(
    ds: xr.Dataset,
    field_name: str,
    lats: np.ndarray,
    lons: np.ndarray,
) -> DerivedField | None:
    """Calcula un campo derivado del dataset IFS.

    Args:
        ds: Dataset con variables IFS (t850, q850, u850, v850, etc.)
        field_name: Clave del campo (ver AVAILABLE_FIELDS)
        lats: Array 1D de latitudes
        lons: Array 1D de longitudes

    Returns:
        DerivedField con datos, label, units, cmap. None si faltan datos.
    """
    compute_funcs = {
        "theta_e_850": _compute_theta_e_850,
        "grad_theta_e_850": _compute_grad_theta_e_850,
        "thickness_1000_500": _compute_thickness,
        "temp_advection_850": _compute_temp_advection_850,
        "wind_speed_850": _compute_wind_speed_850,
    }

    func = compute_funcs.get(field_name)
    if func is None:
        return None

    return func(ds, lats, lons)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Elimina dimensiones extra (tiempo, etc.)."""
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _theta_e(t_k: np.ndarray, q: np.ndarray, p_hpa: float) -> np.ndarray:
    """Calcula temperatura potencial equivalente.

    θe = θ * exp(Lv * r / (cp * T))
    θ  = T * (1000 / p)^(Rd/cp)
    r  = q / (1 - q)

    Args:
        t_k: Temperatura en Kelvin (2D)
        q: Humedad especifica en kg/kg (2D)
        p_hpa: Presion en hPa

    Returns:
        θe en Kelvin (2D)
    """
    # Temperatura potencial
    theta = t_k * (1000.0 / p_hpa) ** (RD / CP)
    # Mixing ratio
    r = np.clip(q, 0, None) / np.clip(1.0 - q, 1e-10, None)
    # Theta-e
    theta_e = theta * np.exp(LV * r / (CP * t_k))
    return theta_e


def _compute_theta_e_850(
    ds: xr.Dataset, lats: np.ndarray, lons: np.ndarray,
) -> DerivedField | None:
    """θe a 850 hPa en grados Celsius."""
    if "t850" not in ds or "q850" not in ds:
        return None

    t850 = _ensure_2d(ds["t850"].values)
    q850 = _ensure_2d(ds["q850"].values)
    te = _theta_e(t850, q850, 850.0)
    te_c = te - 273.15  # a Celsius

    return DerivedField(
        data=smooth_field(te_c, sigma=1.5),
        label="θe 850 hPa",
        units="°C",
        cmap="RdYlBu_r",
    )


def _compute_grad_theta_e_850(
    ds: xr.Dataset, lats: np.ndarray, lons: np.ndarray,
) -> DerivedField | None:
    """Magnitud del gradiente de θe a 850 hPa (K/100km)."""
    if "t850" not in ds or "q850" not in ds:
        return None

    t850 = _ensure_2d(ds["t850"].values)
    q850 = _ensure_2d(ds["q850"].values)
    te = _theta_e(t850, q850, 850.0)
    te_smooth = smooth_field(te, sigma=2.0)

    gx, gy = spherical_gradient(te_smooth, lats, lons)
    grad_mag = np.sqrt(gx**2 + gy**2)
    # Convertir de K/m a K/100km
    grad_mag_100km = grad_mag * 1e5

    return DerivedField(
        data=grad_mag_100km,
        label="|∇θe| 850 hPa",
        units="K/100km",
        cmap="YlOrRd",
    )


def _compute_thickness(
    ds: xr.Dataset, lats: np.ndarray, lons: np.ndarray,
) -> DerivedField | None:
    """Espesor 1000-500 hPa (dam) estimado desde T media.

    Usa ecuacion hipsometrica: ΔZ = (Rd/g) * Tv_media * ln(p1/p2)
    Con Tv ≈ T * (1 + 0.608*q) y T_media = promedio de t850 y t500.
    """
    if "t850" not in ds or "t500" not in ds:
        return None

    t850 = _ensure_2d(ds["t850"].values)
    t500 = _ensure_2d(ds["t500"].values)

    # Temperatura media de la capa
    t_mean = (t850 + t500) / 2.0

    # Correccion por humedad si disponible
    if "q850" in ds and "q500" in ds:
        q850 = _ensure_2d(ds["q850"].values)
        q500 = _ensure_2d(ds["q500"].values)
        q_mean = (q850 + q500) / 2.0
        tv_mean = t_mean * (1.0 + 0.608 * np.clip(q_mean, 0, None))
    else:
        tv_mean = t_mean

    # Espesor en metros: ΔZ = (Rd/g) * Tv * ln(1000/500)
    thickness_m = (RD / G) * tv_mean * np.log(1000.0 / 500.0)
    # Convertir a decametros
    thickness_dam = thickness_m / 10.0

    return DerivedField(
        data=smooth_field(thickness_dam, sigma=1.5),
        label="Espesor 1000-500 hPa",
        units="dam",
        cmap="RdYlBu_r",
    )


def _compute_temp_advection_850(
    ds: xr.Dataset, lats: np.ndarray, lons: np.ndarray,
) -> DerivedField | None:
    """Adveccion de temperatura a 850 hPa (K/3h).

    ADV = -(u * dT/dx + v * dT/dy)
    Positivo = adveccion calida, Negativo = adveccion fria.
    """
    if "t850" not in ds or "u850" not in ds or "v850" not in ds:
        return None

    t850 = _ensure_2d(ds["t850"].values)
    u850 = _ensure_2d(ds["u850"].values)
    v850 = _ensure_2d(ds["v850"].values)

    t_smooth = smooth_field(t850, sigma=2.0)
    gx, gy = spherical_gradient(t_smooth, lats, lons)

    # Adveccion: -V·∇T
    adv = -(u850 * gx + v850 * gy)
    # Convertir de K/s a K/3h
    adv_3h = adv * 3600.0 * 3.0

    return DerivedField(
        data=smooth_field(adv_3h, sigma=1.5),
        label="Adv. T 850 hPa",
        units="K/3h",
        cmap="bwr",
        center_zero=True,
    )


def _compute_wind_speed_850(
    ds: xr.Dataset, lats: np.ndarray, lons: np.ndarray,
) -> DerivedField | None:
    """Velocidad del viento a 850 hPa (kt)."""
    if "u850" not in ds or "v850" not in ds:
        return None

    u850 = _ensure_2d(ds["u850"].values)
    v850 = _ensure_2d(ds["v850"].values)

    speed_ms = np.sqrt(u850**2 + v850**2)
    speed_kt = speed_ms * 1.94384  # m/s a kt

    return DerivedField(
        data=speed_kt,
        label="Viento 850 hPa",
        units="kt",
        cmap="YlGnBu",
    )
