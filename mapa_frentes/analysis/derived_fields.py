"""Campos meteorologicos derivados del IFS para fondo de mapa.

Calcula campos utiles para el trazado manual/hibrido de frentes:
- Theta-e a 850/700 hPa (temperatura potencial equivalente)
- Gradientes de Theta-e y T a 850 hPa
- Espesor 1000-500 hPa (thickness)
- Adveccion de temperatura 850 hPa
- Velocidad del viento 850/500 hPa
- Vorticidad, temperatura y humedad a 850 hPa
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


# Registro de campos disponibles (clave → etiqueta para UI)
AVAILABLE_FIELDS = {
    "none": "Sin fondo",
    "theta_e_850": "θe 850 hPa",
    "theta_e_700": "θe 700 hPa",
    "grad_theta_e_850": "|∇θe| 850 hPa",
    "grad_t_850": "|∇T| 850 hPa",
    "thickness_1000_500": "Espesor 1000-500",
    "temp_advection_850": "Adv. T 850 hPa",
    "wind_speed_850": "Viento 850 hPa",
    "wind_speed_500": "Viento 500 hPa",
    "vorticity_850": "Vorticidad 850 hPa",
    "temp_850": "T 850 hPa",
    "humidity_850": "q 850 hPa",
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
    if field_name == "none":
        return None

    compute_funcs = {
        "theta_e_850": _compute_theta_e_850,
        "theta_e_700": _compute_theta_e_700,
        "grad_theta_e_850": _compute_grad_theta_e_850,
        "grad_t_850": _compute_grad_t_850,
        "thickness_1000_500": _compute_thickness,
        "temp_advection_850": _compute_temp_advection_850,
        "wind_speed_850": _compute_wind_speed_850,
        "wind_speed_500": _compute_wind_speed_500,
        "vorticity_850": _compute_vorticity_850,
        "temp_850": _compute_temp_850,
        "humidity_850": _compute_humidity_850,
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
    """
    theta = t_k * (1000.0 / p_hpa) ** (RD / CP)
    r = np.clip(q, 0, None) / np.clip(1.0 - q, 1e-10, None)
    theta_e = theta * np.exp(LV * r / (CP * t_k))
    return theta_e


# --- Funciones de calculo individuales ---

def _compute_theta_e_850(ds, lats, lons):
    if "t850" not in ds or "q850" not in ds:
        return None
    t850 = _ensure_2d(ds["t850"].values)
    q850 = _ensure_2d(ds["q850"].values)
    te_c = _theta_e(t850, q850, 850.0) - 273.15
    return DerivedField(
        data=smooth_field(te_c, sigma=1.5),
        label="θe 850 hPa", units="°C", cmap="RdYlBu_r",
    )


def _compute_theta_e_700(ds, lats, lons):
    if "t700" not in ds or "q700" not in ds:
        return None
    t700 = _ensure_2d(ds["t700"].values)
    q700 = _ensure_2d(ds["q700"].values)
    te_c = _theta_e(t700, q700, 700.0) - 273.15
    return DerivedField(
        data=smooth_field(te_c, sigma=1.5),
        label="θe 700 hPa", units="°C", cmap="RdYlBu_r",
    )


def _compute_grad_theta_e_850(ds, lats, lons):
    if "t850" not in ds or "q850" not in ds:
        return None
    t850 = _ensure_2d(ds["t850"].values)
    q850 = _ensure_2d(ds["q850"].values)
    te = _theta_e(t850, q850, 850.0)
    te_smooth = smooth_field(te, sigma=2.0)
    gx, gy = spherical_gradient(te_smooth, lats, lons)
    grad_mag = np.sqrt(gx**2 + gy**2) * 1e5  # K/m → K/100km
    return DerivedField(
        data=grad_mag,
        label="|∇θe| 850 hPa", units="K/100km", cmap="YlOrRd",
    )


def _compute_grad_t_850(ds, lats, lons):
    if "t850" not in ds:
        return None
    t850 = _ensure_2d(ds["t850"].values)
    t_smooth = smooth_field(t850, sigma=2.0)
    gx, gy = spherical_gradient(t_smooth, lats, lons)
    grad_mag = np.sqrt(gx**2 + gy**2) * 1e5  # K/m → K/100km
    return DerivedField(
        data=grad_mag,
        label="|∇T| 850 hPa", units="K/100km", cmap="YlOrRd",
    )


def _compute_thickness(ds, lats, lons):
    if "t850" not in ds or "t500" not in ds:
        return None
    t850 = _ensure_2d(ds["t850"].values)
    t500 = _ensure_2d(ds["t500"].values)
    t_mean = (t850 + t500) / 2.0
    if "q850" in ds and "q500" in ds:
        q850 = _ensure_2d(ds["q850"].values)
        q500 = _ensure_2d(ds["q500"].values)
        q_mean = (q850 + q500) / 2.0
        tv_mean = t_mean * (1.0 + 0.608 * np.clip(q_mean, 0, None))
    else:
        tv_mean = t_mean
    thickness_dam = (RD / G) * tv_mean * np.log(1000.0 / 500.0) / 10.0
    return DerivedField(
        data=smooth_field(thickness_dam, sigma=1.5),
        label="Espesor 1000-500 hPa", units="dam", cmap="RdYlBu_r",
    )


def _compute_temp_advection_850(ds, lats, lons):
    if "t850" not in ds or "u850" not in ds or "v850" not in ds:
        return None
    t850 = _ensure_2d(ds["t850"].values)
    u850 = _ensure_2d(ds["u850"].values)
    v850 = _ensure_2d(ds["v850"].values)
    t_smooth = smooth_field(t850, sigma=2.0)
    gx, gy = spherical_gradient(t_smooth, lats, lons)
    adv_3h = -(u850 * gx + v850 * gy) * 3600.0 * 3.0
    return DerivedField(
        data=smooth_field(adv_3h, sigma=1.5),
        label="Adv. T 850 hPa", units="K/3h", cmap="bwr", center_zero=True,
    )


def _compute_wind_speed_850(ds, lats, lons):
    if "u850" not in ds or "v850" not in ds:
        return None
    u850 = _ensure_2d(ds["u850"].values)
    v850 = _ensure_2d(ds["v850"].values)
    speed_kt = np.sqrt(u850**2 + v850**2) * 1.94384
    return DerivedField(
        data=speed_kt, label="Viento 850 hPa", units="kt", cmap="YlGnBu",
    )


def _compute_wind_speed_500(ds, lats, lons):
    if "u500" not in ds or "v500" not in ds:
        return None
    u500 = _ensure_2d(ds["u500"].values)
    v500 = _ensure_2d(ds["v500"].values)
    speed_kt = np.sqrt(u500**2 + v500**2) * 1.94384
    return DerivedField(
        data=speed_kt, label="Viento 500 hPa", units="kt", cmap="YlGnBu",
    )


def _compute_vorticity_850(ds, lats, lons):
    if "vo850" not in ds:
        return None
    vo = _ensure_2d(ds["vo850"].values)
    vo_scaled = vo * 1e5  # s⁻¹ → 10⁻⁵ s⁻¹
    return DerivedField(
        data=smooth_field(vo_scaled, sigma=1.5),
        label="Vorticidad 850 hPa", units="10⁻⁵ s⁻¹", cmap="PiYG_r",
        center_zero=True,
    )


def _compute_temp_850(ds, lats, lons):
    if "t850" not in ds:
        return None
    t850 = _ensure_2d(ds["t850"].values)
    t_c = t850 - 273.15
    return DerivedField(
        data=t_c, label="T 850 hPa", units="°C", cmap="RdYlBu_r",
    )


def _compute_humidity_850(ds, lats, lons):
    if "q850" not in ds:
        return None
    q850 = _ensure_2d(ds["q850"].values)
    q_gkg = q850 * 1000.0  # kg/kg → g/kg
    return DerivedField(
        data=q_gkg, label="q 850 hPa", units="g/kg", cmap="YlGnBu",
    )


def compute_precipitation(ds: xr.Dataset) -> np.ndarray | None:
    """Calcula precipitacion en mm desde el dataset.

    Returns:
        Array 2D de precipitacion en mm, o None si no hay datos tp.
    """
    if "tp" not in ds:
        return None
    tp = _ensure_2d(ds["tp"].values)
    tp_mm = tp * 1000.0  # ECMWF tp viene en metros de agua
    tp_mm = np.clip(tp_mm, 0, None)
    return smooth_field(tp_mm, sigma=1.0)
