"""
Algoritmo TFP (Thermal Front Parameter) para deteccion de frentes.

Implementación basada en Hewson (1998) con el cambio "contour-then-mask"
(Sansom & Catto 2024):

1) Calcular theta_w (850 hPa)
2) Suavizar
3) Calcular grad(theta_w) y |grad|
4) Calcular:
   - TFL = ∇² |∇theta_w|   (front locator; frente = contorno TFL=0)
   - TFP = ∇|∇theta_w| · (∇theta_w / |∇theta_w|)   (criterio K1 <= 0)
   - |∇theta_w|_ABZ = |∇theta_w| + m*chi*|∇|∇theta_w|| (criterio K2 >= 0)
5) Extraer contornos TFL=0 (find_contours)
6) Enmascarar puntos del contorno con:
   - TFP <= K1
   - ABZ >= K2
   - (opc) F diagnostic
   - (opc) zona ciclonica (laplaciana MSLP > umbral)
7) Segmentar, suavizar spline, filtrar longitud minima
"""

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr
from metpy.calc import dewpoint_from_specific_humidity, frontogenesis, wet_bulb_potential_temperature
from metpy.units import units
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, laplace  # laplace: solo para MSLP legacy

from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.connector import cluster_and_connect, _smooth_spline, _polyline_length_deg
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType
from mapa_frentes.utils.geo import spherical_gradient, spherical_laplacian
from mapa_frentes.utils.smoothing import smooth_field, smooth_field_npass

logger = logging.getLogger(__name__)

# Constantes Parfitt et al. (2017) para diagnostico F (tu añadido)
OMEGA = 7.2921e-5
GRAD_SCALE = 4.5e-6
MIN_CORIOLIS_LAT = 10.0

# H98 ABZ: m = 1/sqrt(2)
ABZ_M = 1.0 / np.sqrt(2.0)

EARTH_RADIUS_M = 6.371e6


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _remove_time_dim(ds: xr.Dataset) -> xr.Dataset:
    for time_dim in ["time", "valid_time", "step", "verify_time"]:
        if time_dim in ds.dims:
            ds = ds.isel({time_dim: 0})
    return ds


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim > 2:
        arr = arr[0]
    return arr


def _grid_length_m(lats_1d: np.ndarray, lons_1d: np.ndarray) -> float:
    """
    Aproxima la longitud de malla χ (H98 Eq.3) en metros para una rejilla lat-lon regular.
    Usamos un valor representativo (mediana) con cos(lat) en la latitud media del dominio.
    """
    if len(lats_1d) < 2 or len(lons_1d) < 2:
        return 0.0
    dlat_deg = float(np.median(np.abs(np.diff(lats_1d))))
    dlon_deg = float(np.median(np.abs(np.diff(lons_1d))))
    lat0 = float(np.median(lats_1d))
    dy = dlat_deg * np.pi / 180.0 * EARTH_RADIUS_M
    dx = dlon_deg * np.pi / 180.0 * EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))
    # “grid length” χ: valor típico; geométrica va bien si dx!=dy
    return float(np.sqrt(max(dx, 1.0) * max(dy, 1.0)))


def _polyline_length_km(lats: np.ndarray, lons: np.ndarray) -> float:
    """Longitud total de una polilínea en km usando haversine (Sansom & Catto 2024: 250 km)."""
    if len(lats) < 2:
        return 0.0
    total = 0.0
    for i in range(len(lats) - 1):
        lat1, lon1 = np.radians(lats[i]), np.radians(lons[i])
        lat2, lon2 = np.radians(lats[i + 1]), np.radians(lons[i + 1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        total += 2 * np.arcsin(np.sqrt(a))
    return float(total * EARTH_RADIUS_M / 1000.0)


def _latband_mask(lats_1d: np.ndarray, lat_min: float, lat_max: float) -> np.ndarray:
    """
    Máscara 2D (ny,nx) True en banda latitudinal para cuantiles "extra-trópicos".
    """
    lat_mask_1d = (lats_1d >= lat_min) & (lats_1d <= lat_max)
    return lat_mask_1d[:, None]


def _finite_vals(arr: np.ndarray, mask2d: np.ndarray | None = None) -> np.ndarray:
    if mask2d is None:
        vals = arr[np.isfinite(arr)]
    else:
        vals = arr[np.isfinite(arr) & mask2d]
    return vals


# -----------------------------------------------------------------------------
# Thermo
# -----------------------------------------------------------------------------

def compute_theta_w(ds: xr.Dataset) -> np.ndarray:
    ds = _remove_time_dim(ds)

    pressure = 850.0 * units.hPa
    t850 = _ensure_2d(ds["t850"].values) * units.kelvin
    q850 = _ensure_2d(ds["q850"].values) * units("kg/kg")

    td = dewpoint_from_specific_humidity(pressure, t850, q850)
    theta_w = wet_bulb_potential_temperature(pressure, t850, td)
    return _ensure_2d(theta_w.magnitude)


# -----------------------------------------------------------------------------
# Core fields: grad, TFL, TFP, ABZ
# -----------------------------------------------------------------------------

def compute_h98_fields(
    theta_w: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    smooth_sigma: float,
    gradmag_smooth_sigma: float | None = None,
    smoothing_method: str = "gaussian",
    smoothing_passes: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve (tfl, tfp, grad_mag, grad_gradmag_mag, gradmag_smooth, gmag_x, gmag_y)

    - grad_mag: |∇θW|
    - TFL: ∇² |∇θW| (H98 Eq.1)  -> contorno 0 = localizador del frente
    - TFP: ∇|∇θW| · (∇θW/|∇θW|) (H98 Eq.2)  -> criterio K1 (<=0)
    - grad_gradmag_mag: |∇|∇θW|| (para ABZ)
    - gmag_x, gmag_y: componentes de ∇|∇θW| (para clasificación Hewson)

    Args:
        smoothing_method: "gaussian" (legacy) o "npass" (Sansom & Catto 2024).
        smoothing_passes: Numero de pasadas para metodo "npass" (96 para 0.25deg).
    """
    theta_w = _ensure_2d(theta_w)

    # 1) suavizar theta_w
    if smoothing_method == "npass":
        logger.info("Suavizado: %d pasadas de 5-point average (Sansom & Catto 2024)", smoothing_passes)
        theta_s = smooth_field_npass(theta_w, n_passes=smoothing_passes)
    else:
        theta_s = smooth_field(theta_w, sigma=smooth_sigma)

    # 2) grad(theta_w) en esfera
    gx, gy = spherical_gradient(theta_s, lats, lons)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag_safe = np.where(grad_mag > 1e-12, grad_mag, 1e-12)

    # 3) unit vector del gradiente
    ux = gx / grad_mag_safe
    uy = gy / grad_mag_safe

    # 4) grad_mag_s: version suavizada de |∇θ| (solo para legacy/estabilizacion)
    # El paper (Sansom & Catto 2024) NO re-suaviza grad_mag: TFL y TFP se calculan
    # directamente sobre |∇θ_W| derivado del campo ya suavizado.
    # Solo re-suavizamos si se pide explicitamente (gradmag_smooth_sigma != None en modo gaussian).
    if smoothing_method == "gaussian" and gradmag_smooth_sigma is not None:
        grad_mag_s = smooth_field(grad_mag, sigma=gradmag_smooth_sigma)
    else:
        grad_mag_s = grad_mag  # paper: sin re-suavizado

    # 5) grad(|grad|)
    gmag_x, gmag_y = spherical_gradient(grad_mag_s, lats, lons)
    grad_gradmag_mag = np.sqrt(gmag_x**2 + gmag_y**2)

    # 6) TFP (H98 Eq.2): ∇|∇θ| · (∇θ/|∇θ|)
    tfp = (gmag_x * ux + gmag_y * uy)

    # 7) TFL (H98 Eq.1): ∇²|∇θ|
    # Laplaciano esferico con cos(lat) (Sansom & Catto 2024, Sect. 3.4)
    tfl = spherical_laplacian(grad_mag_s, lats, lons)

    return tfl, tfp, grad_mag, grad_gradmag_mag, grad_mag_s, gmag_x, gmag_y


def compute_abz_gradient(
    grad_mag: np.ndarray,
    grad_gradmag_mag: np.ndarray,
    chi_m: float,
) -> np.ndarray:
    """
    |∇θ|_ABZ = |∇θ| + m*chi*|∇|∇θ||   (H98 Eq.3)
    """
    return grad_mag + (ABZ_M * chi_m * grad_gradmag_mag)


# -----------------------------------------------------------------------------
# Extra: F diagnostic (tu filtro dinámico)
# -----------------------------------------------------------------------------

def _compute_f_diagnostic(
    grad_mag: np.ndarray,
    vorticity: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    f_coriolis = 2 * OMEGA * np.abs(np.sin(np.radians(lats)))
    f_min = 2 * OMEGA * np.sin(np.radians(MIN_CORIOLIS_LAT))
    f_coriolis = np.maximum(f_coriolis, f_min)
    f_2d = f_coriolis[:, np.newaxis] * np.ones((1, grad_mag.shape[1]))
    return (grad_mag / GRAD_SCALE) * (np.abs(vorticity) / f_2d)


# -----------------------------------------------------------------------------
# Contour-then-mask extractor (paper)
# -----------------------------------------------------------------------------

def _split_by_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    mask: np.ndarray,
    min_points: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start = None

    for i in range(len(mask)):
        if mask[i]:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_points:
                segments.append((lats[start:i].copy(), lons[start:i].copy()))
            start = None

    if start is not None and len(lats) - start >= min_points:
        segments.append((lats[start:].copy(), lons[start:].copy()))

    return segments


def _extract_contour_fronts_h98(
    tfl: np.ndarray,
    tfp: np.ndarray,
    abz: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    cfg: AppConfig,
    vorticity: np.ndarray | None = None,
    msl: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Frentes = contornos TFL=0, luego máscara con:
      - TFP <= K1
      - ABZ >= K2
    + opcional: F diagnostic, filtro MSLP ciclónico.
    """
    from skimage.measure import find_contours

    tfp_cfg = cfg.tfp
    margin = int(tfp_cfg.contour_margin_px)

    # -------------------
    # Umbrales K1/K2 (paper sugiere cuantiles climato; aquí permitimos cuantiles instantáneos en banda lat)
    # -------------------
    if tfp_cfg.use_quantile_thresholds:
        band = _latband_mask(lats, tfp_cfg.quantile_lat_min, tfp_cfg.quantile_lat_max)
        tfp_vals = _finite_vals(tfp, band)
        abz_vals = _finite_vals(abz, band)

        if tfp_vals.size == 0 or abz_vals.size == 0:
            logger.warning("Cuantiles: sin datos en banda lat; usando K1/K2 fijos.")
            K1 = float(tfp_cfg.k1_tfp)
            K2 = float(tfp_cfg.k2_abz)
        else:
            # K1: cuantil bajo (más negativo) típico 0.25 en paper
            K1q = float(np.quantile(tfp_vals, tfp_cfg.k1_tfp_quantile))
            # K2: cuantil medio típico 0.50 en paper
            K2q = float(np.quantile(abz_vals, tfp_cfg.k2_abz_quantile))

            # "cap" para no irnos a umbrales absurdos por un caso raro
            K1 = min(float(tfp_cfg.k1_tfp), K1q) if tfp_cfg.k1_use_cap else K1q
            K2 = max(float(tfp_cfg.k2_abz), K2q) if tfp_cfg.k2_use_floor else K2q

        logger.info(
            "H98 thresholds: K1(TFP)<= %.3e (q=%.2f), K2(ABZ)>= %.3e (q=%.2f)",
            K1, tfp_cfg.k1_tfp_quantile, K2, tfp_cfg.k2_abz_quantile
        )
    else:
        K1 = float(tfp_cfg.k1_tfp)
        K2 = float(tfp_cfg.k2_abz)
        logger.info("H98 thresholds (fixed): K1=%.3e, K2=%.3e", K1, K2)

    # -------------------
    # Contornos TFL=0  (esto es lo que define el paper/H98)
    # -------------------
    raw_contours = find_contours(tfl, 0.0)
    logger.info("Contornos TFL=0 extraidos: %d", len(raw_contours))

    row_idx = np.arange(len(lats))
    col_idx = np.arange(len(lons))

    interp_tfp = RegularGridInterpolator((row_idx, col_idx), tfp, bounds_error=False, fill_value=np.nan)
    interp_abz = RegularGridInterpolator((row_idx, col_idx), abz, bounds_error=False, fill_value=np.nan)

    # F diagnostic (opcional)
    interp_f = None
    if vorticity is not None and tfp_cfg.use_f_diagnostic:
        f_field = _compute_f_diagnostic(grad_mag=abz, vorticity=vorticity, lats=lats)  # usa ABZ como proxy grad
        interp_f = RegularGridInterpolator((row_idx, col_idx), f_field, bounds_error=False, fill_value=np.nan)

        if tfp_cfg.use_f_quantile_threshold:
            band = _latband_mask(lats, tfp_cfg.quantile_lat_min, tfp_cfg.quantile_lat_max)
            f_vals = _finite_vals(f_field, band)
            if f_vals.size > 0:
                tfp_cfg_fthr = float(np.quantile(f_vals, tfp_cfg.f_quantile))
                f_thr = tfp_cfg_fthr
            else:
                f_thr = float(tfp_cfg.f_diagnostic_threshold)
        else:
            f_thr = float(tfp_cfg.f_diagnostic_threshold)

        logger.info("F diagnostic: threshold=%.3f", f_thr)
    else:
        f_thr = None

    # MSLP laplaciana (opcional)
    interp_msl = None
    if msl is not None and tfp_cfg.use_mslp_filter:
        msl_s = gaussian_filter(msl, sigma=float(tfp_cfg.mslp_laplacian_sigma))
        msl_lapl = laplace(msl_s)
        interp_msl = RegularGridInterpolator((row_idx, col_idx), msl_lapl, bounds_error=False, fill_value=np.nan)

        if tfp_cfg.mslp_use_percentile_cut:
            band = _latband_mask(lats, tfp_cfg.quantile_lat_min, tfp_cfg.quantile_lat_max)
            v = _finite_vals(msl_lapl, band)
            if v.size > 0:
                msl_cut = float(np.quantile(v, tfp_cfg.mslp_laplacian_percentile))
            else:
                msl_cut = float(tfp_cfg.mslp_laplacian_cut)
        else:
            msl_cut = float(tfp_cfg.mslp_laplacian_cut)

        logger.info("MSLP laplacian cut: %.3e", msl_cut)
    else:
        msl_cut = None

    polylines: list[tuple[np.ndarray, np.ndarray]] = []

    for contour in raw_contours:
        if len(contour) < int(tfp_cfg.min_front_points):
            continue

        rows, cols = contour[:, 0], contour[:, 1]

        # margen de borde
        keep = (
            (rows >= margin) & (rows <= len(lats) - margin - 1) &
            (cols >= margin) & (cols <= len(lons) - margin - 1)
        )

        # interpolar variables de máscara al contorno (paper)
        tfp_vals = interp_tfp(contour)
        abz_vals = interp_abz(contour)

        keep &= np.isfinite(tfp_vals) & np.isfinite(abz_vals)
        keep &= (tfp_vals <= K1)          # K1 <= 0
        keep &= (abz_vals >= K2)          # K2 >= 0

        # F (opcional)
        if interp_f is not None and f_thr is not None:
            f_vals = interp_f(contour)
            keep &= np.isfinite(f_vals) & (f_vals >= f_thr)

        # MSLP ciclónica (opcional)
        if interp_msl is not None and msl_cut is not None:
            msl_vals = interp_msl(contour)
            keep &= np.isfinite(msl_vals) & (msl_vals >= msl_cut)

        # pixel -> lat/lon
        c_lats = np.interp(rows, row_idx, lats)
        c_lons = np.interp(cols, col_idx, lons)

        # segmentar
        segments = _split_by_mask(c_lats, c_lons, keep, int(tfp_cfg.min_front_points))

        for seg_lats, seg_lons in segments:
            length_km = _polyline_length_km(seg_lats, seg_lons)
            if length_km < float(tfp_cfg.min_front_length_km):
                continue

            # downsample
            if len(seg_lats) > int(tfp_cfg.max_points_per_front):
                step = max(1, len(seg_lats) // int(tfp_cfg.max_points_per_front))
                seg_lats = seg_lats[::step]
                seg_lons = seg_lons[::step]

            # suavizado
            if len(seg_lats) >= 4 and float(tfp_cfg.spline_smoothing) > 0:
                seg_lats, seg_lons = _smooth_spline(seg_lats, seg_lons, smoothing=float(tfp_cfg.spline_smoothing))

            if len(seg_lats) >= 3:
                polylines.append((seg_lats, seg_lons))

    # ordenar por longitud (km, gran-circulo)
    polylines.sort(key=lambda p: _polyline_length_km(p[0], p[1]), reverse=True)

    # limitar
    if int(tfp_cfg.max_fronts) > 0 and len(polylines) > int(tfp_cfg.max_fronts):
        polylines = polylines[: int(tfp_cfg.max_fronts)]

    logger.info("Frentes por contorno (H98): %d polylines", len(polylines))
    return polylines


# -----------------------------------------------------------------------------
# Legacy pipeline (mantengo por si quieres comparar)
# -----------------------------------------------------------------------------

def find_zero_crossings(
    tfp: np.ndarray,
    grad_mag: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    gradient_threshold: float,
    vorticity: np.ndarray | None = None,
    vort_threshold: float = 5.0e-5,
    vort_factor: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = tfp.shape
    margin = 3
    window = 4
    local_pct = 70

    abs_vort = np.abs(vorticity) if vorticity is not None else None
    boosted_threshold = gradient_threshold * vort_factor

    front_lats: list[float] = []
    front_lons: list[float] = []

    for j in range(margin, ny - margin):
        for i in range(margin, nx - margin - 1):
            t0, t1 = tfp[j, i], tfp[j, i + 1]
            if t0 * t1 >= 0:
                continue

            frac = t0 / (t0 - t1)
            gm = grad_mag[j, i] * (1 - frac) + grad_mag[j, i + 1] * frac

            thr = gradient_threshold
            if abs_vort is not None:
                vor_val = abs_vort[j, i] * (1 - frac) + abs_vort[j, min(i + 1, nx - 1)] * frac
                if vor_val > vort_threshold:
                    thr = boosted_threshold

            if gm < thr:
                continue

            j_lo, j_hi = max(0, j - window), min(ny, j + window + 1)
            i_lo, i_hi = max(0, i - window), min(nx, i + window + 1)
            gm_patch = grad_mag[j_lo:j_hi, i_lo:i_hi]
            if gm < np.percentile(gm_patch, local_pct):
                continue

            lon_zc = lons[i] + frac * (lons[i + 1] - lons[i])
            front_lats.append(float(lats[j]))
            front_lons.append(float(lon_zc))

    for j in range(margin, ny - margin - 1):
        for i in range(margin, nx - margin):
            t0, t1 = tfp[j, i], tfp[j + 1, i]
            if t0 * t1 >= 0:
                continue

            frac = t0 / (t0 - t1)
            gm = grad_mag[j, i] * (1 - frac) + grad_mag[j + 1, i] * frac

            thr = gradient_threshold
            if abs_vort is not None:
                vor_val = abs_vort[j, i] * (1 - frac) + abs_vort[min(j + 1, ny - 1), i] * frac
                if vor_val > vort_threshold:
                    thr = boosted_threshold

            if gm < thr:
                continue

            j_lo, j_hi = max(0, j - window), min(ny, j + window + 1)
            i_lo, i_hi = max(0, i - window), min(nx, i + window + 1)
            gm_patch = grad_mag[j_lo:j_hi, i_lo:i_hi]
            if gm < np.percentile(gm_patch, local_pct):
                continue

            lat_zc = lats[j] + frac * (lats[j + 1] - lats[j])
            front_lats.append(float(lat_zc))
            front_lons.append(float(lons[i]))

    return np.array(front_lats), np.array(front_lons)


def _mslp_cyclonic_filter(
    front_lats: np.ndarray,
    front_lons: np.ndarray,
    msl: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    msl_smooth = gaussian_filter(msl, sigma=sigma)
    lapl = laplace(msl_smooth)

    interp = RegularGridInterpolator((lats, lons), lapl, bounds_error=False, fill_value=0)
    points = np.column_stack([front_lats, front_lons])
    lapl_at_fronts = interp(points)

    mask = lapl_at_fronts > 0
    return front_lats[mask], front_lons[mask]


def _frontogenesis_filter(
    front_lats: np.ndarray,
    front_lons: np.ndarray,
    ds: xr.Dataset,
    lats: np.ndarray,
    lons: np.ndarray,
    theta_w: np.ndarray,
    percentile: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    u850 = _ensure_2d(ds["u850"].values)
    v850 = _ensure_2d(ds["v850"].values)

    dlat = abs(np.diff(lats[:2]).item())
    dlon = abs(np.diff(lons[:2]).item())
    mean_lat = float(np.mean(lats))
    dy = dlat * np.pi / 180.0 * EARTH_RADIUS_M * units.meter
    dx = dlon * np.pi / 180.0 * EARTH_RADIUS_M * np.cos(np.radians(mean_lat)) * units.meter

    theta_w_q = theta_w * units.kelvin
    u_q = u850 * units("m/s")
    v_q = v850 * units("m/s")

    try:
        fronto = frontogenesis(theta_w_q, u_q, v_q, dx=dx, dy=dy)
        fronto_values = _ensure_2d(fronto.magnitude)
    except Exception as e:
        logger.warning("Error calculando frontogenesis: %s. Saltando filtro.", e)
        return front_lats, front_lons

    interp = RegularGridInterpolator((lats, lons), fronto_values, bounds_error=False, fill_value=0)
    points = np.column_stack([front_lats, front_lons])
    fronto_at_fronts = interp(points)

    threshold = np.percentile(fronto_at_fronts, percentile)
    mask = fronto_at_fronts >= threshold
    return front_lats[mask], front_lons[mask]


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------

def compute_tfp_fronts(ds: xr.Dataset, cfg: AppConfig) -> FrontCollection:
    ds = _remove_time_dim(ds)

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    tfp_cfg = cfg.tfp

    logger.info("TFP/H98: calculando theta_w...")
    theta_w = compute_theta_w(ds)

    # grid length chi
    chi_m = _grid_length_m(lats, lons)

    # campos H98
    tfl, tfp, grad_mag, grad_gradmag_mag, grad_mag_s, gmag_x, gmag_y = compute_h98_fields(
        theta_w=theta_w,
        lats=lats,
        lons=lons,
        smooth_sigma=float(tfp_cfg.smooth_sigma),
        gradmag_smooth_sigma=float(tfp_cfg.gradmag_smooth_sigma) if tfp_cfg.gradmag_smooth_sigma is not None else None,
        smoothing_method=str(getattr(tfp_cfg, 'smoothing_method', 'gaussian')),
        smoothing_passes=int(getattr(tfp_cfg, 'smoothing_passes', 96)),
    )

    # ABZ (H98 Eq.3): usar grad_mag sin re-suavizar (paper) o grad_mag_s (legacy)
    abz_grad = grad_mag if getattr(tfp_cfg, 'abz_use_raw_gradient', True) else grad_mag_s
    abz = compute_abz_gradient(grad_mag=abz_grad, grad_gradmag_mag=grad_gradmag_mag, chi_m=chi_m)

    # vorticidad
    vorticity = None
    if "vo850" in ds.data_vars:
        vorticity = _ensure_2d(ds["vo850"].values)
    else:
        if tfp_cfg.use_f_diagnostic:
            logger.warning("vo850 no disponible: se ignora F diagnostic.")

    # mslp
    msl_data = None
    if "msl" in ds.data_vars:
        msl_data = _ensure_2d(ds["msl"].values)

    # extraer frentes
    if tfp_cfg.use_contour_method:
        logger.info("Usando metodo H98 contour-then-mask (TFL=0 + K1/K2)...")
        polylines = _extract_contour_fronts_h98(
            tfl=tfl,
            tfp=tfp,
            abz=abz,
            lats=lats,
            lons=lons,
            cfg=cfg,
            vorticity=vorticity,
            msl=msl_data,
        )
    else:
        logger.info("Usando metodo legacy (zero-crossings + DBSCAN)...")
        front_lats, front_lons = find_zero_crossings(
            tfp=tfp,
            grad_mag=grad_mag_s,
            lats=lats,
            lons=lons,
            gradient_threshold=float(tfp_cfg.gradient_threshold),
            vorticity=vorticity,
            vort_threshold=float(tfp_cfg.vorticity_boost_threshold),
            vort_factor=float(tfp_cfg.vorticity_boost_factor),
        )

        if tfp_cfg.use_mslp_filter and msl_data is not None and len(front_lats) > 0:
            front_lats, front_lons = _mslp_cyclonic_filter(
                front_lats, front_lons, msl_data, lats, lons, sigma=float(tfp_cfg.mslp_laplacian_sigma),
            )

        if tfp_cfg.use_frontogenesis_filter and len(front_lats) > 0:
            front_lats, front_lons = _frontogenesis_filter(
                front_lats, front_lons, ds, lats, lons, theta_w, percentile=int(tfp_cfg.frontogenesis_percentile),
            )

        if len(front_lats) == 0:
            logger.warning("No se encontraron puntos de frente.")
            c = FrontCollection()
            c.metadata = {"lats": lats, "lons": lons}
            return c

        polylines = cluster_and_connect(
            front_lats, front_lons,
            eps_deg=float(tfp_cfg.dbscan_eps_deg),
            min_samples=int(tfp_cfg.dbscan_min_samples),
            min_points=int(tfp_cfg.min_front_points),
            simplify_tol=float(tfp_cfg.simplify_tolerance_deg),
            min_front_length_deg=float(tfp_cfg.min_front_length_deg),
            max_hop_deg=float(tfp_cfg.max_hop_deg),
            angular_weight=float(tfp_cfg.angular_weight),
            spline_smoothing=float(tfp_cfg.spline_smoothing),
            merge_distance_deg=float(tfp_cfg.merge_distance_deg),
            max_fronts=int(tfp_cfg.max_fronts),
        )

    # colección
    collection = FrontCollection()
    for coord_name in ("time", "valid_time"):
        if coord_name in ds.coords:
            collection.valid_time = str(ds.coords[coord_name].values)[:16]
            break

    # metadata útil para tu classifier/depuración
    collection.metadata = {
        "lats": lats,
        "lons": lons,
        "chi_m": chi_m,
        "tfl": tfl,
        "tfp": tfp,
        "abz": abz,
        "grad_mag": grad_mag,
        "gmag_x": gmag_x,
        "gmag_y": gmag_y,
    }

    for i, (plats, plons) in enumerate(polylines):
        collection.add(
            Front(
                front_type=FrontType.COLD,  # classifier decide luego
                lats=plats,
                lons=plons,
                id=f"tfp_{i:03d}",
            )
        )

    logger.info("Frentes detectados: %d", len(collection))
    return collection
