"""Algoritmo TFP (Thermal Front Parameter) para deteccion de frentes.

Implementa el metodo de Hewson (1998) basado en la temperatura
potencial del bulbo humedo (theta_w) a 850 hPa.

Pipeline:
1. Calcular theta_w a partir de t850, q850 (via dewpoint + MetPy)
2. Doble suavizado gaussiano
3. Calcular |nabla(theta_w)| en coordenadas esfericas
4. TFP = -nabla(|nabla(theta_w)|) . (nabla(theta_w) / |nabla(theta_w)|)
5. Zero-crossings con filtro de maximo local y umbral adaptativo
6. Clustering DBSCAN + ordenacion nearest-neighbor -> polilineas suaves
"""

import logging

import numpy as np
import xarray as xr
from metpy.calc import (
    dewpoint_from_specific_humidity,
    frontogenesis,
    wet_bulb_potential_temperature,
)
from metpy.units import units
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, laplace

from mapa_frentes.config import AppConfig
from mapa_frentes.fronts.connector import cluster_and_connect
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType
from mapa_frentes.utils.geo import spherical_gradient
from mapa_frentes.utils.smoothing import smooth_field

logger = logging.getLogger(__name__)


def _remove_time_dim(ds: xr.Dataset) -> xr.Dataset:
    """Extrae el primer time step si existen dimensiones temporales."""
    for time_dim in ["time", "valid_time", "step", "verify_time"]:
        if time_dim in ds.dims:
            ds = ds.isel({time_dim: 0})
    return ds


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Asegura que un array es 2D."""
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim > 2:
        arr = arr[0]
    return arr


def compute_theta_w(ds: xr.Dataset) -> np.ndarray:
    """Calcula la temperatura potencial del bulbo humedo a 850 hPa."""
    ds = _remove_time_dim(ds)

    pressure = 850.0 * units.hPa
    t850 = _ensure_2d(ds["t850"].values) * units.kelvin
    q850 = _ensure_2d(ds["q850"].values) * units("kg/kg")

    td = dewpoint_from_specific_humidity(pressure, t850, q850)
    theta_w = wet_bulb_potential_temperature(pressure, t850, td)

    return _ensure_2d(theta_w.magnitude)


def compute_tfp_field(
    theta_w: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula el campo TFP y la magnitud del gradiente.

    Usa doble suavizado: sigma completo para theta_w y sigma/2
    adicional sobre |nabla(theta_w)| antes de derivar de nuevo.
    """
    theta_w = _ensure_2d(theta_w)

    # 1. Suavizar theta_w
    theta_smooth = smooth_field(theta_w, sigma=sigma)

    # 2. Gradiente de theta_w
    gx, gy = spherical_gradient(theta_smooth, lats, lons)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Evitar division por cero
    grad_mag_safe = np.where(grad_mag > 1e-12, grad_mag, 1e-12)

    # 3. Direccion unitaria del gradiente
    ux = gx / grad_mag_safe
    uy = gy / grad_mag_safe

    # 4. Suavizado extra de grad_mag (derivadas 2do orden son ruidosas)
    grad_mag_smooth = smooth_field(grad_mag, sigma=max(sigma * 0.6, 2.0))

    # 5. Gradiente de |nabla(theta_w)|
    gmag_x, gmag_y = spherical_gradient(grad_mag_smooth, lats, lons)

    # 6. TFP = -nabla(|grad|) . (grad/|grad|)
    tfp = -(gmag_x * ux + gmag_y * uy)

    return tfp, grad_mag


def find_zero_crossings(
    tfp: np.ndarray,
    grad_mag: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    gradient_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Encuentra zero-crossings del TFP con filtros de calidad.

    Filtros aplicados:
    1. Umbral de gradiente (configurable)
    2. Maximo local del gradiente en ventana de +-4 puntos (percentil 70)
    3. Mascara de borde (descartar los 3 puntos de borde del dominio)
    """
    ny, nx = tfp.shape
    margin = 3  # puntos de margen a ignorar (efectos de borde del suavizado)
    window = 4  # semi-ventana para filtro de maximo local
    local_pct = 70  # percentil para filtro de maximo local

    front_lats = []
    front_lons = []

    # Zero-crossings eje x (longitud)
    for j in range(margin, ny - margin):
        for i in range(margin, nx - margin - 1):
            t0, t1 = tfp[j, i], tfp[j, i + 1]
            if t0 * t1 >= 0:
                continue

            frac = t0 / (t0 - t1)
            gm = grad_mag[j, i] * (1 - frac) + grad_mag[j, i + 1] * frac

            if gm < gradient_threshold:
                continue

            # Maximo local 2D (ventana cuadrada)
            j_lo, j_hi = max(0, j - window), min(ny, j + window + 1)
            i_lo, i_hi = max(0, i - window), min(nx, i + window + 1)
            gm_patch = grad_mag[j_lo:j_hi, i_lo:i_hi]
            if gm < np.percentile(gm_patch, local_pct):
                continue

            lon_zc = lons[i] + frac * (lons[i + 1] - lons[i])
            front_lats.append(lats[j])
            front_lons.append(lon_zc)

    # Zero-crossings eje y (latitud)
    for j in range(margin, ny - margin - 1):
        for i in range(margin, nx - margin):
            t0, t1 = tfp[j, i], tfp[j + 1, i]
            if t0 * t1 >= 0:
                continue

            frac = t0 / (t0 - t1)
            gm = grad_mag[j, i] * (1 - frac) + grad_mag[j + 1, i] * frac

            if gm < gradient_threshold:
                continue

            j_lo, j_hi = max(0, j - window), min(ny, j + window + 1)
            i_lo, i_hi = max(0, i - window), min(nx, i + window + 1)
            gm_patch = grad_mag[j_lo:j_hi, i_lo:i_hi]
            if gm < np.percentile(gm_patch, local_pct):
                continue

            lat_zc = lats[j] + frac * (lats[j + 1] - lats[j])
            front_lats.append(lat_zc)
            front_lons.append(lons[i])

    return np.array(front_lats), np.array(front_lons)


def _mslp_cyclonic_filter(
    front_lats: np.ndarray,
    front_lons: np.ndarray,
    msl: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Filtra puntos de frente: solo mantiene los que estan en zonas ciclonicas.

    Calcula la Laplaciana de la MSLP suavizada. Donde Laplaciana > 0, la
    curvatura indica vaguada/borrasca (region ciclonica). Los frentes en
    zonas anticiclonicas (Laplaciana < 0) se descartan.
    """
    msl_smooth = gaussian_filter(msl, sigma=sigma)
    lapl = laplace(msl_smooth)

    interp = RegularGridInterpolator(
        (lats, lons), lapl, bounds_error=False, fill_value=0
    )
    points = np.column_stack([front_lats, front_lons])
    lapl_at_fronts = interp(points)

    # Mantener solo puntos en zona ciclonica (Laplaciana > 0)
    mask = lapl_at_fronts > 0
    return front_lats[mask], front_lons[mask]


def _frontogenesis_filter(
    front_lats: np.ndarray,
    front_lons: np.ndarray,
    ds: xr.Dataset,
    lats: np.ndarray,
    lons: np.ndarray,
    theta_w: np.ndarray,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filtra puntos de frente por frontogenesis positiva.

    Calcula la funcion de frontogenesis de Petterssen usando MetPy.
    Rechaza puntos donde frontogenesis <= threshold (zona frontolitica,
    donde el gradiente se debilita). Esto elimina fragmentos espurios
    en zonas donde no hay actividad frontal real.
    """
    u850 = _ensure_2d(ds["u850"].values)
    v850 = _ensure_2d(ds["v850"].values)

    # Calcular espaciado del grid
    dlat = abs(float(np.diff(lats[:2])))
    dlon = abs(float(np.diff(lons[:2])))
    dx = dlon * units.degrees_E
    dy = dlat * units.degrees_N

    # MetPy frontogenesis: F = d|grad(theta)|/dt
    theta_w_q = theta_w * units.kelvin
    u_q = u850 * units("m/s")
    v_q = v850 * units("m/s")

    try:
        fronto = frontogenesis(theta_w_q, u_q, v_q, dx=dx, dy=dy)
        fronto_values = _ensure_2d(fronto.magnitude)
    except Exception as e:
        logger.warning("Error calculando frontogenesis: %s. Saltando filtro.", e)
        return front_lats, front_lons

    # Interpolar frontogenesis en los puntos de frente
    interp = RegularGridInterpolator(
        (lats, lons), fronto_values, bounds_error=False, fill_value=0
    )
    points = np.column_stack([front_lats, front_lons])
    fronto_at_fronts = interp(points)

    # Mantener solo puntos con frontogenesis positiva (gradiente reforzandose)
    mask = fronto_at_fronts > threshold
    return front_lats[mask], front_lons[mask]


def compute_tfp_fronts(ds: xr.Dataset, cfg: AppConfig) -> FrontCollection:
    """Pipeline completo TFP."""
    tfp_cfg = cfg.tfp
    ds = _remove_time_dim(ds)

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    # 1. theta_w
    logger.info("Calculando theta_w...")
    theta_w = compute_theta_w(ds)
    logger.info(
        "theta_w: min=%.1f, max=%.1f, shape=%s",
        np.nanmin(theta_w), np.nanmax(theta_w), theta_w.shape,
    )

    # 2. TFP
    logger.info("Calculando campo TFP (sigma=%.1f)...", tfp_cfg.smooth_sigma)
    tfp, grad_mag = compute_tfp_field(theta_w, lats, lons, tfp_cfg.smooth_sigma)

    # Diagnostico
    gm_nonzero = grad_mag[grad_mag > 0]
    p50 = np.percentile(gm_nonzero, 50)
    p90 = np.percentile(gm_nonzero, 90)
    p99 = np.percentile(gm_nonzero, 99)
    logger.info(
        "|nabla(theta_w)|: p50=%.2e, p90=%.2e, p99=%.2e, max=%.2e K/m",
        p50, p90, p99, np.max(gm_nonzero),
    )

    # Umbral adaptativo: usar el configurado, pero si es demasiado alto
    # respecto a los datos, usar p85 como fallback
    threshold = tfp_cfg.gradient_threshold
    p85 = np.percentile(gm_nonzero, 85)
    if threshold > p90:
        logger.warning(
            "Umbral (%.2e) > p90 (%.2e). Usando p85=%.2e como fallback.",
            threshold, p90, p85,
        )
        threshold = p85

    logger.info("Umbral gradiente efectivo: %.2e K/m", threshold)

    # 3. Zero-crossings
    logger.info("Buscando zero-crossings...")
    front_lats, front_lons = find_zero_crossings(
        tfp, grad_mag, lats, lons, threshold
    )
    logger.info("Puntos de frente encontrados: %d", len(front_lats))

    # 4. Filtro MSLP: descartar frentes en zonas anticiclonicas
    if tfp_cfg.use_mslp_filter and "msl" in ds.data_vars and len(front_lats) > 0:
        msl_data = _ensure_2d(ds["msl"].values)
        n_before = len(front_lats)
        front_lats, front_lons = _mslp_cyclonic_filter(
            front_lats, front_lons, msl_data, lats, lons,
            sigma=tfp_cfg.mslp_laplacian_sigma,
        )
        logger.info(
            "Filtro MSLP ciclonico: %d -> %d puntos (eliminados %d en zonas anticiclonicas)",
            n_before, len(front_lats), n_before - len(front_lats),
        )

    # 5. Filtro frontogenesis: descartar puntos frontoliticos
    if tfp_cfg.use_frontogenesis_filter and len(front_lats) > 0:
        n_before = len(front_lats)
        front_lats, front_lons = _frontogenesis_filter(
            front_lats, front_lons, ds, lats, lons, theta_w,
            threshold=tfp_cfg.frontogenesis_threshold,
        )
        logger.info(
            "Filtro frontogenesis: %d -> %d puntos (eliminados %d frontoliticos)",
            n_before, len(front_lats), n_before - len(front_lats),
        )

    if len(front_lats) == 0:
        logger.warning("No se encontraron puntos de frente.")
        return FrontCollection()

    # 7. Clustering y conexion
    logger.info("Conectando puntos en polilineas...")
    polylines = cluster_and_connect(
        front_lats, front_lons,
        eps_deg=tfp_cfg.dbscan_eps_deg,
        min_samples=tfp_cfg.dbscan_min_samples,
        min_points=tfp_cfg.min_front_points,
        simplify_tol=tfp_cfg.simplify_tolerance_deg,
        min_front_length_deg=tfp_cfg.min_front_length_deg,
        max_hop_deg=tfp_cfg.max_hop_deg,
        angular_weight=tfp_cfg.angular_weight,
        spline_smoothing=tfp_cfg.spline_smoothing,
        merge_distance_deg=tfp_cfg.merge_distance_deg,
        max_fronts=tfp_cfg.max_fronts,
    )

    # 8. Crear FrontCollection
    collection = FrontCollection()
    for coord_name in ("time", "valid_time"):
        if coord_name in ds.coords:
            collection.valid_time = str(ds.coords[coord_name].values)[:16]
            break

    for i, (plats, plons) in enumerate(polylines):
        front = Front(
            front_type=FrontType.COLD,
            lats=plats,
            lons=plons,
            id=f"tfp_{i:03d}",
        )
        collection.add(front)

    logger.info("Frentes conectados: %d", len(collection))
    return collection
