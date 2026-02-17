"""Algoritmo TFP (Thermal Front Parameter) para deteccion de frentes.

Implementa el metodo de Hewson (1998) mejorado con:
- Contour-then-mask (Sansom & Catto 2024): extrae contornos TFP=0 como lineas
  coherentes, luego filtra por gradiente y diagnostico F.
- Diagnostico F (Parfitt et al. 2017): F = (|grad_theta| / escala) * (|vor| / |f|)
  Combina informacion termica y dinamica para eliminar falsos positivos.
- Vorticity boost: reduce umbral de gradiente en zonas de alta vorticidad.

Pipeline:
1. Calcular theta_w a partir de t850, q850 (via dewpoint + MetPy)
2. Suavizado gaussiano (sigma alto para 0.25deg: ~8)
3. Calcular |nabla(theta_w)| en coordenadas esfericas
4. TFP = -nabla(|nabla(theta_w)|) . (nabla(theta_w) / |nabla(theta_w)|)
5. Extraer contornos TFP=0 (skimage.measure.find_contours)
6. Filtrar puntos: gradiente > umbral, F > 1, zona ciclonica
7. Segmentar, suavizar con spline, filtro de longitud minima
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
from mapa_frentes.fronts.connector import (
    cluster_and_connect,
    _smooth_spline,
    _polyline_length_deg,
)
from mapa_frentes.fronts.models import Front, FrontCollection, FrontType
from mapa_frentes.utils.geo import spherical_gradient
from mapa_frentes.utils.smoothing import smooth_field

from collections import deque

logger = logging.getLogger(__name__)

# Constantes para el diagnostico F (Parfitt et al. 2017)
OMEGA = 7.2921e-5           # velocidad angular de la Tierra (rad/s)
GRAD_SCALE = 4.5e-6         # escala tipica de gradiente frontal (K/m) = 0.45 K/100km
MIN_CORIOLIS_LAT = 10.0     # latitud minima para evitar division por cero en f


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcula el campo TFP, magnitud del gradiente, y gradiente de la magnitud.

    Usa doble suavizado: sigma completo para theta_w y sigma/2
    adicional sobre |nabla(theta_w)| antes de derivar de nuevo.

    Returns:
        (tfp, grad_mag, gmag_x, gmag_y) donde gmag_x/gmag_y son las
        componentes de nabla(|nabla(theta_w)|), necesarias para la
        clasificacion frio/calido (Hewson 1998 Eq. 4).
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

    # 5. Gradiente de |nabla(theta_w)| (usado para TFP y clasificacion Hewson)
    gmag_x, gmag_y = spherical_gradient(grad_mag_smooth, lats, lons)

    # 6. TFP = -nabla(|grad|) . (grad/|grad|)
    tfp = -(gmag_x * ux + gmag_y * uy)

    return tfp, grad_mag, gmag_x, gmag_y


# ============================================================================
# Nuevo pipeline: contour-then-mask (Sansom & Catto 2024)
# ============================================================================

def _compute_f_diagnostic(
    grad_mag: np.ndarray,
    vorticity: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """Calcula el diagnostico F de Parfitt et al. (2017).

    F = (|nabla(theta_w)| / grad_scale) * (|vorticity| / |f|)

    Donde:
    - grad_scale = 4.5e-6 K/m (escala tipica de gradiente frontal)
    - f = 2*Omega*sin(lat) (parametro de Coriolis)
    - Se clipea |f| en lat=10deg para evitar singularidad ecuatorial

    Returns:
        Campo F 2D (mismo shape que grad_mag).
    """
    # Coriolis 2D
    f_coriolis = 2 * OMEGA * np.abs(np.sin(np.radians(lats)))
    f_min = 2 * OMEGA * np.sin(np.radians(MIN_CORIOLIS_LAT))
    f_coriolis = np.maximum(f_coriolis, f_min)
    f_2d = f_coriolis[:, np.newaxis] * np.ones((1, grad_mag.shape[1]))

    return (grad_mag / GRAD_SCALE) * (np.abs(vorticity) / f_2d)


def _split_by_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    mask: np.ndarray,
    min_points: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Divide un contorno en segmentos donde mask es True.

    Cada segmento continuo de puntos validos se separa.
    Solo se devuelven segmentos con >= min_points puntos.
    """
    segments = []
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


def _extract_contour_fronts(
    tfp_field: np.ndarray,
    grad_mag: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    cfg: AppConfig,
    vorticity: np.ndarray | None = None,
    msl: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extrae frentes como contornos TFP=0, luego filtra (contour-then-mask).

    Metodo de Sansom & Catto (2024): primero extraer contornos continuos
    del campo TFP=0, luego filtrar puntos por gradiente, diagnostico F,
    y zona ciclonica. Esto produce frentes mucho mas coherentes que el
    metodo clasico de mask-then-join (DBSCAN + nearest-neighbor).

    Filtros aplicados en cada punto del contorno:
    1. Margen de borde (3 pixeles)
    2. |nabla(theta_w)| >= umbral (con vorticity boost)
    3. F >= threshold (diagnostico F de Parfitt 2017, si habilitado)
    4. Laplaciana(MSLP) > 0 (zona ciclonica, si habilitado)

    Returns:
        Lista de (lats, lons) polylines.
    """
    from skimage.measure import find_contours

    tfp_cfg = cfg.tfp
    margin = 3
    threshold = tfp_cfg.gradient_threshold

    # --- Extraer contornos TFP=0 ---
    raw_contours = find_contours(tfp_field, 0.0)
    logger.info("Contornos TFP=0 extraidos: %d", len(raw_contours))

    # --- Preparar interpoladores en espacio pixel ---
    row_idx = np.arange(len(lats))
    col_idx = np.arange(len(lons))

    interp_grad = RegularGridInterpolator(
        (row_idx, col_idx), grad_mag,
        bounds_error=False, fill_value=0,
    )

    # F diagnostic
    interp_f = None
    if vorticity is not None and tfp_cfg.use_f_diagnostic:
        f_field = _compute_f_diagnostic(grad_mag, vorticity, lats)
        interp_f = RegularGridInterpolator(
            (row_idx, col_idx), f_field,
            bounds_error=False, fill_value=0,
        )
        logger.info(
            "F diagnostic: p50=%.2f, p90=%.2f, p99=%.2f",
            np.percentile(f_field, 50),
            np.percentile(f_field, 90),
            np.percentile(f_field, 99),
        )

    # Vorticity boost
    interp_vort = None
    if vorticity is not None and tfp_cfg.use_vorticity_boost:
        interp_vort = RegularGridInterpolator(
            (row_idx, col_idx), np.abs(vorticity),
            bounds_error=False, fill_value=0,
        )

    # MSLP Laplacian
    interp_msl = None
    if msl is not None and tfp_cfg.use_mslp_filter:
        msl_smooth = gaussian_filter(msl, sigma=tfp_cfg.mslp_laplacian_sigma)
        msl_lapl = laplace(msl_smooth)
        interp_msl = RegularGridInterpolator(
            (row_idx, col_idx), msl_lapl,
            bounds_error=False, fill_value=0,
        )

    boosted_threshold = threshold * tfp_cfg.vorticity_boost_factor
    polylines = []

    for contour in raw_contours:
        if len(contour) < tfp_cfg.min_front_points:
            continue

        rows, cols = contour[:, 0], contour[:, 1]

        # 1. Margen de borde
        mask = (
            (rows >= margin) & (rows <= len(lats) - margin - 1)
            & (cols >= margin) & (cols <= len(lons) - margin - 1)
        )

        # 2. Gradiente con vorticity boost
        gm_vals = interp_grad(contour)
        eff_threshold = np.full(len(contour), threshold)
        if interp_vort is not None:
            vort_vals = interp_vort(contour)
            boost_mask = vort_vals > tfp_cfg.vorticity_boost_threshold
            eff_threshold[boost_mask] = boosted_threshold
        mask &= (gm_vals >= eff_threshold)

        # 3. F diagnostic
        if interp_f is not None:
            f_vals = interp_f(contour)
            mask &= (f_vals >= tfp_cfg.f_diagnostic_threshold)

        # 4. MSLP ciclonica
        if interp_msl is not None:
            msl_vals = interp_msl(contour)
            mask &= (msl_vals > 0)

        # Convertir pixel -> lat/lon
        c_lats = np.interp(rows, row_idx, lats)
        c_lons = np.interp(cols, col_idx, lons)

        # Segmentar en trozos continuos validos
        segments = _split_by_mask(c_lats, c_lons, mask, tfp_cfg.min_front_points)

        for seg_lats, seg_lons in segments:
            length = _polyline_length_deg(seg_lats, seg_lons)
            if length < tfp_cfg.min_front_length_deg:
                continue

            # Downsample si el contorno tiene demasiados puntos
            if len(seg_lats) > 200:
                step = max(1, len(seg_lats) // 100)
                seg_lats = seg_lats[::step]
                seg_lons = seg_lons[::step]

            # Suavizar con spline
            if len(seg_lats) >= 4:
                seg_lats, seg_lons = _smooth_spline(
                    seg_lats, seg_lons, smoothing=tfp_cfg.spline_smoothing,
                )

            if len(seg_lats) >= 3:
                polylines.append((seg_lats, seg_lons))

    # Ordenar por longitud (mas largo primero)
    polylines.sort(
        key=lambda p: _polyline_length_deg(p[0], p[1]), reverse=True
    )

    # Limitar numero maximo
    if tfp_cfg.max_fronts > 0 and len(polylines) > tfp_cfg.max_fronts:
        polylines = polylines[:tfp_cfg.max_fronts]

    logger.info("Frentes por contorno: %d polylines", len(polylines))
    return polylines


# ============================================================================
# Legacy pipeline: zero-crossings + DBSCAN (mantener como fallback)
# ============================================================================

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
    """Encuentra zero-crossings del TFP con filtros de calidad (metodo legacy)."""
    ny, nx = tfp.shape
    margin = 3
    window = 4
    local_pct = 70

    abs_vort = np.abs(vorticity) if vorticity is not None else None
    boosted_threshold = gradient_threshold * vort_factor

    front_lats = []
    front_lons = []

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
            front_lats.append(lats[j])
            front_lons.append(lon_zc)

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
    """Filtra puntos de frente: solo mantiene los que estan en zonas ciclonicas."""
    msl_smooth = gaussian_filter(msl, sigma=sigma)
    lapl = laplace(msl_smooth)

    interp = RegularGridInterpolator(
        (lats, lons), lapl, bounds_error=False, fill_value=0
    )
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
    """Filtra puntos de frente por frontogenesis."""
    u850 = _ensure_2d(ds["u850"].values)
    v850 = _ensure_2d(ds["v850"].values)

    dlat = abs(np.diff(lats[:2]).item())
    dlon = abs(np.diff(lons[:2]).item())
    mean_lat = np.mean(lats)
    dy = dlat * np.pi / 180.0 * 6.371e6 * units.meter
    dx = dlon * np.pi / 180.0 * 6.371e6 * np.cos(np.radians(mean_lat)) * units.meter

    theta_w_q = theta_w * units.kelvin
    u_q = u850 * units("m/s")
    v_q = v850 * units("m/s")

    try:
        fronto = frontogenesis(theta_w_q, u_q, v_q, dx=dx, dy=dy)
        fronto_values = _ensure_2d(fronto.magnitude)
    except Exception as e:
        logger.warning("Error calculando frontogenesis: %s. Saltando filtro.", e)
        return front_lats, front_lons

    interp = RegularGridInterpolator(
        (lats, lons), fronto_values, bounds_error=False, fill_value=0,
    )
    points = np.column_stack([front_lats, front_lons])
    fronto_at_fronts = interp(points)

    threshold = np.percentile(fronto_at_fronts, percentile)
    mask = fronto_at_fronts >= threshold
    logger.debug(
        "Frontogenesis: p%d=%.2e, descartando %d de %d puntos",
        percentile, threshold, np.sum(~mask), len(mask),
    )
    return front_lats[mask], front_lons[mask]


# ============================================================================
# Pipeline principal
# ============================================================================

def compute_tfp_fronts(ds: xr.Dataset, cfg: AppConfig) -> FrontCollection:
    """Pipeline completo TFP.

    Usa contour-then-mask (Sansom & Catto 2024) si use_contour_method=True,
    o el metodo legacy (DBSCAN + nearest-neighbor) si False.
    """
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
    tfp_field, grad_mag, gmag_x, gmag_y = compute_tfp_field(
        theta_w, lats, lons, tfp_cfg.smooth_sigma,
    )

    # Diagnostico
    gm_nonzero = grad_mag[grad_mag > 0]
    p50 = np.percentile(gm_nonzero, 50)
    p90 = np.percentile(gm_nonzero, 90)
    p99 = np.percentile(gm_nonzero, 99)
    logger.info(
        "|nabla(theta_w)|: p50=%.2e, p90=%.2e, p99=%.2e, max=%.2e K/m",
        p50, p90, p99, np.max(gm_nonzero),
    )

    # Umbral adaptativo
    threshold = tfp_cfg.gradient_threshold
    p85 = np.percentile(gm_nonzero, 85)
    if threshold > p90:
        logger.warning(
            "Umbral (%.2e) > p90 (%.2e). Usando p85=%.2e como fallback.",
            threshold, p90, p85,
        )
        threshold = p85
    # Guardar threshold efectivo para el contour method
    tfp_cfg_effective_threshold = threshold

    logger.info("Umbral gradiente efectivo: %.2e K/m", threshold)

    # 3. Extraer vorticidad (para F diagnostic y/o vorticity boost)
    vorticity = None
    if "vo850" in ds.data_vars:
        vorticity = _ensure_2d(ds["vo850"].values)
        if tfp_cfg.use_vorticity_boost:
            logger.info(
                "Vorticity boost activo: threshold=%.2e, factor=%.2f",
                tfp_cfg.vorticity_boost_threshold,
                tfp_cfg.vorticity_boost_factor,
            )
        if tfp_cfg.use_f_diagnostic:
            logger.info("F diagnostic activo: threshold=%.2f", tfp_cfg.f_diagnostic_threshold)
    elif tfp_cfg.use_f_diagnostic:
        logger.warning("F diagnostic configurado pero vo850 no disponible.")

    # 4. Extraer MSL
    msl_data = None
    if "msl" in ds.data_vars:
        msl_data = _ensure_2d(ds["msl"].values)

    # --- Elegir metodo de extraccion ---
    if tfp_cfg.use_contour_method:
        # Nuevo: contour-then-mask (Sansom & Catto 2024)
        logger.info("Usando metodo contour-then-mask...")

        # Temporalmente ajustar el threshold en cfg para pasarlo
        original_threshold = tfp_cfg.gradient_threshold
        tfp_cfg.gradient_threshold = tfp_cfg_effective_threshold

        polylines = _extract_contour_fronts(
            tfp_field, grad_mag, lats, lons, cfg,
            vorticity=vorticity,
            msl=msl_data,
        )

        # Restaurar
        tfp_cfg.gradient_threshold = original_threshold

    else:
        # Legacy: zero-crossings + DBSCAN
        logger.info("Usando metodo legacy (zero-crossings + DBSCAN)...")
        front_lats, front_lons = find_zero_crossings(
            tfp_field, grad_mag, lats, lons, threshold,
            vorticity=vorticity,
            vort_threshold=tfp_cfg.vorticity_boost_threshold,
            vort_factor=tfp_cfg.vorticity_boost_factor,
        )
        logger.info("Puntos de frente encontrados: %d", len(front_lats))

        if tfp_cfg.use_mslp_filter and msl_data is not None and len(front_lats) > 0:
            n_before = len(front_lats)
            front_lats, front_lons = _mslp_cyclonic_filter(
                front_lats, front_lons, msl_data, lats, lons,
                sigma=tfp_cfg.mslp_laplacian_sigma,
            )
            logger.info("Filtro MSLP: %d -> %d puntos", n_before, len(front_lats))

        if tfp_cfg.use_frontogenesis_filter and len(front_lats) > 0:
            n_before = len(front_lats)
            front_lats, front_lons = _frontogenesis_filter(
                front_lats, front_lons, ds, lats, lons, theta_w,
                percentile=tfp_cfg.frontogenesis_percentile,
            )
            logger.info("Filtro frontogenesis: %d -> %d puntos", n_before, len(front_lats))

        if len(front_lats) == 0:
            logger.warning("No se encontraron puntos de frente.")
            c = FrontCollection()
            c.metadata = {"gmag_x": gmag_x, "gmag_y": gmag_y, "lats": lats, "lons": lons}
            return c

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

    # --- Crear FrontCollection ---
    collection = FrontCollection()
    for coord_name in ("time", "valid_time"):
        if coord_name in ds.coords:
            collection.valid_time = str(ds.coords[coord_name].values)[:16]
            break

    # Guardar campos para clasificacion Hewson (Eq. 4)
    collection.metadata = {
        "gmag_x": gmag_x,
        "gmag_y": gmag_y,
        "lats": lats,
        "lons": lons,
    }

    for i, (plats, plons) in enumerate(polylines):
        front = Front(
            front_type=FrontType.COLD,
            lats=plats,
            lons=plons,
            id=f"tfp_{i:03d}",
        )
        collection.add(front)

    logger.info("Frentes detectados: %d", len(collection))
    return collection

def hysteresis_grow(core: np.ndarray, grow: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Crece core dentro de grow (ambos booleanos)."""
    out = np.zeros_like(core, dtype=bool)
    q = deque()

    ys, xs = np.where(core)
    for y, x in zip(ys, xs):
        out[y, x] = True
        q.append((y, x))

    neigh = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        neigh += [(-1,-1),(-1,1),(1,-1),(1,1)]

    H, W = core.shape
    while q:
        y, x = q.popleft()
        for dy, dx in neigh:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                if grow[yy, xx] and not out[yy, xx]:
                    out[yy, xx] = True
                    q.append((yy, xx))
    return out
