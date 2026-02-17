"""MAPA_FRENTES - Prototipo web con FastAPI.

Backend que envuelve las funciones existentes de mapa_frentes/ como endpoints REST.
Campos derivados se renderizan como PNG transparentes (matplotlib headless).
Isobaras se extraen como GeoJSON. Frentes como GeoJSON editable.
"""

import io
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import uvicorn

# Asegurar que mapa_frentes es importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mapa_frentes.config import load_config
from mapa_frentes.analysis.derived_fields import (
    compute_derived_field, AVAILABLE_FIELDS, DerivedField,
)
from mapa_frentes.fronts.models import FrontCollection
from mapa_frentes.fronts.io import collection_to_geojson, collection_from_geojson

logger = logging.getLogger(__name__)
app = FastAPI(title="MAPA_FRENTES Web")


# ---------------------------------------------------------------------------
# Estado global de sesion (single-user prototype)
# ---------------------------------------------------------------------------

class SessionState:
    def __init__(self):
        self.cfg = load_config(PROJECT_ROOT / "config.yaml")
        self.ds = None
        self.lats = None
        self.lons = None
        self.msl_smooth = None
        self.levels = None
        self.centers = []
        self.fronts = FrontCollection()
        self.field_cache: dict[str, bytes] = {}
        self.derived_cache: dict[str, DerivedField] = {}
        self.isobar_geojson = None
        self.date_info = ""


state = SessionState()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_coords():
    """Extrae lats/lons del dataset actual."""
    ds = state.ds
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    state.lats = ds[lat_name].values
    state.lons = ds[lon_name].values


def _compute_viewport_range(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    south: float | None,
    north: float | None,
    west: float | None,
    east: float | None,
) -> tuple[float, float]:
    """Calcula vmin/vmax como percentiles 5 y 95 (viewport si existe)."""

    def _p05_p95(arr: np.ndarray) -> tuple[float, float]:
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            # fallback duro si todo es NaN
            return 0.0, 1.0
        vmin = float(np.percentile(vals, 5))
        vmax = float(np.percentile(vals, 95))
        # evitar rango degenerado (todo casi constante)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            mn = float(np.nanmin(vals))
            mx = float(np.nanmax(vals))
            if mn == mx:
                eps = 1e-6 if mn == 0 else abs(mn) * 1e-3
                return mn - eps, mx + eps
            return mn, mx
        return vmin, vmax

    # Si no hay viewport, percentiles del campo completo
    if south is None or north is None or west is None or east is None:
        return _p05_p95(data)

    lat_mask = (lats >= south) & (lats <= north)
    lon_mask = (lons >= west) & (lons <= east)

    # Si el viewport no cae dentro de la malla, usa campo completo
    if not np.any(lat_mask) or not np.any(lon_mask):
        return _p05_p95(data)

    subset = data[np.ix_(lat_mask, lon_mask)]
    if subset.size == 0 or np.all(~np.isfinite(subset)):
        return _p05_p95(data)

    return _p05_p95(subset)



def _lat_to_mercator_y(lat_deg: np.ndarray) -> np.ndarray:
    """Convierte latitud en grados a coordenada y de Web Mercator (radianes)."""
    lat_rad = np.deg2rad(np.clip(lat_deg, -85, 85))
    return np.log(np.tan(np.pi / 4 + lat_rad / 2))


def _render_field_png(
    derived: DerivedField,
    lats: np.ndarray,
    lons: np.ndarray,
    width_px: int = 900,
    vmin: float | None = None,
    vmax: float | None = None,
) -> bytes:
    """Renderiza un DerivedField como PNG transparente en proyeccion Web Mercator.

    El PNG cubre exactamente [lons.min(), lons.max()] x [lats.min(), lats.max()]
    para alinearse con L.imageOverlay en Leaflet (que usa Web Mercator).
    Los datos se re-interpolan a un grid uniforme en coordenada y Mercator
    para que los pixeles coincidan con los tiles de Leaflet.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Convertir lats a espacio Mercator y crear grid uniforme
    merc_y = _lat_to_mercator_y(lats)
    merc_y_min, merc_y_max = float(merc_y.min()), float(merc_y.max())
    n_merc = len(lats)
    merc_y_uniform = np.linspace(merc_y_min, merc_y_max, n_merc)

    # Latitudes correspondientes al grid Mercator uniforme
    lats_merc = np.rad2deg(2 * np.arctan(np.exp(merc_y_uniform)) - np.pi / 2)

    # Interpolar datos al grid Mercator
    # lats del dataset pueden estar en orden descendente
    lat_sorted = lats if lats[0] < lats[-1] else lats[::-1]
    data_sorted = derived.data if lats[0] < lats[-1] else derived.data[::-1, :]

    interp = RegularGridInterpolator(
        (lat_sorted, lons), data_sorted,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    lon2d_merc, lat2d_merc = np.meshgrid(lons, lats_merc)
    data_merc = interp((lat2d_merc, lon2d_merc))

    # Calcular aspect ratio en espacio Mercator
    merc_height = merc_y_max - merc_y_min
    lon_width = float(lons.max() - lons.min())
    aspect = merc_height / lon_width if lon_width > 0 else 1.0
    height_px = int(width_px * aspect)
    height_px = max(height_px, 100)

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    num_levels = 20
    if vmin is None:
        vmin = float(np.nanmin(derived.data))
    if vmax is None:
        vmax = float(np.nanmax(derived.data))

    if derived.center_zero:
        abs_max = max(abs(vmin), abs(vmax))
        levels = np.linspace(-abs_max, abs_max, num_levels)
        norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        levels = np.linspace(vmin, vmax, num_levels)
        norm = None

    # Plotear en espacio Mercator (eje y = merc_y_uniform)
    lon2d_plot, merc2d_plot = np.meshgrid(lons, merc_y_uniform)
    ax.contourf(
        lon2d_plot, merc2d_plot, data_merc,
        levels=levels, cmap=derived.cmap, norm=norm,
        alpha=0.7, extend="both",
    )
    ax.set_xlim(float(lons.min()), float(lons.max()))
    ax.set_ylim(merc_y_min, merc_y_max)
    ax.set_aspect("auto")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _get_derived(field_name: str) -> DerivedField | None:
    """Obtiene campo derivado con cache."""
    if field_name in state.derived_cache:
        return state.derived_cache[field_name]
    derived = compute_derived_field(state.ds, field_name, state.lats, state.lons)
    if derived is not None:
        state.derived_cache[field_name] = derived
    return derived


def _extract_isobar_geojson() -> dict:
    """Extrae contornos de isobaras como GeoJSON LineStrings."""
    lon2d, lat2d = np.meshgrid(state.lons, state.lats)

    fig, ax = plt.subplots(figsize=(1, 1))
    cs = ax.contour(lon2d, lat2d, state.msl_smooth, levels=state.levels)

    features = []
    # allsegs: list[list[ndarray]] indexado por nivel
    if hasattr(cs, "allsegs"):
        for level_idx, segs in enumerate(cs.allsegs):
            level_val = float(cs.levels[level_idx])
            is_master = (level_val % 20 == 0)
            for seg in segs:
                if len(seg) < 2:
                    continue
                coords = seg.tolist()
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "level": level_val,
                        "is_master": is_master,
                    },
                })
    else:
        # Fallback para matplotlib < 3.8
        for i, level_val in enumerate(cs.levels):
            is_master = (float(level_val) % 20 == 0)
            for path in cs.collections[i].get_paths():
                vertices = path.vertices
                if len(vertices) < 2:
                    continue
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": vertices.tolist(),
                    },
                    "properties": {
                        "level": float(level_val),
                        "is_master": is_master,
                    },
                })
    plt.close(fig)
    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/load")
async def load_data(request: Request):
    """Descarga datos ECMWF, lee GRIB, calcula isobaras y centros."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    date_str = body.get("date")
    step = body.get("step", 0)

    dt = None
    if date_str:
        try:
            dt = datetime.strptime(date_str, "%Y%m%d%H")
        except ValueError:
            raise HTTPException(400, f"Formato de fecha invalido: {date_str}. Use YYYYMMDDHH")

    from mapa_frentes.data.ecmwf_download import download_ecmwf
    from mapa_frentes.data.grib_reader import read_grib_files
    from mapa_frentes.analysis.isobars import smooth_mslp, compute_isobar_levels
    from mapa_frentes.analysis.pressure_centers import detect_pressure_centers

    grib_paths = download_ecmwf(state.cfg, date=dt, step=step)
    state.ds = read_grib_files(grib_paths, state.cfg)
    _extract_coords()

    state.msl_smooth = smooth_mslp(
        state.ds["msl"], sigma=state.cfg.isobars.smooth_sigma,
    )
    state.levels = compute_isobar_levels(
        state.msl_smooth, interval=state.cfg.isobars.interval_hpa,
    )
    state.centers = detect_pressure_centers(
        state.msl_smooth, state.lats, state.lons, state.cfg,
    )

    # Invalidar caches
    state.field_cache.clear()
    state.derived_cache.clear()
    state.isobar_geojson = None
    state.fronts = FrontCollection()

    # Info de tiempo
    state.date_info = ""
    for coord_name in ("time", "valid_time"):
        if coord_name in state.ds.coords:
            state.date_info = str(state.ds.coords[coord_name].values)[:16]
            break

    return {
        "status": "ok",
        "n_centers": len(state.centers),
        "date_info": state.date_info,
    }


@app.get("/api/bounds")
async def get_bounds():
    """Devuelve limites geograficos y campos disponibles."""
    area = state.cfg.area
    data_bounds = None
    if state.lats is not None:
        data_bounds = {
            "south": float(state.lats.min()),
            "north": float(state.lats.max()),
            "west": float(state.lons.min()),
            "east": float(state.lons.max()),
        }
    return {
        "area": {
            "lat_min": area.lat_min, "lat_max": area.lat_max,
            "lon_min": area.lon_min, "lon_max": area.lon_max,
        },
        "data_bounds": data_bounds,
        "fields": {k: v for k, v in AVAILABLE_FIELDS.items() if k != "none"},
        "date_info": state.date_info,
    }


@app.get("/api/fields/{field_name}/image")
async def get_field_image(
    field_name: str,
    south: float | None = Query(None),
    north: float | None = Query(None),
    west: float | None = Query(None),
    east: float | None = Query(None),
):
    """Devuelve PNG transparente de un campo derivado.

    Si se pasan south/north/west/east, la escala de color se ajusta
    al rango de valores dentro del viewport visible.
    """
    if state.ds is None:
        raise HTTPException(400, "No hay datos cargados. Llame POST /api/load primero.")
    if field_name not in AVAILABLE_FIELDS or field_name == "none":
        raise HTTPException(404, f"Campo desconocido: {field_name}")

    # Con viewport, no usar cache (depende de la vista)
    has_viewport = south is not None
    cache_key = field_name if not has_viewport else None

    if cache_key and cache_key in state.field_cache:
        return StreamingResponse(
            io.BytesIO(state.field_cache[cache_key]),
            media_type="image/png",
        )

    derived = _get_derived(field_name)
    if derived is None:
        raise HTTPException(404, f"No se puede calcular el campo: {field_name}")

    vmin, vmax = None, None
    if has_viewport:
        vmin, vmax = _compute_viewport_range(
            derived.data, state.lats, state.lons, south, north, west, east,
        )

    png_bytes = _render_field_png(derived, state.lats, state.lons, vmin=vmin, vmax=vmax)
    if cache_key:
        state.field_cache[cache_key] = png_bytes
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/api/fields/{field_name}/colorbar")
async def get_field_colorbar(
    field_name: str,
    south: float | None = Query(None),
    north: float | None = Query(None),
    west: float | None = Query(None),
    east: float | None = Query(None),
):
    """Devuelve metadata de la colorbar para un campo: vmin, vmax, label, units, cmap, center_zero."""
    if state.ds is None:
        raise HTTPException(400, "No hay datos cargados.")
    if field_name not in AVAILABLE_FIELDS or field_name == "none":
        raise HTTPException(404, f"Campo desconocido: {field_name}")

    derived = _get_derived(field_name)
    if derived is None:
        raise HTTPException(404, f"No se puede calcular el campo: {field_name}")

    vmin, vmax = _compute_viewport_range(
        derived.data, state.lats, state.lons, south, north, west, east,
    )

    return {
        "field": field_name,
        "vmin": round(vmin, 4),
        "vmax": round(vmax, 4),
        "label": derived.label,
        "units": derived.units,
        "cmap": derived.cmap,
        "center_zero": derived.center_zero,
    }


def _render_precipitation_png(
    ds,
    lats: np.ndarray,
    lons: np.ndarray,
    cfg,
    width_px: int = 1400,
) -> bytes:
    """Renderiza precipitacion como PNG transparente en Web Mercator."""
    from scipy.interpolate import RegularGridInterpolator
    from mapa_frentes.analysis.derived_fields import compute_precipitation

    precip_mm = compute_precipitation(ds)
    if precip_mm is None:
        logger.info("Precipitacion render: compute_precipitation devolvio None")
        return b""

    pcfg = cfg.precipitation
    vmax = float(np.nanmax(precip_mm))
    logger.info("Precipitacion render: max=%.2f mm, umbral=%.2f mm", vmax, pcfg.threshold_mm)
    if vmax <= pcfg.threshold_mm:
        return b""

    # Re-interpolar a espacio Mercator
    merc_y = _lat_to_mercator_y(lats)
    merc_y_min, merc_y_max = float(merc_y.min()), float(merc_y.max())
    n_merc = len(lats)
    merc_y_uniform = np.linspace(merc_y_min, merc_y_max, n_merc)
    lats_merc = np.rad2deg(2 * np.arctan(np.exp(merc_y_uniform)) - np.pi / 2)

    lat_sorted = lats if lats[0] < lats[-1] else lats[::-1]
    data_sorted = precip_mm if lats[0] < lats[-1] else precip_mm[::-1, :]

    interp = RegularGridInterpolator(
        (lat_sorted, lons), data_sorted,
        method="linear", bounds_error=False, fill_value=0,
    )
    lon2d_merc, lat2d_merc = np.meshgrid(lons, lats_merc)
    data_merc = interp((lat2d_merc, lon2d_merc))

    # Figure
    merc_height = merc_y_max - merc_y_min
    lon_width = float(lons.max() - lons.min())
    aspect = merc_height / lon_width if lon_width > 0 else 1.0
    height_px = max(int(width_px * aspect), 100)

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    levels = np.linspace(pcfg.threshold_mm, vmax, pcfg.num_levels)
    lon2d_plot, merc2d_plot = np.meshgrid(lons, merc_y_uniform)
    ax.contourf(
        lon2d_plot, merc2d_plot, data_merc,
        levels=levels, cmap=pcfg.cmap,
        alpha=pcfg.alpha, extend="max",
    )
    ax.set_xlim(float(lons.min()), float(lons.max()))
    ax.set_ylim(merc_y_min, merc_y_max)
    ax.set_aspect("auto")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_wind_vectors_png(
    ds,
    lats: np.ndarray,
    lons: np.ndarray,
    level: int,
    cfg,
    width_px: int = 1400,
) -> bytes:
    """Renderiza vectores de viento como PNG transparente en Web Mercator.

    La componente v se escala por la derivada de la proyeccion Mercator
    para que las flechas apunten en la direccion correcta en el mapa.
    """
    from scipy.interpolate import RegularGridInterpolator

    u_name, v_name = f"u{level}", f"v{level}"
    if u_name not in ds or v_name not in ds:
        return b""

    wv_cfg = cfg.wind_vectors
    thin = wv_cfg.thin_factor

    u = ds[u_name].values
    v = ds[v_name].values
    while u.ndim > 2:
        u = u[0]
    while v.ndim > 2:
        v = v[0]

    u_kt = u * 1.94384
    v_kt = v * 1.94384

    # Re-interpolar a espacio Mercator
    merc_y = _lat_to_mercator_y(lats)
    merc_y_min, merc_y_max = float(merc_y.min()), float(merc_y.max())
    n_merc = len(lats)
    merc_y_uniform = np.linspace(merc_y_min, merc_y_max, n_merc)
    lats_merc = np.rad2deg(2 * np.arctan(np.exp(merc_y_uniform)) - np.pi / 2)

    lat_sorted = lats if lats[0] < lats[-1] else lats[::-1]
    u_sorted = u_kt if lats[0] < lats[-1] else u_kt[::-1, :]
    v_sorted = v_kt if lats[0] < lats[-1] else v_kt[::-1, :]

    interp_u = RegularGridInterpolator(
        (lat_sorted, lons), u_sorted,
        method="linear", bounds_error=False, fill_value=0,
    )
    interp_v = RegularGridInterpolator(
        (lat_sorted, lons), v_sorted,
        method="linear", bounds_error=False, fill_value=0,
    )
    lon2d_merc, lat2d_merc = np.meshgrid(lons, lats_merc)
    u_merc = interp_u((lat2d_merc, lon2d_merc))
    v_merc = interp_v((lat2d_merc, lon2d_merc))

    # Escalar v por la derivada de Mercator: dy/dlat = 1/cos(lat)
    # Esto corrige la distorsion para que las flechas apunten bien
    cos_lat = np.cos(np.deg2rad(lat2d_merc))
    cos_lat = np.clip(cos_lat, 0.1, None)
    v_merc_scaled = v_merc / cos_lat

    # Thinning
    lon2d_plot, merc2d_plot = np.meshgrid(lons, merc_y_uniform)
    u_thin = u_merc[::thin, ::thin]
    v_thin = v_merc_scaled[::thin, ::thin]
    lon_thin = lon2d_plot[::thin, ::thin]
    merc_thin = merc2d_plot[::thin, ::thin]

    # Figure - alta resolucion
    merc_height = merc_y_max - merc_y_min
    lon_width = float(lons.max() - lons.min())
    aspect = merc_height / lon_width if lon_width > 0 else 1.0
    height_px = max(int(width_px * aspect), 100)

    dpi = 150
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # En web el sistema de coordenadas (grados x Mercator) es distinto a Cartopy,
    # factores configurables en config.yaml: web_scale_factor, web_width_factor
    web_scale = wv_cfg.scale * wv_cfg.web_scale_factor
    web_width = wv_cfg.width * wv_cfg.web_width_factor
    ax.quiver(
        lon_thin, merc_thin, u_thin, v_thin,
        angles="xy",
        scale=web_scale,
        scale_units="width",
        width=web_width,
        color=wv_cfg.color,
        alpha=wv_cfg.alpha,
        headwidth=4, headlength=5, headaxislength=4,
    )
    ax.set_xlim(float(lons.min()), float(lons.max()))
    ax.set_ylim(merc_y_min, merc_y_max)
    ax.set_aspect("auto")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _empty_png() -> bytes:
    """Genera un PNG transparente 1x1 (para respuestas vacias sin error 404)."""
    fig = plt.figure(figsize=(1, 1), dpi=1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@app.get("/api/precipitation/status")
async def get_precipitation_status():
    """Devuelve si hay datos de precipitacion disponibles."""
    if state.ds is None:
        return {"available": False, "reason": "No hay datos cargados"}
    if "tp" not in state.ds:
        return {"available": False, "reason": "Sin datos de precipitacion. Cargue con step > 0 (ej: T+024)"}
    tp_vals = state.ds["tp"].values
    while tp_vals.ndim > 2:
        tp_vals = tp_vals[0]
    tp_max = float(np.nanmax(tp_vals))
    tp_mm = tp_max * 1000.0
    logger.info("Precipitacion status: tp_max=%.6f m (%.2f mm)", tp_max, tp_mm)
    if tp_mm <= state.cfg.precipitation.threshold_mm:
        return {"available": False, "reason": f"Precipitacion maxima = {tp_mm:.1f} mm (bajo umbral {state.cfg.precipitation.threshold_mm} mm). Use step mayor."}
    return {"available": True, "tp_max_mm": round(tp_mm, 2)}


@app.get("/api/precipitation/image")
async def get_precipitation_image():
    """Devuelve PNG transparente con precipitacion acumulada."""
    if state.ds is None:
        raise HTTPException(400, "No hay datos cargados.")

    png_bytes = _render_precipitation_png(
        state.ds, state.lats, state.lons, state.cfg,
    )
    if not png_bytes:
        logger.info("Precipitacion: sin datos significativos (tp no disponible o < umbral)")
        png_bytes = _empty_png()

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/api/wind_vectors/{level}/image")
async def get_wind_vectors_image(level: int):
    """Devuelve PNG transparente con vectores de viento a un nivel dado (850, 700)."""
    if state.ds is None:
        raise HTTPException(400, "No hay datos cargados.")
    if level not in (500, 700, 850):
        raise HTTPException(400, f"Nivel no soportado: {level}. Use 500, 700 o 850.")

    png_bytes = _render_wind_vectors_png(
        state.ds, state.lats, state.lons, level, state.cfg,
    )
    if not png_bytes:
        raise HTTPException(404, f"No hay datos de viento a {level} hPa.")

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/api/wind_vectors/{level}/data")
async def get_wind_vectors_data(level: int):
    """Devuelve datos de viento thinned como JSON para dibujar en canvas Leaflet."""
    if state.ds is None:
        raise HTTPException(400, "No hay datos cargados.")
    if level not in (500, 700, 850):
        raise HTTPException(400, f"Nivel no soportado: {level}. Use 500, 700 o 850.")

    u_name, v_name = f"u{level}", f"v{level}"
    if u_name not in state.ds or v_name not in state.ds:
        raise HTTPException(404, f"No hay datos de viento a {level} hPa.")

    wv_cfg = state.cfg.wind_vectors
    thin = wv_cfg.thin_factor

    u = state.ds[u_name].values.copy()
    v = state.ds[v_name].values.copy()
    while u.ndim > 2:
        u = u[0]
    while v.ndim > 2:
        v = v[0]

    # m/s a nudos
    u_kt = u * 1.94384
    v_kt = v * 1.94384

    lats = state.lats
    lons = state.lons

    # Thinning
    u_thin = u_kt[::thin, ::thin]
    v_thin = v_kt[::thin, ::thin]
    lat_thin = lats[::thin]
    lon_thin = lons[::thin]

    # Construir lista plana de puntos
    lon2d, lat2d = np.meshgrid(lon_thin, lat_thin)
    mask = np.isfinite(u_thin) & np.isfinite(v_thin)

    return JSONResponse({
        "lat": lat2d[mask].round(2).tolist(),
        "lon": lon2d[mask].round(2).tolist(),
        "u": u_thin[mask].round(1).tolist(),
        "v": v_thin[mask].round(1).tolist(),
        "color": wv_cfg.color,
        "alpha": wv_cfg.alpha,
    })


@app.get("/api/isobars")
async def get_isobars():
    """Devuelve isobaras como GeoJSON LineStrings."""
    if state.msl_smooth is None:
        raise HTTPException(400, "No hay datos cargados.")

    if state.isobar_geojson is not None:
        return state.isobar_geojson

    state.isobar_geojson = _extract_isobar_geojson()
    return state.isobar_geojson


@app.get("/api/centers")
async def get_centers():
    """Devuelve centros de presion como JSON."""
    if state.ds is None:
        raise HTTPException(400, "No hay datos cargados.")

    pc_cfg = state.cfg.pressure_centers

    def _center_label(c):
        base = pc_cfg.high_label if c.type == "H" else pc_cfg.low_label
        # Borrascas > 1005 hPa: minuscula (débiles)
        if c.type == "L" and c.value > 1005:
            return base.lower()
        return base.upper() if c.primary else base.lower()

    return [
        {
            "id": c.id,
            "type": c.type,
            "lat": c.lat,
            "lon": c.lon,
            "value": round(c.value, 1),
            "primary": c.primary,
            "name": c.name,
            "label": _center_label(c),
        }
        for c in state.centers
    ]


@app.get("/api/fronts/detect")
async def detect_fronts():
    """Ejecuta deteccion automatica TFP + clasificacion."""
    if state.ds is None:
        raise HTTPException(400, "No hay datos cargados.")

    from mapa_frentes.fronts.tfp import compute_tfp_fronts
    from mapa_frentes.fronts.classifier import classify_fronts
    from mapa_frentes.fronts.association import (
        associate_fronts_to_centers,
        filter_fronts_near_lows,
    )

    collection = compute_tfp_fronts(state.ds, state.cfg)
    # Asociar a centros ANTES de clasificar (el clasificador necesita los centros)
    if state.centers:
        collection = associate_fronts_to_centers(
            collection, state.centers, state.cfg,
        )
        # Filtrar frentes no asociados a borrascas
        collection = filter_fronts_near_lows(
            collection, state.centers, state.cfg,
        )
    collection = classify_fronts(collection, state.ds, state.cfg, centers=state.centers)
    state.fronts = collection
    return collection_to_geojson(collection)


@app.get("/api/fronts")
async def get_fronts():
    """Devuelve frentes actuales como GeoJSON."""
    return collection_to_geojson(state.fronts)


@app.put("/api/fronts")
async def save_fronts(request: Request):
    """Guarda frentes editados desde el cliente."""
    geojson = await request.json()
    state.fronts = collection_from_geojson(geojson)
    return {"status": "ok", "n_fronts": len(state.fronts)}


@app.get("/api/coastlines")
async def get_coastlines():
    """Devuelve costas y fronteras como GeoJSON (Natural Earth 50m)."""
    if hasattr(state, "_coastline_geojson") and state._coastline_geojson:
        return state._coastline_geojson

    import cartopy.feature as cfeature

    features = []

    def _extract_lines(source, layer_name):
        for geom in source.geometries():
            if geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": coords},
                            "properties": {"layer": layer_name},
                        })
            elif geom.geom_type == "LineString":
                coords = list(geom.coords)
                if len(coords) >= 2:
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": {"layer": layer_name},
                    })

    _extract_lines(
        cfeature.NaturalEarthFeature("physical", "coastline", "50m"),
        "coastline",
    )
    _extract_lines(
        cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "50m"),
        "border",
    )

    state._coastline_geojson = {"type": "FeatureCollection", "features": features}
    return state._coastline_geojson


@app.post("/api/fronts/generate/{center_id}")
async def generate_from_center(center_id: str):
    """Genera frentes desde un centro de presion especifico."""
    if state.ds is None:
        raise HTTPException(400, "No hay datos cargados.")

    center = next((c for c in state.centers if c.id == center_id), None)
    if center is None:
        raise HTTPException(404, f"Centro {center_id} no encontrado")

    from mapa_frentes.fronts.center_fronts import generate_fronts_from_center
    new_fronts = generate_fronts_from_center(center, state.ds, state.cfg)
    for f in new_fronts:
        state.fronts.add(f)

    return collection_to_geojson(state.fronts)


@app.get("/api/export")
async def export_map_png(
    field: str = Query("none"),
    clean: bool = Query(False),
    wind_levels: list[int] = Query([], description="Niveles de viento: 850, 700, 500"),
    precip: bool = Query(False, description="Incluir precipitacion"),
):
    """Exporta mapa en proyeccion Lambert (PNG 300 DPI) con simbologia WMO.

    - field: campo de fondo (o "none")
    - clean: si True, omite campo de fondo (solo basemap + isobaras + centros + frentes)
    - wind_levels: niveles de viento a dibujar
    - precip: incluir overlay de precipitacion
    """
    if state.ds is None:
        raise HTTPException(400, "No hay datos cargados.")

    from mapa_frentes.plotting.map_canvas import create_map_figure
    from mapa_frentes.plotting.isobar_renderer import (
        draw_isobars, draw_pressure_labels, draw_background_field,
        draw_precipitation, draw_wind_vectors,
    )
    from mapa_frentes.plotting.front_renderer import draw_fronts

    fig, ax = create_map_figure(state.cfg)

    # Campo de fondo (antes de isobaras para que quede debajo)
    if not clean and field != "none" and field in AVAILABLE_FIELDS:
        derived = _get_derived(field)
        if derived is not None:
            draw_background_field(ax, derived, state.lons, state.lats, state.cfg)

    # Precipitacion
    if precip:
        draw_precipitation(ax, state.ds, state.lons, state.lats, state.cfg)

    # Isobaras
    if state.msl_smooth is not None:
        draw_isobars(ax, state.msl_smooth, state.lons, state.lats, state.levels, state.cfg)

    # Centros de presion
    if state.centers:
        draw_pressure_labels(ax, state.centers, state.cfg)

    # Vectores de viento opcionales
    for wl in wind_levels:
        if wl in (500, 700, 850):
            draw_wind_vectors(ax, state.ds, state.lons, state.lats, wl, state.cfg)

    # Frentes con simbolos WMO (MetPy)
    # skip_orient=True: el usuario ya ajustó la dirección con flip en la web
    if len(state.fronts) > 0:
        draw_fronts(ax, state.fronts, state.cfg, skip_orient=True)

    # Titulo con fecha
    if state.date_info:
        ax.set_title(state.date_info, fontsize=10, loc="right", color="#555")

    # Renderizar a PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    filename = "mapa_frentes"
    if not clean and field != "none":
        filename += f"_{field}"
    filename += ".png"

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/mosaic")
async def export_mosaic_png(
    cols: int = Query(3),
    rows: int = Query(2),
    fields: list[str] = Query([]),
    wind_levels: list[int] = Query([], description="Niveles de viento: 850, 700, 500"),
    precip: bool = Query(False, description="Incluir precipitacion"),
):
    """Exporta mosaico completo en proyeccion Lambert (PNG 300 DPI).

    Cada panel tiene su campo de fondo + isobaras + centros + frentes.
    """
    if state.ds is None:
        raise HTTPException(400, "No hay datos cargados.")

    from mapa_frentes.plotting.map_canvas import build_projection, apply_base_cartography
    from mapa_frentes.plotting.isobar_renderer import (
        draw_isobars, draw_pressure_labels, draw_background_field,
        draw_precipitation, draw_wind_vectors,
    )
    from mapa_frentes.plotting.front_renderer import draw_fronts

    n_panels = cols * rows
    # Rellenar con "none" si faltan campos
    panel_fields = list(fields)
    while len(panel_fields) < n_panels:
        panel_fields.append("none")

    projection = build_projection(state.cfg)
    fig_w = 6 * cols
    fig_h = 4.5 * rows
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": projection},
    )
    if rows == 1 and cols == 1:
        all_axes = [axes]
    elif rows == 1 or cols == 1:
        all_axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    else:
        all_axes = axes.flatten().tolist()

    for i, ax in enumerate(all_axes):
        if i >= n_panels:
            ax.set_visible(False)
            continue

        field_name = panel_fields[i]
        apply_base_cartography(ax, state.cfg, lightweight=False)

        # Campo de fondo
        if field_name != "none" and field_name in AVAILABLE_FIELDS:
            derived = _get_derived(field_name)
            if derived is not None:
                draw_background_field(ax, derived, state.lons, state.lats, state.cfg)

        # Precipitacion
        if precip:
            draw_precipitation(ax, state.ds, state.lons, state.lats, state.cfg)

        # Isobaras
        if state.msl_smooth is not None:
            draw_isobars(ax, state.msl_smooth, state.lons, state.lats, state.levels, state.cfg)

        # Centros
        if state.centers:
            draw_pressure_labels(ax, state.centers, state.cfg)

        # Vectores de viento
        for wl in wind_levels:
            if wl in (500, 700, 850):
                draw_wind_vectors(ax, state.ds, state.lons, state.lats, wl, state.cfg)

        # Frentes
        if len(state.fronts) > 0:
            draw_fronts(ax, state.fronts, state.cfg, skip_orient=True)

        # Titulo del panel
        label = AVAILABLE_FIELDS.get(field_name, field_name) if field_name != "none" else "Base"
        ax.set_title(label, fontsize=8, pad=3)

    # Titulo general
    if state.date_info:
        fig.suptitle(state.date_info, fontsize=11, color="#555", y=0.99)

    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.94, bottom=0.02,
        wspace=0.05, hspace=0.10,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    filename = f"mosaico_{cols}x{rows}.png"
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Static files y arranque
# ---------------------------------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)
