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
    """Calcula vmin/vmax del campo restringido al viewport visible."""
    if south is None or north is None or west is None or east is None:
        return float(np.nanmin(data)), float(np.nanmax(data))

    lat_mask = (lats >= south) & (lats <= north)
    lon_mask = (lons >= west) & (lons <= east)

    if not np.any(lat_mask) or not np.any(lon_mask):
        return float(np.nanmin(data)), float(np.nanmax(data))

    subset = data[np.ix_(lat_mask, lon_mask)]
    if subset.size == 0 or np.all(np.isnan(subset)):
        return float(np.nanmin(data)), float(np.nanmax(data))

    return float(np.nanmin(subset)), float(np.nanmax(subset))


def _render_field_png(
    derived: DerivedField,
    lats: np.ndarray,
    lons: np.ndarray,
    width_px: int = 900,
    vmin: float | None = None,
    vmax: float | None = None,
) -> bytes:
    """Renderiza un DerivedField como PNG transparente sin basemap.

    El PNG cubre exactamente [lons.min(), lons.max()] x [lats.min(), lats.max()]
    para alinearse con L.imageOverlay en Leaflet.
    Si se pasan vmin/vmax, se usan para la escala de color.
    """
    aspect = len(lats) / len(lons)
    height_px = int(width_px * aspect)
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    lon2d, lat2d = np.meshgrid(lons, lats)
    data = derived.data
    num_levels = 20

    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))

    if derived.center_zero:
        abs_max = max(abs(vmin), abs(vmax))
        levels = np.linspace(-abs_max, abs_max, num_levels)
        norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        levels = np.linspace(vmin, vmax, num_levels)
        norm = None

    ax.contourf(
        lon2d, lat2d, data,
        levels=levels, cmap=derived.cmap, norm=norm,
        alpha=0.7, extend="both",
    )
    ax.set_xlim(float(lons.min()), float(lons.max()))
    ax.set_ylim(float(lats.min()), float(lats.max()))
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
    return [
        {
            "id": c.id,
            "type": c.type,
            "lat": c.lat,
            "lon": c.lon,
            "value": round(c.value, 1),
            "primary": c.primary,
            "name": c.name,
            "label": pc_cfg.high_label if c.type == "H" else pc_cfg.low_label,
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
    from mapa_frentes.fronts.association import associate_fronts_to_centers

    collection = compute_tfp_fronts(state.ds, state.cfg)
    collection = classify_fronts(collection, state.ds, state.cfg)
    if state.centers:
        collection = associate_fronts_to_centers(
            collection, state.centers, state.cfg,
        )
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
