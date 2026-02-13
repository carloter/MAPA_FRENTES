"""Lee ficheros GRIB2 a xarray Dataset usando cfgrib.

El ECMWF IFS Open Data SCDA empaqueta todas las variables (superficie
y niveles de presion) en un mismo bundle GRIB. Usamos filter_by_keys
de cfgrib para extraer solo lo que necesitamos en cada caso.
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from mapa_frentes.config import AppConfig, AreaConfig

logger = logging.getLogger(__name__)


def _compute_data_area(cfg: AppConfig) -> AreaConfig:
    """Calcula el bounding box geografico necesario para cubrir toda el
    area visible de la proyeccion Lambert (incluidas las esquinas).

    Con Lambert Conformal, el rectangulo en coordenadas proyectadas
    se convierte en un trapecio curvo en coordenadas geograficas.
    Las esquinas del mapa proyectado pueden extenderse mucho mas alla
    del rectangulo lon/lat de visualizacion. Esta funcion calcula
    exactamente que area geografica se necesita.
    """
    import cartopy.crs as ccrs

    area = cfg.area
    proj_name = cfg.plotting.projection

    if proj_name != "LambertConformal":
        return area.padded()

    # Construir la proyeccion
    params = cfg.plotting.projection_params
    proj = ccrs.LambertConformal(
        central_longitude=params.get("central_longitude", -15.0),
        central_latitude=params.get("central_latitude", 45.0),
        standard_parallels=params.get("standard_parallels", [30.0, 60.0]),
    )
    geo = ccrs.PlateCarree()

    # 1. Muestrear densamente el borde del area de visualizacion en coords geo
    n = 200
    lons_bottom = np.linspace(area.lon_min, area.lon_max, n)
    lons_top = np.linspace(area.lon_min, area.lon_max, n)
    lats_left = np.linspace(area.lat_min, area.lat_max, n)
    lats_right = np.linspace(area.lat_min, area.lat_max, n)

    edge_lons = np.concatenate([
        lons_bottom,                        # borde inferior
        np.full(n, area.lon_max),           # borde derecho
        lons_top,                           # borde superior
        np.full(n, area.lon_min),           # borde izquierdo
    ])
    edge_lats = np.concatenate([
        np.full(n, area.lat_min),           # borde inferior
        lats_right,                         # borde derecho
        np.full(n, area.lat_max),           # borde superior
        lats_left,                          # borde izquierdo
    ])

    # 2. Transformar bordes a coordenadas proyectadas
    pts_proj = proj.transform_points(geo, edge_lons, edge_lats)
    valid = np.isfinite(pts_proj[:, 0]) & np.isfinite(pts_proj[:, 1])
    x_min = pts_proj[valid, 0].min()
    x_max = pts_proj[valid, 0].max()
    y_min = pts_proj[valid, 1].min()
    y_max = pts_proj[valid, 1].max()

    # 3. Muestrear el bounding box PROYECTADO (rectangular en Lambert)
    #    y transformar de vuelta a coordenadas geograficas
    bx = np.concatenate([
        np.linspace(x_min, x_max, n),  # inferior
        np.full(n, x_max),             # derecha
        np.linspace(x_max, x_min, n),  # superior
        np.full(n, x_min),             # izquierda
    ])
    by = np.concatenate([
        np.full(n, y_min),             # inferior
        np.linspace(y_min, y_max, n),  # derecha
        np.full(n, y_max),             # superior
        np.linspace(y_max, y_min, n),  # izquierda
    ])

    pts_geo = geo.transform_points(proj, bx, by)
    valid = np.isfinite(pts_geo[:, 0]) & np.isfinite(pts_geo[:, 1])

    # 4. Bounding box geografico + margen de seguridad
    margin = 2.0
    data_area = AreaConfig(
        lon_min=max(float(pts_geo[valid, 0].min()) - margin, -180.0),
        lon_max=min(float(pts_geo[valid, 0].max()) + margin, 180.0),
        lat_min=max(float(pts_geo[valid, 1].min()) - margin, -90.0),
        lat_max=min(float(pts_geo[valid, 1].max()) + margin, 90.0),
        data_padding_deg=0.0,
    )

    logger.info(
        "Area datos (Lambert): lon=[%.1f, %.1f] lat=[%.1f, %.1f]",
        data_area.lon_min, data_area.lon_max,
        data_area.lat_min, data_area.lat_max,
    )
    return data_area


def read_grib_files(
    grib_paths: dict[str, Path],
    cfg: AppConfig,
) -> xr.Dataset:
    """Lee ficheros GRIB2 y combina en un unico Dataset.

    Usa filter_by_keys para separar correctamente:
    - MSL de las variables de superficie
    - t, q, u, v del nivel de presion 850 hPa

    Args:
        grib_paths: Dict con claves 'surface' y 'pressure' apuntando a GRIB2.
        cfg: Configuracion de la aplicacion.

    Returns:
        xr.Dataset con variables: msl, t850, q850, u850, v850
        recortado al area de interes.
    """
    area = _compute_data_area(cfg)  # Area calculada para cubrir proyeccion Lambert
    level = cfg.ecmwf.pressure_level  # 850

    # --- 1. Leer MSL (mean sea level pressure) ---
    sfc_path = grib_paths["surface"]
    logger.info("Leyendo MSL de: %s", sfc_path)
    msl = _read_variable(sfc_path, shortName="msl")
    if msl is None:
        # Fallback: intentar con filtro por typeOfLevel
        msl = _read_variable(sfc_path, typeOfLevel="meanSea")
    if msl is None:
        raise RuntimeError(
            f"No se encontro variable 'msl' en {sfc_path}. "
            "Verifica que el GRIB contiene presion a nivel del mar."
        )
    logger.info("MSL leida: shape=%s", msl.shape)

    # --- 2. Leer variables a 850 hPa ---
    # Intentar primero del fichero de presion, luego del de superficie
    # (el SCDA puede meter todo en un solo fichero)
    pl_path = grib_paths["pressure"]
    pl_vars = {}

    for param in cfg.ecmwf.pressure_params:  # t, q, u, v
        var = _read_variable(pl_path, shortName=param, level=level,
                             typeOfLevel="isobaricInhPa")
        if var is None:
            # Probar en el fichero de superficie (mismo bundle SCDA)
            var = _read_variable(sfc_path, shortName=param, level=level,
                                 typeOfLevel="isobaricInhPa")
        if var is None:
            logger.warning("Variable '%s' a %d hPa no encontrada", param, level)
        else:
            pl_vars[param] = var
            logger.info(
                "%s a %d hPa: min=%.1f, max=%.1f",
                param, level, float(var.min()), float(var.max()),
            )

    if not pl_vars:
        raise RuntimeError(
            f"No se encontraron variables de presion a {level} hPa. "
            "El fichero GRIB puede no contener niveles isobaricos."
        )

    # --- 3. Construir Dataset combinado ---
    lat_name = _find_coord(msl, ["latitude", "lat"])
    lon_name = _find_coord(msl, ["longitude", "lon"])

    ds = xr.Dataset()
    ds["msl"] = msl

    var_mapping = {"t": "t850", "q": "q850", "u": "u850", "v": "v850"}
    for src, dst in var_mapping.items():
        if src in pl_vars:
            ds[dst] = pl_vars[src]

    # Recortar al area de interes
    ds = _crop_to_area(ds, lat_name, lon_name, area)

    # Convertir msl de Pa a hPa si es necesario
    msl_mean = float(np.nanmean(ds["msl"].values))
    if msl_mean > 50000:
        ds["msl"] = ds["msl"] / 100.0
        ds["msl"].attrs["units"] = "hPa"

    logger.info(
        "Dataset final: %s | shape lat=%d, lon=%d",
        list(ds.data_vars),
        ds.sizes.get(lat_name, 0),
        ds.sizes.get(lon_name, 0),
    )

    # Sanity checks
    if "t850" in ds:
        t_mean = float(np.nanmean(ds["t850"].values))
        logger.info("t850 media: %.1f K (debe ser ~260-280 K)", t_mean)
        if t_mean < 200 or t_mean > 330:
            logger.warning(
                "t850 tiene valores sospechosos (media=%.1f K). "
                "Verificar que se leyo del nivel correcto.", t_mean
            )

    return ds


def _read_variable(
    grib_path: Path,
    shortName: str | None = None,
    typeOfLevel: str | None = None,
    level: int | None = None,
) -> xr.DataArray | None:
    """Lee una variable especifica de un GRIB usando filter_by_keys.

    Returns:
        DataArray con la variable, o None si no se encuentra.
    """
    filter_keys = {}
    if shortName is not None:
        filter_keys["shortName"] = shortName
    if typeOfLevel is not None:
        filter_keys["typeOfLevel"] = typeOfLevel
    if level is not None:
        filter_keys["level"] = level

    try:
        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": filter_keys},
        )
        # Devolver la primera (y normalmente unica) variable de datos
        data_vars = list(ds.data_vars)
        if data_vars:
            return ds[data_vars[0]]
        return None
    except Exception as e:
        logger.debug(
            "No se pudo leer %s con filtro %s: %s",
            grib_path.name, filter_keys, e,
        )
        return None


def _find_coord(da: xr.DataArray, candidates: list[str]) -> str:
    """Encuentra el nombre de la coordenada entre candidatos."""
    for name in candidates:
        if name in da.dims or name in da.coords:
            return name
    raise KeyError(f"No se encontro coordenada entre {candidates}. "
                   f"Disponibles: {list(da.coords)}")


def _crop_to_area(ds, lat_name, lon_name, area):
    """Recorta el dataset al area de interes."""
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    # Manejar longitudes 0-360 vs -180-180
    if lons.max() > 180:
        ds = ds.assign_coords({lon_name: ((ds[lon_name] + 180) % 360) - 180})
        ds = ds.sortby(lon_name)

    # Determinar orden de latitudes
    if lats[0] > lats[-1]:
        lat_slice = slice(area.lat_max, area.lat_min)
    else:
        lat_slice = slice(area.lat_min, area.lat_max)

    lon_slice = slice(area.lon_min, area.lon_max)

    ds = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
    return ds
