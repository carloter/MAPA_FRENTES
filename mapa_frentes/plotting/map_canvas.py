"""Figura Cartopy base: proyeccion, costas, bordes."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from mapa_frentes.config import AppConfig


def build_projection(cfg: AppConfig) -> ccrs.Projection:
    """Construye la proyeccion Cartopy a partir de la configuracion.

    Soporta LambertConformal (estilo AEMET) y PlateCarree como fallback.
    """
    proj_name = cfg.plotting.projection
    params = cfg.plotting.projection_params

    if proj_name == "LambertConformal":
        std_parallels = params.get("standard_parallels", [30.0, 60.0])
        return ccrs.LambertConformal(
            central_longitude=params.get("central_longitude", -15.0),
            central_latitude=params.get("central_latitude", 45.0),
            standard_parallels=std_parallels,
        )

    # Fallback: PlateCarree
    return ccrs.PlateCarree()


def create_map_figure(cfg: AppConfig) -> tuple[Figure, plt.Axes]:
    """Crea una figura Matplotlib con Cartopy para el mapa de frentes.

    Estilo inspirado en los mapas AEMET: fondo suave, costas detalladas,
    fronteras punteadas, gridlines cada 10 grados.
    """
    plot_cfg = cfg.plotting
    area = cfg.area

    projection = build_projection(cfg)
    fig = plt.figure(figsize=plot_cfg.figsize, dpi=plot_cfg.dpi)
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.set_extent(
        [area.lon_min, area.lon_max, area.lat_min, area.lat_max],
        crs=ccrs.PlateCarree(),
    )

    # Fondo: colores suaves tipo AEMET
    ax.add_feature(
        cfeature.OCEAN, facecolor=plot_cfg.ocean_color, zorder=0
    )
    ax.add_feature(
        cfeature.LAND, facecolor=plot_cfg.land_color, zorder=0
    )

    # Lagos para mayor detalle visual
    try:
        ax.add_feature(
            cfeature.LAKES.with_scale("50m"),
            facecolor=plot_cfg.ocean_color,
            edgecolor=plot_cfg.coastline_color,
            linewidth=0.3,
            zorder=0,
        )
    except Exception:
        pass

    # Costas con resolucion 50m
    try:
        ax.add_feature(
            cfeature.COASTLINE.with_scale("50m"),
            edgecolor=plot_cfg.coastline_color,
            linewidth=plot_cfg.coastline_linewidth,
            zorder=1,
        )
    except Exception:
        ax.add_feature(
            cfeature.COASTLINE,
            edgecolor=plot_cfg.coastline_color,
            linewidth=plot_cfg.coastline_linewidth,
            zorder=1,
        )

    # Fronteras punteadas
    ax.add_feature(
        cfeature.BORDERS,
        edgecolor=plot_cfg.border_color,
        linewidth=plot_cfg.border_linewidth,
        linestyle=":",
        zorder=1,
    )

    # Gridlines cada 10 grados
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.4,
        linestyle="--",
        x_inline=False,
        y_inline=False,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7, "color": "gray"}
    gl.ylabel_style = {"size": 7, "color": "gray"}
    gl.xlocator = MultipleLocator(10)
    gl.ylocator = MultipleLocator(10)

    return fig, ax
