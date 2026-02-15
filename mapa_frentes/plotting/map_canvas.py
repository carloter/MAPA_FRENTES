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


def apply_base_cartography(ax, cfg: AppConfig, lightweight: bool = False):
    """Aplica cartografia base a un GeoAxes.

    Args:
        ax: GeoAxes de Cartopy.
        cfg: Configuracion.
        lightweight: Si True, usa resolucion 110m (para mosaico).
    """
    plot_cfg = cfg.plotting
    area = cfg.area

    ax.set_extent(
        [area.lon_min, area.lon_max, area.lat_min, area.lat_max],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.OCEAN, facecolor=plot_cfg.ocean_color, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=plot_cfg.land_color, zorder=0)

    scale = "110m" if lightweight else "50m"
    try:
        ax.add_feature(
            cfeature.COASTLINE.with_scale(scale),
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

    ax.add_feature(
        cfeature.BORDERS,
        edgecolor=plot_cfg.border_color,
        linewidth=plot_cfg.border_linewidth,
        linestyle=":",
        zorder=1,
    )


def create_mosaic_figure(
    cfg: AppConfig,
    field_names: list[str],
    nrows: int = 2,
    ncols: int = 3,
) -> tuple[Figure, dict]:
    """Crea figura con NxM subplots GeoAxes para vista mosaico.

    Args:
        cfg: Configuracion.
        field_names: Lista de nombres de campo (uno por panel).
        nrows: Numero de filas.
        ncols: Numero de columnas.

    Returns:
        fig: Figure.
        axes_dict: dict {field_name: GeoAxes} en orden.
    """
    projection = build_projection(cfg)
    fig, axes_grid = plt.subplots(
        nrows, ncols,
        figsize=(cfg.plotting.figsize[0], cfg.plotting.figsize[1]),
        dpi=max(cfg.plotting.dpi // 2, 72),  # menor DPI para rendimiento
        subplot_kw={"projection": projection},
    )

    # Aplanar grid de axes
    if nrows == 1 and ncols == 1:
        all_axes = [axes_grid]
    else:
        all_axes = axes_grid.flatten()

    axes_dict = {}
    for i, ax in enumerate(all_axes):
        if i < len(field_names):
            name = field_names[i]
            axes_dict[name] = ax
            apply_base_cartography(ax, cfg, lightweight=True)
        else:
            ax.set_visible(False)

    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.95, bottom=0.02,
        wspace=0.05, hspace=0.08,
    )

    return fig, axes_dict
