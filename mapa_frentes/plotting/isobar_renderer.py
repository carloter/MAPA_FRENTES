"""Renderizado de isobaras y etiquetas H/L en el mapa."""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.axes import Axes

from mapa_frentes.config import AppConfig
from mapa_frentes.analysis.pressure_centers import PressureCenter


def draw_isobars(
    ax: Axes,
    msl_smooth: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    levels: np.ndarray,
    cfg: AppConfig,
):
    """Dibuja las isobaras sobre el mapa.

    Isobaras normales en gris fino, isobaras cada 20 hPa mas gruesas.
    Etiquetas en todas las isobaras normales.
    """
    iso_cfg = cfg.isobars
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Isobaras normales
    cs = ax.contour(
        lon2d, lat2d, msl_smooth,
        levels=levels,
        colors=iso_cfg.color,
        linewidths=iso_cfg.linewidth,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.clabel(
        cs,
        inline=True,
        fontsize=iso_cfg.label_fontsize,
        fmt="%d",
    )

    # Isobaras maestras (cada 20 hPa) mas gruesas
    master_levels = levels[levels % 20 == 0]
    if len(master_levels) > 0:
        ax.contour(
            lon2d, lat2d, msl_smooth,
            levels=master_levels,
            colors=iso_cfg.color,
            linewidths=iso_cfg.linewidth * 2.0,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )


def draw_pressure_labels(
    ax: Axes,
    centers: list[PressureCenter],
    cfg: AppConfig,
):
    """Dibuja las etiquetas H/L con halo blanco, estilo AEMET.

    Las letras H/L son grandes y destacadas. El valor de presion
    aparece debajo en un recuadro blanco semitransparente.
    Solo se dibujan centros dentro del area de visualizacion.
    """
    pc_cfg = cfg.pressure_centers
    area = cfg.area  # Area de visualizacion (sin padding)

    for center in centers:
        # Filtrar centros fuera del area visible
        if (center.lon < area.lon_min or center.lon > area.lon_max
                or center.lat < area.lat_min or center.lat > area.lat_max):
            continue
        color = pc_cfg.h_color if center.type == "H" else pc_cfg.l_color

        # Letra H/L grande con halo blanco
        ax.text(
            center.lon, center.lat,
            center.type,
            transform=ccrs.PlateCarree(),
            fontsize=pc_cfg.fontsize + 2,
            fontweight=pc_cfg.fontweight,
            color=color,
            ha="center",
            va="center",
            zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
            ),
        )
        # Valor de presion debajo
        ax.text(
            center.lon, center.lat - 1.4,
            f"{center.value:.0f}",
            transform=ccrs.PlateCarree(),
            fontsize=pc_cfg.fontsize - 3,
            fontweight="bold",
            color=color,
            ha="center",
            va="top",
            zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
            ),
        )
