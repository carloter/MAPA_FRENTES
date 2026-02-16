"""Renderizado de isobaras, etiquetas A/B, campos de fondo y vectores de viento."""

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mapa_frentes.config import AppConfig
from mapa_frentes.analysis.pressure_centers import PressureCenter
from mapa_frentes.analysis.derived_fields import DerivedField


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
    """Dibuja etiquetas de centros de presion estilo AEMET.

    - Primarios: B/A mayuscula, tamano grande
    - Secundarios: b/a minuscula, tamano menor
    - Si tiene nombre (borrasca nombrada): se muestra junto a la etiqueta
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
        base_label = pc_cfg.high_label if center.type == "H" else pc_cfg.low_label

        # Borrascas > 1005 hPa: minuscula (débiles)
        if center.type == "L" and center.value > 1005:
            label = base_label.lower()
            fontsize = pc_cfg.fontsize - 2
        elif center.primary:
            label = base_label.upper()
            fontsize = pc_cfg.fontsize + 2
        else:
            label = base_label.lower()
            fontsize = pc_cfg.fontsize - 2

        # Letra A/B (o a/b) con halo blanco
        ax.text(
            center.lon, center.lat,
            label,
            transform=ccrs.PlateCarree(),
            fontsize=fontsize,
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

        # Nombre de la borrasca (si tiene)
        if center.name:
            ax.text(
                center.lon + 1.0, center.lat,
                center.name,
                transform=ccrs.PlateCarree(),
                fontsize=pc_cfg.fontsize - 2,
                fontweight="bold",
                fontstyle="italic",
                color="black",
                ha="left",
                va="center",
                zorder=7,
            )

        # Valor de presion debajo
        value_fontsize = pc_cfg.fontsize - 3 if center.primary else pc_cfg.fontsize - 5
        ax.text(
            center.lon, center.lat - 1.4,
            f"{center.value:.0f}",
            transform=ccrs.PlateCarree(),
            fontsize=value_fontsize,
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


def draw_background_field(
    ax: Axes,
    derived: DerivedField,
    lons: np.ndarray,
    lats: np.ndarray,
    cfg: AppConfig,
) -> None:
    """Dibuja un campo derivado como fondo sombreado (contourf).

    IMPORTANTE: la colorbar se dibuja en un "inset axes" para NO
    reducir el tamaño del mapa (evita el shrink del axes principal).

    Args:
        ax: Axes de matplotlib con proyeccion cartopy.
        derived: DerivedField con datos, label, units, cmap, center_zero.
        lons: Array 1D de longitudes.
        lats: Array 1D de latitudes.
        cfg: Configuracion de la app.
    """
    bg_cfg = cfg.background_field
    lon2d, lat2d = np.meshgrid(lons, lats)

    data = derived.data
    num_levels = bg_cfg.num_levels

    # Calcular niveles de contorno
    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))

    # Evitar crash si todos los valores son iguales
    if vmin >= vmax:
        return

    if derived.center_zero:
        # Simetrico alrededor de 0
        abs_max = max(abs(vmin), abs(vmax))
        levels = np.linspace(-abs_max, abs_max, num_levels)
        norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        levels = np.linspace(vmin, vmax, num_levels)
        norm = None

    cf = ax.contourf(
        lon2d, lat2d, data,
        levels=levels,
        cmap=derived.cmap,
        norm=norm,
        alpha=bg_cfg.alpha,
        transform=ccrs.PlateCarree(),
        zorder=1.5,
        extend="both",
    )

    if not bg_cfg.colorbar:
        return

    fig = ax.get_figure()

    # --- Eliminar colorbar anterior de ESTE axes si existe ---
    if hasattr(ax, "_bg_cbar") and ax._bg_cbar is not None:
        try:
            ax._bg_cbar.remove()
        except Exception:
            pass
        ax._bg_cbar = None

    # --- Crear eje inset para la colorbar (NO encoge el axes principal) ---
    cax = inset_axes(
        ax,
        width="2.1%",
        height="92%",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.04, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )

    ax._bg_cbar = fig.colorbar(cf, cax=cax, orientation="vertical")

    ax._bg_cbar.set_label(f"{derived.label} ({derived.units})", fontsize=7)
    ax._bg_cbar.ax.tick_params(labelsize=6, length=3)

    try:
        ax._bg_cbar.ax.yaxis.set_major_formatter("{x:.0f}")
    except Exception:
        pass


def draw_precipitation(
    ax: Axes,
    ds: xr.Dataset,
    lons: np.ndarray,
    lats: np.ndarray,
    cfg: AppConfig,
) -> None:
    """Dibuja precipitacion como overlay transparente.

    Valores por debajo del umbral configurado son completamente transparentes.
    """
    from mapa_frentes.analysis.derived_fields import compute_precipitation

    precip_mm = compute_precipitation(ds)
    if precip_mm is None:
        return

    vmax = float(np.nanmax(precip_mm))
    pcfg = cfg.precipitation
    if vmax <= pcfg.threshold_mm:
        return  # nada que dibujar

    lon2d, lat2d = np.meshgrid(lons, lats)

    # Niveles desde el umbral hasta el maximo
    levels = np.linspace(pcfg.threshold_mm, vmax, pcfg.num_levels)

    ax.contourf(
        lon2d, lat2d, precip_mm,
        levels=levels,
        cmap=pcfg.cmap,
        alpha=pcfg.alpha,
        transform=ccrs.PlateCarree(),
        zorder=1.8,
        extend="max",
    )


def draw_wind_vectors(
    ax: Axes,
    ds: xr.Dataset,
    lons: np.ndarray,
    lats: np.ndarray,
    level: int,
    cfg: AppConfig,
) -> None:
    """Dibuja vectores de viento (quiver) a un nivel de presion dado.

    Args:
        ax: Axes de matplotlib con proyeccion cartopy.
        ds: Dataset con variables u{level} y v{level}.
        lons: Array 1D de longitudes.
        lats: Array 1D de latitudes.
        level: Nivel de presion en hPa (850, 700, 500).
        cfg: Configuracion de la app.
    """
    u_name = f"u{level}"
    v_name = f"v{level}"
    if u_name not in ds or v_name not in ds:
        return

    wv_cfg = cfg.wind_vectors
    thin = wv_cfg.thin_factor

    u = ds[u_name].values
    v = ds[v_name].values
    # Eliminar dimensiones extra (tiempo, etc.)
    while u.ndim > 2:
        u = u[0]
    while v.ndim > 2:
        v = v[0]

    # Convertir a nudos
    u_kt = u * 1.94384
    v_kt = v * 1.94384

    # Thinning: cada N puntos
    lon2d, lat2d = np.meshgrid(lons, lats)
    u_thin = u_kt[::thin, ::thin]
    v_thin = v_kt[::thin, ::thin]
    lon_thin = lon2d[::thin, ::thin]
    lat_thin = lat2d[::thin, ::thin]

    q = ax.quiver(
        lon_thin, lat_thin, u_thin, v_thin,
        transform=ccrs.PlateCarree(),
        scale=wv_cfg.scale,
        width=wv_cfg.width,
        color=wv_cfg.color,
        alpha=wv_cfg.alpha,
        zorder=3,
    )

    # Leyenda de escala
    ax.quiverkey(
        q, 0.92, 0.02, 20,
        "20 kt",
        labelpos="W",
        coordinates="axes",
        fontproperties={"size": 6},
        labelsep=0.03,
    )

