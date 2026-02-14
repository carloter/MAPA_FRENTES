"""Script de prueba para las funcionalidades avanzadas de detección de frentes.

Prueba:
1. Descarga de datos multi-nivel
2. Detección robusta de oclusiones
3. Agrupado por sistemas ciclónicos
4. Ranking de frentes principales/secundarios
5. Visualización con filtrado
"""

import logging
import sys
from pathlib import Path

# Añadir el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt

from mapa_frentes.config import load_config
from mapa_frentes.data.ecmwf_download import download_ecmwf
from mapa_frentes.data.grib_reader import read_grib_files
from mapa_frentes.fronts.tfp import compute_tfp_fronts
from mapa_frentes.fronts.classifier import classify_fronts
from mapa_frentes.fronts.cyclone_systems import build_cyclone_systems
from mapa_frentes.fronts.ranking import rank_all_systems
from mapa_frentes.fronts.io import save_systems
from mapa_frentes.analysis.pressure_centers import detect_pressure_centers
from mapa_frentes.utils.smoothing import smooth_field
from mapa_frentes.plotting.map_canvas import create_map_figure
from mapa_frentes.plotting.front_renderer import draw_fronts, draw_front_legend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Pipeline completo de prueba."""
    logger.info("=== TEST: Funcionalidades Avanzadas ===")

    # 1. Cargar configuración
    logger.info("1. Cargando configuración...")
    cfg = load_config()

    # Verificar que está habilitada la detección multi-nivel
    if not cfg.occlusion.enabled:
        logger.warning("Detección de oclusiones deshabilitada. Habilitando...")
        cfg.occlusion.enabled = True

    logger.info("   - Niveles de presión: %s", cfg.ecmwf.pressure_levels)
    logger.info("   - Parámetros: %s", cfg.ecmwf.pressure_params)
    logger.info("   - Oclusión: habilitada=%s, min_score=%.2f",
                cfg.occlusion.enabled, cfg.occlusion.min_score)

    # 2. Descargar datos
    logger.info("2. Descargando datos ECMWF multi-nivel...")
    try:
        grib_paths = download_ecmwf(cfg, force=False)
        logger.info("   - Superficie: %s", grib_paths["surface"])
        logger.info("   - Presión: %s", grib_paths["pressure"])
    except Exception as e:
        logger.error("Error descargando datos: %s", e)
        logger.info("Continuando con datos cacheados si existen...")
        # Intentar usar cache
        from datetime import datetime
        date_tag = datetime.utcnow().strftime("%Y%m%d%H")
        cache_dir = Path(cfg.data.cache_dir)
        sfc_target = cache_dir / f"ecmwf_sfc_latest.grib2"
        pl_target = cache_dir / f"ecmwf_pl_multi_latest.grib2"

        if not sfc_target.exists() or not pl_target.exists():
            logger.error("No hay datos cacheados. Abortando.")
            return 1

        grib_paths = {"surface": sfc_target, "pressure": pl_target}

    # 3. Leer datos
    logger.info("3. Leyendo datos GRIB...")
    ds = read_grib_files(grib_paths, cfg)
    logger.info("   - Variables: %s", list(ds.data_vars))
    logger.info("   - Shape: lat=%d, lon=%d", ds.sizes.get("latitude", 0), ds.sizes.get("lon", 0))

    # Verificar datos multi-nivel
    has_500 = "t500" in ds
    has_700 = "t700" in ds
    has_850 = "t850" in ds
    has_vo = "vo850" in ds
    logger.info("   - Datos multi-nivel: 500hPa=%s, 700hPa=%s, 850hPa=%s, vo=%s",
                has_500, has_700, has_850, has_vo)

    if not (has_500 and has_700 and has_850):
        logger.warning("Faltan datos multi-nivel. Detección robusta limitada.")

    # 4. Detectar centros de presión
    logger.info("4. Detectando centros de presión...")
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    msl = ds["msl"].values
    msl_smooth = smooth_field(msl, sigma=cfg.isobars.smooth_sigma)

    centers = detect_pressure_centers(msl_smooth, lats, lons, cfg)
    low_centers = [c for c in centers if c.type == "L"]
    high_centers = [c for c in centers if c.type == "H"]
    logger.info("   - Centros L: %d (primarios: %d)",
                len(low_centers), sum(1 for c in low_centers if c.primary))
    logger.info("   - Centros H: %d (primarios: %d)",
                len(high_centers), sum(1 for c in high_centers if c.primary))

    # 5. Detectar frentes TFP
    logger.info("5. Detectando frentes TFP...")
    front_collection = compute_tfp_fronts(ds, cfg)
    logger.info("   - Frentes detectados: %d", len(front_collection))

    # 6. Clasificar frentes (con detección robusta de oclusiones)
    logger.info("6. Clasificando frentes (detección robusta de oclusiones)...")
    front_collection = classify_fronts(front_collection, ds, cfg, centers=low_centers)

    # Mostrar estadísticas de clasificación
    from mapa_frentes.fronts.models import FrontType
    stats = {}
    for ftype in FrontType:
        count = sum(1 for f in front_collection if f.front_type == ftype)
        if count > 0:
            stats[ftype.value] = count

    logger.info("   - Estadísticas de clasificación:")
    for ftype, count in stats.items():
        logger.info("     * %s: %d", ftype, count)

    # Mostrar frentes con scores de oclusión
    occluded_fronts = [f for f in front_collection
                      if f.front_type in (FrontType.OCCLUDED, FrontType.COLD_OCCLUDED,
                                         FrontType.WARM_OCCLUDED, FrontType.WARM_SECLUSION)]
    if occluded_fronts:
        logger.info("   - Oclusiones detectadas:")
        for front in occluded_fronts:
            logger.info("     * %s: tipo=%s, score=%.2f",
                       front.id[:12], front.front_type.value, front.occlusion_score)

    # 7. Construir sistemas ciclónicos
    logger.info("7. Construyendo sistemas ciclónicos...")
    cyclone_systems = build_cyclone_systems(front_collection, centers, cfg)
    logger.info("   - Sistemas detectados: %d", len(cyclone_systems))
    logger.info("   - Frentes no asociados: %d", len(cyclone_systems.unassociated_fronts))

    for system in cyclone_systems:
        logger.info("   - Sistema %s:", system.id)
        logger.info("     * Centro: %.1f°N, %.1f°E, %.0f hPa",
                   system.center.lat, system.center.lon, system.center.value)
        logger.info("     * Frentes: %d (primario: %s)",
                   len(system.fronts), system.is_primary)
        logger.info("     * Centros secundarios: %d", len(system.secondary_centers))

    # 8. Aplicar ranking
    logger.info("8. Aplicando ranking de importancia...")
    rank_all_systems(cyclone_systems, ds, cfg, threshold=0.60)

    # Mostrar frentes principales de cada sistema
    for system in cyclone_systems:
        primary_fronts = [f for f in system.fronts if f.is_primary]
        secondary_fronts = [f for f in system.fronts if not f.is_primary]
        logger.info("   - Sistema %s: %d primarios, %d secundarios",
                   system.id, len(primary_fronts), len(secondary_fronts))

        if primary_fronts:
            logger.info("     Frentes principales:")
            for front in primary_fronts[:3]:  # Top 3
                logger.info("       * %s: tipo=%s, score=%.2f",
                           front.id[:12], front.front_type.value, front.importance_score)

    # 9. Guardar sistemas
    logger.info("9. Guardando sistemas ciclónicos...")
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    systems_file = output_dir / "cyclone_systems.geojson"
    save_systems(cyclone_systems, systems_file)
    logger.info("   - Guardado en: %s", systems_file)

    # 10. Generar mapa de prueba
    logger.info("10. Generando mapa de visualización...")
    fig, ax = create_map_figure(cfg)

    # Dibujar todos los frentes con importancia
    draw_fronts(ax, front_collection, cfg, show_importance=True)
    draw_front_legend(ax, cfg)

    # Añadir título
    ax.set_title(
        f"Test: Detección Robusta de Oclusiones + Sistemas Ciclónicos\n"
        f"{len(cyclone_systems)} sistemas, {len(front_collection)} frentes "
        f"({sum(1 for f in front_collection if f.is_primary)} primarios)",
        fontsize=12, fontweight="bold"
    )

    # Guardar mapa
    map_file = output_dir / "test_advanced_features.png"
    plt.savefig(map_file, dpi=150, bbox_inches="tight")
    logger.info("   - Mapa guardado en: %s", map_file)

    logger.info("=== TEST COMPLETADO ===")
    logger.info("Resumen:")
    logger.info("  - Sistemas ciclónicos: %d", len(cyclone_systems))
    logger.info("  - Frentes totales: %d", len(front_collection))
    logger.info("  - Frentes primarios: %d", sum(1 for f in front_collection if f.is_primary))
    logger.info("  - Oclusiones detectadas: %d", len(occluded_fronts))

    return 0


if __name__ == "__main__":
    sys.exit(main())
