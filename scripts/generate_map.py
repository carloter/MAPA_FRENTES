#!/usr/bin/env python
"""CLI: Generar mapa de frentes sin GUI (headless)."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mapa_frentes.config import load_config
from mapa_frentes.data.ecmwf_download import download_ecmwf
from mapa_frentes.data.grib_reader import read_grib_files
from mapa_frentes.analysis.isobars import smooth_mslp, compute_isobar_levels
from mapa_frentes.analysis.pressure_centers import detect_pressure_centers
from mapa_frentes.plotting.map_canvas import create_map_figure
from mapa_frentes.plotting.isobar_renderer import draw_isobars, draw_pressure_labels
from mapa_frentes.plotting.export import export_map


def main():
    parser = argparse.ArgumentParser(
        description="Generar mapa de frentes meteorologicos"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Fecha YYYYMMDDHH. Default: ultimo disponible.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Ruta al fichero config.yaml",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Fichero de salida (PNG/PDF). Default: data/output/mapa_frentes.png",
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Paso de prediccion en horas (0, 3, 6, ..., 144). Default: 0.",
    )
    parser.add_argument(
        "--fronts", action="store_true",
        help="Incluir frentes detectados automaticamente",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="No descargar datos ni comprobar cache, usar ficheros existentes",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Forzar descarga aunque exista cache reciente",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)

    date = None
    if args.date:
        date = datetime.strptime(args.date, "%Y%m%d%H")

    # Descargar (con cache automatico) o usar ficheros existentes
    if args.no_download:
        # Buscar cualquier GRIB existente en cache
        cache_dir = Path(cfg.data.cache_dir)
        sfc_candidates = sorted(cache_dir.glob("ecmwf_sfc_*.grib2"), reverse=True)
        pl_candidates = sorted(cache_dir.glob("ecmwf_pl850_*.grib2"), reverse=True)
        if not sfc_candidates or not pl_candidates:
            logger.error("No se encontraron ficheros GRIB en %s", cache_dir)
            sys.exit(1)
        grib_paths = {
            "surface": sfc_candidates[0],
            "pressure": pl_candidates[0],
        }
        logger.info("Usando cache: %s, %s", grib_paths["surface"].name, grib_paths["pressure"].name)
    else:
        grib_paths = download_ecmwf(cfg, date=date, step=args.step, force=args.force)

    # Leer datos
    ds = read_grib_files(grib_paths, cfg)

    # Detectar coordenadas
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    # Isobaras
    msl_smooth = smooth_mslp(ds["msl"], sigma=cfg.isobars.smooth_sigma)
    levels = compute_isobar_levels(msl_smooth, interval=cfg.isobars.interval_hpa)

    # Centros de presion
    centers = detect_pressure_centers(msl_smooth, lats, lons, cfg)
    logger.info("Centros detectados: %d", len(centers))

    # Crear mapa
    fig, ax = create_map_figure(cfg)

    # Titulo con info del modelo y fecha
    time_str = ""
    for coord_name in ("valid_time", "time"):
        if coord_name in ds.coords:
            time_val = ds.coords[coord_name].values
            time_str = str(time_val)[:16].replace("T", " ") + " UTC"
            break

    step_val = args.step if args.step is not None else cfg.ecmwf.step
    if step_val > 0:
        title_line1 = f"Prediccion T+{step_val:03d}h - MSLP (hPa)"
    else:
        title_line1 = "Analisis en superficie - MSLP (hPa)"
    if args.fronts:
        title_line1 += " + Frentes"
    title_line2 = f"ECMWF IFS {cfg.ecmwf.resol.replace('p','.')}deg"
    if time_str:
        title_line2 += f"  |  Valido: {time_str}"

    ax.set_title(
        f"{title_line1}\n{title_line2}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # Dibujar isobaras y centros H/L
    draw_isobars(ax, msl_smooth, lons, lats, levels, cfg)
    draw_pressure_labels(ax, centers, cfg)

    # Frentes automaticos
    if args.fronts:
        try:
            from mapa_frentes.fronts.tfp import compute_tfp_fronts
            from mapa_frentes.fronts.classifier import classify_fronts
            from mapa_frentes.plotting.front_renderer import draw_fronts, draw_front_legend

            fronts = compute_tfp_fronts(ds, cfg)
            fronts = classify_fronts(fronts, ds, cfg)
            draw_fronts(ax, fronts, cfg)
            draw_front_legend(ax)
            logger.info("Frentes dibujados: %d", len(fronts.fronts))
        except Exception as e:
            logger.error("Error al detectar/dibujar frentes: %s", e)
            raise

    # Exportar
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(cfg.data.output_dir) / "mapa_frentes.png"

    export_map(fig, output_path, cfg)
    print(f"Mapa guardado: {output_path}")


if __name__ == "__main__":
    main()
