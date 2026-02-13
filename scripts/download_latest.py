#!/usr/bin/env python
"""CLI: Descargar datos ECMWF mas recientes y mostrar resumen."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Asegurar que el paquete es importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from mapa_frentes.config import load_config
from mapa_frentes.data.ecmwf_download import download_ecmwf
from mapa_frentes.data.grib_reader import read_grib_files


def main():
    parser = argparse.ArgumentParser(
        description="Descargar datos ECMWF IFS Open Data"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Fecha YYYYMMDDHH (ej: 2025050100). Default: ultimo disponible.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Ruta al fichero config.yaml",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Paso de prediccion en horas (0, 3, 6, ..., 144). Default: 0 (analisis).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar descarga aunque exista cache reciente",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)

    date = None
    if args.date:
        date = datetime.strptime(args.date, "%Y%m%d%H")

    print("=== Descargando datos ECMWF ===")
    grib_paths = download_ecmwf(cfg, date=date, step=args.step, force=args.force)
    for key, path in grib_paths.items():
        print(f"  {key}: {path}")

    print("\n=== Leyendo datos GRIB2 ===")
    ds = read_grib_files(grib_paths, cfg)

    print("\n=== Resumen del Dataset ===")
    print(ds)
    print(f"\nVariables: {list(ds.data_vars)}")
    for var in ds.data_vars:
        v = ds[var]
        print(f"  {var}: min={float(v.min()):.2f}, max={float(v.max()):.2f}, "
              f"shape={v.shape}")


if __name__ == "__main__":
    main()
