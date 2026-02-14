"""Descarga datos ECMWF IFS Open Data via ecmwf-opendata."""

import logging
from datetime import datetime, timezone
from pathlib import Path

from ecmwf.opendata import Client

from mapa_frentes.config import AppConfig

logger = logging.getLogger(__name__)

# Horas de run validas para ECMWF IFS Open Data
VALID_RUN_HOURS = (0, 6, 12, 18)

# Tamano minimo en bytes para considerar un GRIB valido (~100 KB)
MIN_GRIB_SIZE = 100_000

# Antigüedad maxima del cache en horas antes de re-descargar
MAX_CACHE_AGE_HOURS = 12


def _cache_is_valid(path: Path, max_age_hours: float) -> bool:
    """Comprueba si un fichero cacheado existe y no es demasiado antiguo."""
    if not path.exists():
        return False
    if path.stat().st_size < MIN_GRIB_SIZE:
        logger.info("Cache descartado (fichero demasiado pequeno): %s", path)
        return False
    age_hours = (
        datetime.now(timezone.utc)
        - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    ).total_seconds() / 3600
    if age_hours > max_age_hours:
        logger.info(
            "Cache caducado (%.1f h > %d h): %s",
            age_hours, max_age_hours, path,
        )
        return False
    logger.info("Usando cache (%.1f h): %s", age_hours, path)
    return True


def download_ecmwf(
    cfg: AppConfig,
    date: datetime | None = None,
    step: int | None = None,
    target_dir: str | Path | None = None,
    force: bool = False,
    max_cache_hours: float = MAX_CACHE_AGE_HOURS,
) -> dict[str, Path]:
    """Descarga GRIB2 de ECMWF IFS Open Data.

    Si los ficheros ya existen en cache y no son demasiado antiguos,
    se reutilizan sin volver a descargar.

    Args:
        cfg: Configuracion de la aplicacion.
        date: Fecha/hora del run. None = ultimo disponible.
        step: Paso de prediccion en horas (0, 3, 6, ..., 144).
              None = usar cfg.ecmwf.step.
        target_dir: Directorio destino. None = cfg.data.cache_dir.
        force: Si True, fuerza la descarga aunque exista cache.
        max_cache_hours: Horas maximas de antigüedad del cache.

    Returns:
        Dict con claves 'surface' y 'pressure' apuntando a los ficheros GRIB2.
        Si hay múltiples niveles, 'pressure' apunta al archivo combinado.
    """
    if step is None:
        step = cfg.ecmwf.step

    if target_dir is None:
        target_dir = Path(cfg.data.cache_dir)
    else:
        target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Nombres de fichero con fecha y step si se especifica
    if date is not None:
        date_tag = date.strftime("%Y%m%d%H")
        step_tag = f"_T{step:03d}" if step > 0 else ""
        sfc_target = target_dir / f"ecmwf_sfc_{date_tag}{step_tag}.grib2"
        pl_target = target_dir / f"ecmwf_pl_multi_{date_tag}{step_tag}.grib2"
    else:
        step_tag = f"_T{step:03d}" if step > 0 else ""
        sfc_target = target_dir / f"ecmwf_sfc_latest{step_tag}.grib2"
        pl_target = target_dir / f"ecmwf_pl_multi_latest{step_tag}.grib2"

    results = {"surface": sfc_target, "pressure": pl_target}

    # Comprobar cache
    sfc_ok = not force and _cache_is_valid(sfc_target, max_cache_hours)
    pl_ok = not force and _cache_is_valid(pl_target, max_cache_hours)

    if sfc_ok and pl_ok:
        logger.info("Todos los datos en cache, no se descarga nada.")
        return results

    # Preparar cliente y request
    client = Client(source=cfg.ecmwf.source)
    base_request = {
        "step": step,
        "resol": cfg.ecmwf.resol,
    }
    logger.info("Descargando ECMWF: step=%d h, resol=%s", step, cfg.ecmwf.resol)
    if date is not None:
        run_hour = date.hour
        if run_hour not in VALID_RUN_HOURS:
            # Redondear a la hora de run valida mas cercana (hacia atras)
            nearest = max(h for h in VALID_RUN_HOURS if h <= run_hour) if run_hour >= 6 else 0
            logger.warning(
                "Hora %02d no es un run valido ECMWF. Ajustando a %02dZ.",
                run_hour, nearest,
            )
            run_hour = nearest
            date = date.replace(hour=run_hour, minute=0, second=0, microsecond=0)
        base_request["date"] = date.strftime("%Y%m%d")
        base_request["time"] = run_hour

    # 1) Superficie: msl
    if not sfc_ok:
        logger.info("Descargando campos de superficie: %s", cfg.ecmwf.surface_params)
        client.download(
            type="fc",
            param=cfg.ecmwf.surface_params,
            target=str(sfc_target),
            **base_request,
        )
        logger.info("Superficie guardada en: %s", sfc_target)
    else:
        logger.info("Superficie: usando cache")

    # 2) Niveles de presion: Soporta lista de niveles
    if not pl_ok:
        levels = cfg.ecmwf.pressure_levels
        logger.info(
            "Descargando campos en niveles %s: %s", levels, cfg.ecmwf.pressure_params
        )
        client.download(
            type="fc",
            param=cfg.ecmwf.pressure_params,
            levtype="pl",
            levelist=levels,  # Ahora puede ser una lista
            target=str(pl_target),
            **base_request,
        )
        logger.info("Niveles de presion guardados en: %s", pl_target)
    else:
        logger.info("Niveles de presion: usando cache")

    return results
