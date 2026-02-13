"""Exportar mapas a PNG/PDF."""

import logging
from pathlib import Path

from matplotlib.figure import Figure

from mapa_frentes.config import AppConfig

logger = logging.getLogger(__name__)


def export_map(
    fig: Figure,
    output_path: str | Path,
    cfg: AppConfig,
    fmt: str | None = None,
):
    """Exporta la figura a fichero.

    Args:
        fig: Figura Matplotlib.
        output_path: Ruta de salida.
        cfg: Configuracion.
        fmt: Formato ('png' o 'pdf'). None = detectar de la extension.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt is None:
        fmt = output_path.suffix.lstrip(".").lower()
    if fmt not in ("png", "pdf"):
        fmt = cfg.export.default_format

    dpi = cfg.export.png_dpi if fmt == "png" else cfg.export.pdf_dpi

    fig.savefig(
        str(output_path),
        format=fmt,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    logger.info("Mapa exportado: %s (%s, %d dpi)", output_path, fmt, dpi)
