"""Carga configuracion YAML a dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class AreaConfig:
    lon_min: float = -60.0
    lon_max: float = 30.0
    lat_min: float = 25.0
    lat_max: float = 65.0
    data_padding_deg: float = 10.0

    def padded(self) -> "AreaConfig":
        """Devuelve un area expandida para descarga/recorte de datos.

        Con proyeccion Lambert Conformal, el rectangulo en coordenadas
        geograficas se deforma y las esquinas del mapa proyectado pueden
        extenderse mas alla del area de visualizacion. Este padding
        garantiza que toda la zona visible tenga datos.
        """
        p = self.data_padding_deg
        return AreaConfig(
            lon_min=max(self.lon_min - p, -180.0),
            lon_max=min(self.lon_max + p, 180.0),
            lat_min=max(self.lat_min - p, -90.0),
            lat_max=min(self.lat_max + p, 90.0),
            data_padding_deg=0.0,
        )


@dataclass
class IsobarConfig:
    interval_hpa: int = 4
    smooth_sigma: float = 1.5
    linewidth: float = 0.8
    color: str = "black"
    label_fontsize: int = 7


@dataclass
class PressureCentersConfig:
    filter_size: int = 30
    min_distance_deg: float = 5.0
    h_color: str = "blue"
    l_color: str = "red"
    fontsize: int = 14
    fontweight: str = "bold"


@dataclass
class TFPConfig:
    smooth_sigma: float = 6.0
    gradient_threshold: float = 5.0e-6
    min_front_points: int = 6
    dbscan_eps_deg: float = 1.5
    dbscan_min_samples: int = 3
    simplify_tolerance_deg: float = 0.12
    use_mslp_filter: bool = True
    mslp_laplacian_sigma: float = 8.0


@dataclass
class PlottingConfig:
    figsize: List[int] = field(default_factory=lambda: [16, 10])
    dpi: int = 150
    projection: str = "LambertConformal"
    projection_params: dict = field(default_factory=lambda: {
        "central_longitude": -15.0,
        "central_latitude": 45.0,
        "standard_parallels": [30.0, 60.0],
    })
    coastline_color: str = "gray"
    coastline_linewidth: float = 0.5
    border_color: str = "lightgray"
    border_linewidth: float = 0.3
    ocean_color: str = "lightskyblue"
    land_color: str = "antiquewhite"
    front_linewidth: float = 3.0


@dataclass
class ExportConfig:
    default_format: str = "png"
    png_dpi: int = 300
    pdf_dpi: int = 150


@dataclass
class DataConfig:
    cache_dir: str = "data/cache"
    output_dir: str = "data/output"
    sessions_dir: str = "data/sessions"


@dataclass
class ECMWFConfig:
    source: str = "aws"
    model: str = "ifs"
    resol: str = "0p25"
    step: int = 0
    surface_params: List[str] = field(default_factory=lambda: ["msl"])
    pressure_level: int = 850
    pressure_params: List[str] = field(default_factory=lambda: ["t", "q", "u", "v"])


@dataclass
class AppConfig:
    area: AreaConfig = field(default_factory=AreaConfig)
    isobars: IsobarConfig = field(default_factory=IsobarConfig)
    pressure_centers: PressureCentersConfig = field(default_factory=PressureCentersConfig)
    tfp: TFPConfig = field(default_factory=TFPConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ecmwf: ECMWFConfig = field(default_factory=ECMWFConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Carga configuracion desde YAML. Usa defaults si no se encuentra."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig(
        area=AreaConfig(**raw.get("area", {})),
        isobars=IsobarConfig(**raw.get("isobars", {})),
        pressure_centers=PressureCentersConfig(**raw.get("pressure_centers", {})),
        tfp=TFPConfig(**raw.get("tfp", {})),
        plotting=PlottingConfig(**raw.get("plotting", {})),
        export=ExportConfig(**raw.get("export", {})),
        data=DataConfig(**raw.get("data", {})),
        ecmwf=ECMWFConfig(**raw.get("ecmwf", {})),
    )
