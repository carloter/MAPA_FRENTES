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
    filter_size: int = 20
    min_distance_deg: float = 3.5
    secondary_radius_deg: float = 10.0
    high_min_pressure: float = 1012.0   # No marcar H si presion < este valor
    low_max_pressure: float = 1020.0    # No marcar L si presion > este valor
    min_depth_hpa: float = 2.0          # Profundidad minima (dif. con entorno)
    depth_radius_deg: float = 5.0       # Radio para calcular presion media entorno
    h_color: str = "blue"
    l_color: str = "red"
    fontsize: int = 14
    fontweight: str = "bold"
    high_label: str = "A"
    low_label: str = "B"


@dataclass
class TFPConfig:
    smooth_sigma: float = 5.0
    gradient_threshold: float = 4.0e-6
    min_front_points: int = 8
    dbscan_eps_deg: float = 1.8
    dbscan_min_samples: int = 3
    simplify_tolerance_deg: float = 0.10
    use_mslp_filter: bool = True
    mslp_laplacian_sigma: float = 8.0
    # Connector parameters (previously hardcoded)
    min_front_length_deg: float = 5.0
    max_hop_deg: float = 3.0
    angular_weight: float = 0.55
    spline_smoothing: float = 0.6
    merge_distance_deg: float = 3.0
    # Frontogenesis filter
    use_frontogenesis_filter: bool = True
    frontogenesis_percentile: int = 25
    # Max fronts
    max_fronts: int = 15
    # Vorticity boost: reduce gradient threshold where vorticity is high
    use_vorticity_boost: bool = False
    vorticity_boost_threshold: float = 5.0e-5  # |vor| > this activates boost
    vorticity_boost_factor: float = 0.4         # gradient_threshold * this in high-vor zones
    # F diagnostic (Parfitt et al. 2017): thermal gradient × vorticity / coriolis
    use_f_diagnostic: bool = True
    f_diagnostic_threshold: float = 1.0         # F > this to keep front point
    # Contour method (Sansom & Catto 2024): contour-then-mask vs legacy DBSCAN
    use_contour_method: bool = True


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
    front_linewidth: float = 1.8
    front_symbol_size: float = 5.0
    front_symbol_spacing: float = 12.0


@dataclass
class InstabilityLinesConfig:
    enabled: bool = True
    smooth_sigma: float = 4.0
    convergence_threshold: float = 2.0e-5
    min_length_deg: float = 2.0


@dataclass
class TemporalConfig:
    enabled: bool = False
    context_steps: List[int] = field(default_factory=lambda: [-6, -3, 0, 3, 6])
    match_distance_deg: float = 3.0
    min_persistence: int = 2


@dataclass
class CenterFrontsConfig:
    """Configuracion para generacion de frentes desde centros de presion."""
    search_radius_deg: float = 5.0
    trace_step_deg: float = 0.5
    max_front_length_deg: float = 15.0
    max_turn_deg: float = 30.0
    gradient_cutoff_factor: float = 0.3
    spline_smoothing: float = 0.4
    max_association_distance_deg: float = 10.0
    require_low_center: bool = True        # Solo mantener frentes cerca de centros L
    max_distance_to_low_deg: float = 12.0  # Radio maximo de busqueda alrededor de L


@dataclass
class OcclusionConfig:
    """Configuracion para deteccion robusta de oclusiones."""
    enabled: bool = True
    min_score: float = 0.60           # Score mínimo para clasificar como ocluido
    vsi_threshold: float = 0.10       # VSI para warm-core seclusion
    vorticity_threshold: float = 8e-5 # s⁻¹
    t_pattern_radius_deg: float = 5.0 # Radio búsqueda patrón T
    t_pattern_max_angle: float = 90.0 # Ángulo máximo convergencia
    use_multilevel: bool = True       # Si False, solo 850 hPa (fallback)


@dataclass
class PrecipitationConfig:
    """Configuracion para overlay de precipitacion."""
    threshold_mm: float = 0.5   # no dibujar por debajo de este valor
    alpha: float = 0.5          # transparencia del contourf
    cmap: str = "YlGnBu"        # colormap
    num_levels: int = 15         # niveles de contorno


@dataclass
class WindVectorsConfig:
    """Configuracion para vectores de viento (quiver)."""
    thin_factor: int = 8        # cada N puntos de la rejilla
    scale: float = 600.0        # escala del quiver
    width: float = 0.002        # grosor de las flechas
    color: str = "#555555"
    alpha: float = 0.6
    web_scale_factor: float = 4.0   # multiplicador de scale para web (coord. Mercator)
    web_width_factor: float = 0.3   # multiplicador de width para web


@dataclass
class BackgroundFieldConfig:
    """Configuracion para campos de fondo derivados del IFS."""
    default_field: str = "none"   # "none", "theta_e_850", "grad_theta_e_850",
                                  # "thickness_1000_500", "temp_advection_850",
                                  # "wind_speed_850"
    alpha: float = 0.45           # Transparencia del contourf
    num_levels: int = 20          # Numero de niveles de contorno
    colorbar: bool = True         # Mostrar colorbar


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
    pressure_levels: List[int] = field(default_factory=lambda: [500, 700, 850])
    pressure_params: List[str] = field(default_factory=lambda: ["t", "q", "u", "v", "vo"])


@dataclass
class AppConfig:
    area: AreaConfig = field(default_factory=AreaConfig)
    isobars: IsobarConfig = field(default_factory=IsobarConfig)
    pressure_centers: PressureCentersConfig = field(default_factory=PressureCentersConfig)
    tfp: TFPConfig = field(default_factory=TFPConfig)
    instability_lines: InstabilityLinesConfig = field(default_factory=InstabilityLinesConfig)
    center_fronts: CenterFrontsConfig = field(default_factory=CenterFrontsConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ecmwf: ECMWFConfig = field(default_factory=ECMWFConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    background_field: BackgroundFieldConfig = field(default_factory=BackgroundFieldConfig)
    precipitation: PrecipitationConfig = field(default_factory=PrecipitationConfig)
    wind_vectors: WindVectorsConfig = field(default_factory=WindVectorsConfig)


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
        instability_lines=InstabilityLinesConfig(**raw.get("instability_lines", {})),
        center_fronts=CenterFrontsConfig(**raw.get("center_fronts", {})),
        temporal=TemporalConfig(**raw.get("temporal", {})),
        plotting=PlottingConfig(**raw.get("plotting", {})),
        export=ExportConfig(**raw.get("export", {})),
        data=DataConfig(**raw.get("data", {})),
        ecmwf=ECMWFConfig(**raw.get("ecmwf", {})),
        occlusion=OcclusionConfig(**raw.get("occlusion", {})),
        background_field=BackgroundFieldConfig(**raw.get("background_field", {})),
        precipitation=PrecipitationConfig(**raw.get("precipitation", {})),
        wind_vectors=WindVectorsConfig(**raw.get("wind_vectors", {})),
    )
