# Funcionalidades Avanzadas - MAPA_FRENTES

## Descripción

Este documento describe las nuevas funcionalidades implementadas para mejorar la detección de frentes meteorológicos:

1. **Detección robusta de oclusiones** - Método multi-criterio con datos multi-nivel
2. **Agrupado por sistemas ciclónicos** - Organización de frentes por centro L
3. **Ranking de frentes** - Clasificación de frentes principales vs secundarios

---

## 1. Detección Robusta de Oclusiones

### Motivación

El método original clasificaba como "ocluido" cualquier frente que no fuera 70% puro frío o cálido, basándose solo en advección térmica en 850 hPa. Esto era simplista y no capturaba la estructura física real de las oclusiones.

### Nuevo Método Multi-Criterio

El nuevo sistema usa **5 factores** ponderados para detectar oclusiones:

#### Factores de Scoring (0-1):

1. **Estructura Vertical (30%)**: VSI = (θ_e_700 - θ_e_850) / θ_e_mean
   - Oclusiones clásicas: VSI moderado (0.03-0.05)
   - Warm seclusion: VSI alto (> 0.10)

2. **Vorticidad (25%)**: Promedio de vorticidad ciclónica en el frente
   - Oclusiones típicamente: ζ > 8e-5 s⁻¹

3. **Patrón Geométrico (20%)**: Proximidad y ángulo con centros L
   - Busca patrones T/Y cerca de centros de baja presión

4. **Frontogénesis Vertical (15%)**: Ratio frontogénesis_700 / frontogénesis_850
   - > 1 indica elevación del frente

5. **Profundidad del Centro (10%)**: Normalizar (1013 - presión) / 30

#### Subtipos de Oclusión:

- **COLD_OCCLUDED**: Oclusión de tipo frío (aire frío avanzando)
- **WARM_OCCLUDED**: Oclusión de tipo cálido
- **WARM_SECLUSION**: Seclusión cálida (núcleo cálido aislado en ciclón bomba)

### Configuración

En `config.yaml`:

```yaml
occlusion:
  enabled: true
  min_score: 0.60            # Score mínimo para clasificar como ocluido
  vsi_threshold: 0.10        # VSI para warm-core seclusion
  vorticity_threshold: 8.0e-5  # s⁻¹
  t_pattern_radius_deg: 5.0  # Radio búsqueda patrón T
  t_pattern_max_angle: 90.0  # Ángulo máximo convergencia
  use_multilevel: true       # Si False, solo 850 hPa (fallback)
```

### Datos Requeridos

El sistema descarga automáticamente datos de **múltiples niveles de presión**:

```yaml
ecmwf:
  pressure_levels: [500, 700, 850]
  pressure_params:
    - "t"   # temperatura
    - "q"   # humedad específica
    - "u"   # viento zonal
    - "v"   # viento meridional
    - "vo"  # vorticidad relativa (nuevo)
```

---

## 2. Agrupado por Sistemas Ciclónicos

### Motivación

Organizar frentes por el sistema ciclónico al que pertenecen permite:
- Mejor comprensión de la estructura sinóptica
- Análisis por ciclón individual
- Filtrado y visualización selectiva

### Estructura de Datos

#### `CycloneSystem`

Representa un sistema ciclónico completo:

```python
@dataclass
class CycloneSystem:
    center: PressureCenter              # Centro L primario
    fronts: List[Front]                 # Frentes asociados
    secondary_centers: List[PressureCenter]  # Centros secundarios
    name: str                           # Nombre de la borrasca
    valid_time: str
    metadata: dict                      # Estadísticas calculadas
```

**Propiedades**:
- `id`: ID del sistema (usa el ID del centro primario)
- `is_primary`: True si el centro es primario
- `get_fronts_by_type(ftype)`: Filtra frentes por tipo

**Métodos**:
- `compute_statistics()`: Calcula n_fronts, longitud total, bounding box, etc.

#### `CycloneSystemCollection`

Colección de sistemas ciclónicos:

```python
@dataclass
class CycloneSystemCollection:
    systems: List[CycloneSystem]
    unassociated_fronts: List[Front]    # Frentes sin centro asociado
    valid_time: str
```

### Uso

```python
from mapa_frentes.fronts.cyclone_systems import build_cyclone_systems

# Construir sistemas
systems = build_cyclone_systems(front_collection, centers, cfg)

# Acceder a sistemas
for system in systems:
    print(f"Sistema {system.id}: {len(system.fronts)} frentes")
    print(f"  Centro: {system.center.lat}°N, {system.center.lon}°E")
    print(f"  Presión: {system.center.value} hPa")

    # Frentes por tipo
    cold_fronts = system.get_fronts_by_type(FrontType.COLD)
    warm_fronts = system.get_fronts_by_type(FrontType.WARM)

# Frentes huérfanos
print(f"Frentes no asociados: {len(systems.unassociated_fronts)}")
```

### Serialización

Los sistemas se pueden exportar a GeoJSON:

```python
from mapa_frentes.fronts.io import save_systems

# Guardar sistemas
save_systems(systems, "cyclone_systems.geojson")
```

Cada sistema se serializa como un Feature con:
- **Geometría**: MultiLineString (todos los frentes del sistema)
- **Properties**: ID, centro (lat/lon/presión), nombre, metadata, lista de front_ids

---

## 3. Ranking de Frentes Principales/Secundarios

### Motivación

No todos los frentes tienen la misma importancia sinóptica. El ranking permite:
- Identificar frentes principales de cada ciclón
- Filtrar frentes secundarios para mapas más limpios
- Destacar estructuras más importantes

### Algoritmo de Scoring

Cada frente recibe un **importance_score** (0-1) basado en 6 factores:

1. **Longitud (25%)**: Frentes largos más importantes
   - Normalizar: 5° = 0, 20° = 1

2. **Intensidad Térmica (25%)**: Gradiente térmico medio
   - Normalizar: 3e-6 = 0, 8e-6 = 1

3. **Frontogénesis (20%)**: Frentes en desarrollo activo
   - Normalizar: 0 = 0, 5e-10 = 1

4. **Distancia al Centro (15%)**: Frentes cerca del núcleo son primarios
   - Normalizar: 0° = 1, 10° = 0

5. **Tipo de Frente (10%)**: Depende de madurez del ciclón
   - Ciclón en desarrollo (P > 990 hPa): COLD > WARM > OCCLUDED
   - Ciclón maduro (P < 990 hPa): OCCLUDED > COLD > WARM

6. **Profundidad del Centro (5%)**: Centros profundos → frentes importantes

### Clasificación

Frentes con `importance_score >= 0.60` (configurable) se marcan como **primarios**:
- `front.importance_score`: Score numérico (0-1)
- `front.is_primary`: Boolean (True/False)

### Uso

```python
from mapa_frentes.fronts.ranking import rank_all_systems

# Aplicar ranking
rank_all_systems(cyclone_systems, ds, cfg, threshold=0.60)

# Filtrar frentes primarios
for system in cyclone_systems:
    primary = [f for f in system.fronts if f.is_primary]
    secondary = [f for f in system.fronts if not f.is_primary]

    print(f"Sistema {system.id}:")
    print(f"  Primarios: {len(primary)}")
    print(f"  Secundarios: {len(secondary)}")

    # Ver scores
    for front in sorted(system.fronts, key=lambda f: f.importance_score, reverse=True):
        print(f"    {front.id[:12]}: {front.importance_score:.2f} "
              f"({'P' if front.is_primary else 'S'})")
```

### Visualización

El renderizador soporta estilo diferenciado:

```python
from mapa_frentes.plotting.front_renderer import draw_fronts

# Dibujar con estilo diferenciado
draw_fronts(ax, collection, cfg, show_importance=True)

# Frentes secundarios: linewidth × 0.6, alpha = 0.4
# Frentes primarios: linewidth normal, alpha = 1.0
```

---

## Compatibilidad con Código Existente

### Campos Nuevos en `Front`

Todos tienen valores por defecto, no rompen código existente:

```python
occlusion_score: float = 0.0       # Score de detección de oclusión
occlusion_type: str = ""           # Subtipo específico
importance_score: float = 0.0      # Score de importancia
is_primary: bool = False           # Principal vs secundario
```

### Nuevos Tipos de Frente

Se añadieron 3 tipos nuevos al enum `FrontType`:
- `COLD_OCCLUDED`
- `WARM_OCCLUDED`
- `WARM_SECLUSION`

El tipo `OCCLUDED` original se mantiene para compatibilidad.

### Serialización GeoJSON

Los nuevos campos se incluyen en `properties`:

```json
{
  "type": "Feature",
  "geometry": {...},
  "properties": {
    "id": "front_123",
    "front_type": "cold_occluded",
    "occlusion_score": 0.75,
    "occlusion_type": "cold_occluded",
    "importance_score": 0.82,
    "is_primary": true,
    ...
  }
}
```

La deserialización antigua sigue funcionando (campos nuevos se ignoran si ausentes).

---

## Script de Prueba

Ejecutar el script de prueba completo:

```bash
python scripts/test_advanced_features.py
```

**Salida esperada**:
- Descarga datos multi-nivel (500, 700, 850 hPa + vorticidad)
- Detecta frentes TFP
- Clasifica con detección robusta de oclusiones
- Construye sistemas ciclónicos
- Aplica ranking
- Genera mapa con todas las funcionalidades
- Guarda sistemas en GeoJSON

**Archivos generados**:
- `data/output/cyclone_systems.geojson`: Sistemas en formato GeoJSON
- `data/output/test_advanced_features.png`: Mapa de prueba

---

## Ejemplo Completo

```python
from mapa_frentes.config import load_config
from mapa_frentes.data.ecmwf_download import download_ecmwf
from mapa_frentes.data.grib_reader import read_grib_files
from mapa_frentes.fronts.tfp import detect_tfp_fronts
from mapa_frentes.fronts.classifier import classify_fronts
from mapa_frentes.fronts.cyclone_systems import build_cyclone_systems
from mapa_frentes.fronts.ranking import rank_all_systems
from mapa_frentes.analysis.pressure_centers import detect_pressure_centers
from mapa_frentes.utils.smoothing import smooth_field

# 1. Configuración
cfg = load_config()

# 2. Descargar y leer datos
grib_paths = download_ecmwf(cfg)
ds = read_grib_files(grib_paths, cfg)

# 3. Detectar centros de presión
msl_smooth = smooth_field(ds["msl"].values, sigma=cfg.isobars.smooth_sigma)
centers = detect_pressure_centers(msl_smooth, ds.latitude.values, ds.longitude.values, cfg)
low_centers = [c for c in centers if c.type == "L"]

# 4. Detectar y clasificar frentes
fronts = detect_tfp_fronts(ds, cfg)
fronts = classify_fronts(fronts, ds, cfg, centers=low_centers)

# 5. Construir sistemas ciclónicos
systems = build_cyclone_systems(fronts, centers, cfg)

# 6. Aplicar ranking
rank_all_systems(systems, ds, cfg)

# 7. Analizar resultados
for system in systems:
    print(f"\n=== Sistema {system.id} ===")
    print(f"Centro: {system.center.lat:.1f}°N, {system.center.lon:.1f}°E")
    print(f"Presión: {system.center.value:.0f} hPa")
    print(f"Frentes: {len(system.fronts)}")

    # Frentes principales
    primary = [f for f in system.fronts if f.is_primary]
    for front in primary:
        print(f"  → {front.front_type.value}: score={front.importance_score:.2f}")
        if front.occlusion_score > 0:
            print(f"    (oclusión: {front.occlusion_score:.2f})")
```

---

## Próximos Pasos

Posibles mejoras futuras:

1. **Tracking temporal de oclusiones**: Seguir evolución de oclusiones en timesteps
2. **Warm core seclusion**: Mejorar detección usando θ_e en múltiples niveles
3. **Validación**: Comparar con análisis sinópticos oficiales
4. **GUI**: Integrar selector de sistemas y filtro de primarios en la interfaz
5. **Frontogénesis completa**: Implementar cálculo exacto en todos los niveles

---

## Referencias

- Hewson, T. D. (1998). Objective fronts. *Meteorological Applications*, 5(1), 37-65.
- Shapiro, M. A., & Keyser, D. (1990). Fronts, jet streams, and the tropopause. *Extratropical Cyclones: The Erik Palmén Memorial Volume*, 167-191.
- Schultz, D. M., & Vaughan, G. (2011). Occluded fronts and the occlusion process: A fresh look at conventional wisdom. *Bulletin of the American Meteorological Society*, 92(4), 443-466.

---

**Versión**: 1.0
**Fecha**: 2026-02-14
**Autor**: MAPA_FRENTES Development Team
