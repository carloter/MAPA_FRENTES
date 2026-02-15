# MAPA FRENTES - MeteoGalicia

Herramienta para generar mapas de frentes meteorologicos con simbologia WMO estandar.
Detecta frentes automaticamente mediante el algoritmo TFP (Thermal Front Parameter)
y generacion desde centros de baja presion (borrascas), con edicion interactiva.

Disponible en dos interfaces:
- **GUI de escritorio** (PyQt5 + Matplotlib + Cartopy)
- **Prototipo web** (FastAPI + Leaflet) con mosaico multi-campo y simbolos WMO interactivos

**Datos**: ECMWF IFS Open Data (0.25 deg) - MSLP + campos multi-nivel (500, 700, 850 hPa)
**Area**: Atlantico Norte + Europa (~60W-30E, 25N-65N)

---

## Instalacion

### Opcion A: conda (recomendado en Windows)

```bash
conda create -n mapa_frentes python=3.12
conda activate mapa_frentes
conda install -c conda-forge numpy scipy xarray cfgrib eccodes metpy matplotlib cartopy pyqt shapely pyyaml scikit-learn
pip install ecmwf-opendata geojson fastapi uvicorn
```

> **Nota**: `cfgrib` y `eccodes` dan problemas con pip en Windows.
> Instalarlos siempre via `conda-forge`.

### Opcion B: pip (Linux/Mac o si ya tienes eccodes instalado)

```bash
pip install -r requirements.txt
pip install fastapi uvicorn  # para el prototipo web
```

---

## Uso rapido

### 1. GUI de escritorio (PyQt5)

```bash
python -m mapa_frentes
```

Dentro de la aplicacion:
1. **Descargar** - pulsa el boton o menu Archivo > Descargar datos
2. **Detectar frentes** - pulsa el boton (se activa tras descargar). Detecta frentes por TFP y los asocia automaticamente a centros de baja presion
3. **Generar desde borrasca** - cambia al modo "Generar desde B" y haz click en un centro L (borrasca) para generar frentes frio, calido y ocluido automaticamente
4. **Editar** - cambia el modo en la toolbar (Seleccionar, Arrastrar, Anadir, Borrar)
5. **Campo de fondo** - selector en la toolbar para superponer campos derivados (theta_e, gradiente, espesor, adveccion, viento)
6. **Exportar** - menu Archivo > Exportar mapa (PNG o PDF)
7. **Guardar sesion** - Ctrl+S guarda los frentes editados como GeoJSON

### 2. Prototipo web (FastAPI + Leaflet)

```bash
python web/app.py
```

Abre http://127.0.0.1:8000 en el navegador. La interfaz web ofrece:

- **Mosaico configurable** (2x2, 3x2, 3x3, 4x3) con campos derivados independientes por panel
- **Mapas sincronizados** - zoom y pan se sincronizan en todos los paneles
- **Simbolos WMO** - triangulos (frio), semicirculos (calido), combinados (ocluido), alternados (estacionario)
- **Edicion interactiva** - dibujar frentes con seleccion de tipo, click para invertir direccion de simbolos
- **Isobaras y centros H/L** como capas GeoJSON
- **Costas y fronteras** de Natural Earth 50m

### 3. Generar mapa sin GUI (headless)

```bash
# Descarga datos y genera PNG con isobaras + centros H/L
python scripts/generate_map.py

# Con frentes automaticos
python scripts/generate_map.py --fronts

# Fecha concreta (YYYYMMDDHH)
python scripts/generate_map.py --date 2025050100 --fronts

# Usar datos ya descargados (sin volver a descargar)
python scripts/generate_map.py --no-download --fronts

# Guardar en ruta concreta
python scripts/generate_map.py --fronts -o mi_mapa.png
```

El mapa se guarda por defecto en `data/output/mapa_frentes.png`.

### 4. Solo descargar datos

```bash
# Descarga el ultimo analisis disponible
python scripts/download_latest.py

# Fecha concreta
python scripts/download_latest.py --date 2025050100
```

Los GRIB2 se guardan en `data/cache/`.

---

## Atajos de teclado (GUI de escritorio)

| Atajo | Accion |
|-------|--------|
| Ctrl+O | Cargar sesion GeoJSON |
| Ctrl+S | Guardar sesion GeoJSON |
| Ctrl+E | Exportar mapa PNG/PDF |
| Ctrl+Z | Deshacer |
| Ctrl+Y | Rehacer |
| Ctrl+B | Conectar frente a borrasca |
| Delete | Borrar frente seleccionado |
| Ctrl+Q | Salir |

---

## Modos de edicion (GUI de escritorio)

| Modo | Uso |
|------|-----|
| **Navegar** | Zoom y pan con la toolbar de Matplotlib |
| **Seleccionar** | Click en un frente para seleccionarlo |
| **Arrastrar** | Mover vertices del frente seleccionado |
| **Anadir frente** | Click para colocar puntos, doble-click para terminar |
| **Borrar frente** | Click en un frente para eliminarlo |
| **Generar desde B** | Click en un centro de baja presion (B) para generar frentes automaticamente |

El combo **Tipo** cambia el tipo del frente seleccionado o del siguiente que se cree.

---

## Deteccion de frentes

La herramienta ofrece dos metodos complementarios de deteccion:

### TFP (Thermal Front Parameter)
Metodo automatico basado en Hewson (1998). Calcula la temperatura potencial del bulbo humedo (theta_w) a 850 hPa, detecta los zero-crossings del TFP y los conecta en polilineas suaves. Incluye filtros de calidad (zonas ciclonicas, frontogenesis positiva) y clasificacion automatica por adveccion termica (frio/calido/ocluido).

### Generacion desde centros de baja presion
Genera frentes directamente desde borrascas detectadas. El algoritmo:
1. **Scoring azimutal**: muestrea la adveccion termica en un circulo alrededor del centro L para identificar las direcciones del frente frio (adveccion fria, tipicamente S/SW) y calido (adveccion calida, tipicamente E/NE)
2. **Ray-marching**: traza cada frente desde el centro hacia afuera siguiendo la cresta del gradiente de theta_w, ajustando la direccion en cada paso
3. **Oclusion**: si se detectan ambos frentes, genera un segmento ocluido en el sector intermedio
4. **Suavizado**: aplica spline cubica para curvas naturales

Ambos metodos se pueden combinar: detectar con TFP y luego generar frentes adicionales desde centros, o usar cada uno independientemente. Los frentes TFP se asocian automaticamente al centro L mas cercano.

---

## Funcionalidades avanzadas

Ver [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) para documentacion detallada.

- **Deteccion robusta de oclusiones**: Scoring multi-factor (5 criterios: estructura vertical, vorticidad, geometria, frontogenesis, profundidad). Distingue subtipos: COLD_OCCLUDED, WARM_OCCLUDED, WARM_SECLUSION.
- **Sistemas ciclonicos**: Agrupacion automatica de frentes por centro de baja presion (CycloneSystem). Permite analisis por ciclon individual.
- **Ranking de frentes**: Clasificacion primarios vs secundarios con importance_score (6 factores: longitud, gradiente, frontogenesis, distancia al centro, tipo, profundidad).
- **Campos derivados**: theta_e 850, gradiente theta_e, espesor 1000-500, adveccion termica 850, viento 850 hPa. Visualizables como fondo en la GUI y como mosaico en la version web.
- **Datos multi-nivel**: Descarga automatica de campos en 500, 700 y 850 hPa incluyendo vorticidad.

---

## Estructura del proyecto

```
MAPA_FRENTES/
├── config.yaml                  # Configuracion general
├── requirements.txt
├── mapa_frentes/
│   ├── __main__.py              # python -m mapa_frentes
│   ├── config.py                # Carga YAML -> dataclasses
│   ├── data/
│   │   ├── ecmwf_download.py    # Descarga ECMWF IFS Open Data
│   │   └── grib_reader.py       # Lectura GRIB2 -> xarray
│   ├── analysis/
│   │   ├── isobars.py           # Suavizado MSLP, niveles de isobaras
│   │   ├── pressure_centers.py  # Deteccion centros H/L con profundidad
│   │   └── derived_fields.py    # Campos derivados (theta_e, gradientes, etc.)
│   ├── fronts/
│   │   ├── models.py            # Front, FrontCollection, CycloneSystem
│   │   ├── tfp.py               # Deteccion TFP (Hewson 1998)
│   │   ├── classifier.py        # Clasificacion frio/calido/ocluido
│   │   ├── connector.py         # DBSCAN + spline smoothing
│   │   ├── center_fronts.py     # Generacion desde centros L
│   │   ├── association.py       # Asociacion frentes-centros + extension Bezier
│   │   ├── cyclone_systems.py   # Agrupado por sistema ciclonico
│   │   ├── ranking.py           # Scoring importancia + primarios/secundarios
│   │   └── io.py                # Serializacion GeoJSON
│   ├── plotting/
│   │   ├── map_canvas.py        # Canvas Cartopy base
│   │   ├── isobar_renderer.py   # Isobaras + campos de fondo
│   │   ├── front_renderer.py    # Frentes con simbologia WMO (MetPy)
│   │   └── export.py            # Exportacion PNG/PDF
│   ├── gui/
│   │   ├── main_window.py       # Ventana principal PyQt5
│   │   ├── map_widget.py        # Widget de mapa interactivo
│   │   ├── front_editor.py      # Editor de frentes (modos de edicion)
│   │   └── dialogs.py           # Dialogos de configuracion
│   └── utils/
│       ├── geo.py               # Funciones geograficas (gradiente esferico)
│       └── smoothing.py         # Suavizado gaussiano
├── web/
│   ├── app.py                   # Backend FastAPI (endpoints REST)
│   └── static/
│       └── index.html           # Frontend Leaflet (SPA)
├── scripts/
│   ├── download_latest.py       # CLI descarga de datos
│   ├── generate_map.py          # CLI generacion headless
│   └── test_advanced_features.py # Test pipeline completo
├── data/
│   ├── cache/                   # GRIB2 descargados
│   ├── output/                  # Mapas exportados
│   └── sessions/                # Sesiones guardadas (GeoJSON)
└── tests/                       # pytest
```

---

## API REST (prototipo web)

El backend expone los siguientes endpoints:

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/` | Pagina principal (SPA) |
| POST | `/api/load` | Descarga datos ECMWF y calcula isobaras/centros |
| GET | `/api/bounds` | Limites geograficos y campos disponibles |
| GET | `/api/fields/{name}/image` | PNG transparente de campo derivado |
| GET | `/api/isobars` | Isobaras como GeoJSON LineStrings |
| GET | `/api/centers` | Centros de presion H/L como JSON |
| GET | `/api/fronts/detect` | Deteccion automatica TFP + clasificacion |
| GET | `/api/fronts` | Frentes actuales como GeoJSON |
| PUT | `/api/fronts` | Guardar frentes editados |
| POST | `/api/fronts/generate/{id}` | Generar frentes desde un centro L |
| GET | `/api/coastlines` | Costas y fronteras (Natural Earth 50m) |

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Configuracion

Edita `config.yaml` para ajustar:

- **area**: extension geografica del mapa
- **isobars**: intervalo (hPa), suavizado, estilo
- **pressure_centers**: tamano del filtro, distancia minima, umbrales H/L
- **tfp**: sigma de suavizado, umbral de gradiente, parametros DBSCAN
- **center_fronts**: radio de busqueda, paso de ray-marching, longitud maxima, suavizado
- **occlusion**: scoring multi-factor, umbrales VSI y vorticidad
- **plotting**: tamano de figura, colores, grosor de lineas
- **export**: DPI para PNG y PDF

Tambien puedes cambiar estos parametros desde la GUI en menu Configuracion > Parametros.

---

## Referencias

- Hewson, T. D. (1998). Objective fronts. *Meteorological Applications*, 5(1), 37-65.
- Schultz, D. M., & Vaughan, G. (2011). Occluded fronts and the occlusion process. *Bull. Amer. Meteor. Soc.*, 92(4), 443-466.
