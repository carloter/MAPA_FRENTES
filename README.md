# MAPA FRENTES - MeteoGalicia

Herramienta para generar mapas de frentes meteorologicos con simbologia WMO estandar.
Detecta frentes automaticamente mediante el algoritmo TFP (Thermal Front Parameter)
y generacion desde centros de baja presion (borrascas), con edicion interactiva
en una GUI de escritorio PyQt.

**Datos**: ECMWF IFS Open Data (0.25 deg) - MSLP + campos 850 hPa
**Area**: Atlantico Norte + Europa (~60W-30E, 25N-65N)

---

## Instalacion

### Opcion A: conda (recomendado en Windows)

```bash
conda create -n mapa_frentes python=3.12
conda activate mapa_frentes
conda install -c conda-forge numpy scipy xarray cfgrib eccodes metpy matplotlib cartopy pyqt shapely pyyaml scikit-learn
pip install ecmwf-opendata geojson
```

> **Nota**: `cfgrib` y `eccodes` dan problemas con pip en Windows.
> Instalarlos siempre via `conda-forge`.

### Opcion B: pip (Linux/Mac o si ya tienes eccodes instalado)

```bash
pip install -r requirements.txt
```

---

## Uso rapido

### 1. Abrir la GUI

```bash
python -m mapa_frentes
```

Dentro de la aplicacion:
1. **Descargar** - pulsa el boton o menu Archivo > Descargar datos
2. **Detectar frentes** - pulsa el boton (se activa tras descargar). Detecta frentes por TFP y los asocia automaticamente a centros de baja presion
3. **Generar desde borrasca** - cambia al modo "Generar desde B" y haz click en un centro L (borrasca) para generar frentes frio, calido y ocluido automaticamente
4. **Editar** - cambia el modo en la toolbar (Seleccionar, Arrastrar, Anadir, Borrar)
5. **Exportar** - menu Archivo > Exportar mapa (PNG o PDF)
6. **Guardar sesion** - Ctrl+S guarda los frentes editados como GeoJSON

### 2. Generar mapa sin GUI (headless)

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

### 3. Solo descargar datos

```bash
# Descarga el ultimo analisis disponible
python scripts/download_latest.py

# Fecha concreta
python scripts/download_latest.py --date 2025050100
```

Los GRIB2 se guardan en `data/cache/`.

---

## Atajos de teclado (GUI)

| Atajo | Accion |
|-------|--------|
| Ctrl+O | Cargar sesion GeoJSON |
| Ctrl+S | Guardar sesion GeoJSON |
| Ctrl+E | Exportar mapa PNG/PDF |
| Ctrl+Z | Deshacer |
| Ctrl+Y | Rehacer |
| Delete | Borrar frente seleccionado |
| Ctrl+Q | Salir |

---

## Modos de edicion

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

## Estructura del proyecto

```
MAPA_FRENTES/
├── config.yaml                  # Configuracion general
├── requirements.txt
├── mapa_frentes/
│   ├── __main__.py              # python -m mapa_frentes
│   ├── config.py                # Carga YAML -> dataclasses
│   ├── data/                    # Descarga y lectura ECMWF
│   ├── fronts/                  # Modelos, TFP, clasificacion, generacion desde centros, GeoJSON
│   ├── analysis/                # Isobaras, centros H/L (con IDs)
│   ├── plotting/                # Renderizado Cartopy + MetPy
│   ├── gui/                     # PyQt5: ventana, mapa, editor, dialogos
│   └── utils/                   # Suavizado, funciones geo
├── scripts/
│   ├── download_latest.py       # CLI descarga
│   └── generate_map.py          # CLI generacion headless
├── data/
│   ├── cache/                   # GRIB2 descargados
│   ├── output/                  # Mapas exportados
│   └── sessions/                # Sesiones guardadas (GeoJSON)
└── tests/                       # pytest
```

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
- **pressure_centers**: tamano del filtro, distancia minima entre centros
- **tfp**: sigma de suavizado, umbral de gradiente, parametros DBSCAN
- **center_fronts**: radio de busqueda, paso de ray-marching, longitud maxima, suavizado
- **plotting**: tamano de figura, colores, grosor de lineas
- **export**: DPI para PNG y PDF

Tambien puedes cambiar estos parametros desde la GUI en menu Configuracion > Parametros.
