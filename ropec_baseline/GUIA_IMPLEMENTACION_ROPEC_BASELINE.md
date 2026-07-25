# Guía de implementación — `ropec_baseline`

> Guía para **reconstruir y comprender** el proyecto por tu cuenta, función por
> función, con las matemáticas, la estadística y las librerías reales que aparecen
> en el código. No es un tutorial para copiar: cada componente trae pseudocódigo,
> ejercicio, prueba y criterio de aceptación para que puedas **explicarlo, probarlo,
> modificarlo y reconstruirlo sin IA**.
>
> **Regla de honestidad de esta guía:** todo lo que sigue está extraído del código
> real (`utils.py`, `data.py`, `qc.py`, `splits.py`, `model.py`, `train.py`,
> `eval.py`, `run_experiment.py`, `run_multiseed.py`, `run_curve.py`,
> `failure_analysis.py`, `make_figures.py`, `tests/test_splits.py`). Cuando la
> documentación (`README.md`, `METHODS_as_run.md`, `config.yaml`) difiere del código,
> se marca explícitamente en la sección **Auditoría**. No se inventan funciones,
> métricas ni comportamientos.

---

## Tabla de contenido

1. [Descripción científica del baseline](#1-descripción-científica-del-baseline)
2. [Arquitectura del proyecto](#2-arquitectura-del-proyecto)
3. [Mapa de dependencias y de llamadas](#3-mapa-de-dependencias-y-de-llamadas)
4. [Bibliotecas vs scripts · entradas/salidas · artefactos](#4-bibliotecas-vs-scripts--entradassalidas--artefactos)
5. [Orden de implementación (verificado contra el código)](#5-orden-de-implementación-verificado-contra-el-código)
6. [Plantilla de explicación y funciones ancla](#6-plantilla-de-explicación-y-funciones-ancla)
7. [Explicación función por función](#7-explicación-función-por-función)
8. [Imports y librerías por archivo](#8-imports-y-librerías-por-archivo)
9. [Matemáticas y estadística que aparecen](#9-matemáticas-y-estadística-que-aparecen)
   - [9.1 Verificación de cada concepto: dónde estudiarlo y cómo checarlo](#91-verificación-de-cada-concepto-dónde-estudiarlo-y-cómo-checarlo)
10. [Pruebas incrementales obligatorias](#10-pruebas-incrementales-obligatorias)
11. [Sesiones de estudio](#11-sesiones-de-estudio)
12. [Comandos de ejecución](#12-comandos-de-ejecución)
13. [Artefactos esperados](#13-artefactos-esperados)
14. [Errores frecuentes](#14-errores-frecuentes)
15. [Auditoría para comprender, no para cambiar silenciosamente](#15-auditoría-para-comprender-no-para-cambiar-silenciosamente)
16. [Glosario](#16-glosario)
17. [Tabla maestra de funciones](#17-tabla-maestra-de-funciones)
18. [Checklist: «Puedo reconstruir ROPEC baseline sin IA»](#18-checklist-puedo-reconstruir-ropec-baseline-sin-ia)
19. [Anexo A — `eval.py` con plantilla completa de 8 pasos](#anexo-a--evalpy-con-plantilla-completa-de-8-pasos)
20. [Anexo B — primitivas de `train.py` con plantilla completa](#anexo-b--primitivas-de-trainpy-con-plantilla-completa)
21. [Anexo C — método general para replicar experimentos/papers](#anexo-c--método-general-para-replicar-experimentospapers)

---

## 1. Descripción científica del baseline

### 1.1 Pregunta experimental
¿La **procedencia del preentrenamiento** de un único backbone **ResNet-50** gobierna
el rendimiento (y la eficiencia de etiquetas) para **detectar presencia/ausencia de
fractura de platillo tibial** en radiografías del dataset **PlaTiF**?

### 1.2 Variable manipulada (la única) y comparadores
Se manipula **solo la inicialización de pesos**. Tres comparadores (en
`config.yaml → comparators`):

* `radimagenet` — pesos de RadImageNet (radiología general).
* `imagenet` — pesos de ImageNet (imagen natural).
* `random` — inicialización aleatoria (control).

Todo lo demás es **idéntico** entre comparadores: backbone, folds, preprocesamiento,
optimizador, épocas, presupuesto de tuning y semilla por fold. Esto es lo que
convierte el estudio en una **ablación controlada**.

### 1.3 Unidades de análisis (esto es lo más importante)

| Unidad | Qué es | Dónde vive en el código |
|---|---|---|
| **paciente** (`patient_id`) | Persona; un archivo `.mat`. | columna `patient_id` del índice |
| **vista** (`view`) | Una radiografía `imN` del paciente (1..8 por paciente). | columna `view` (`"im0"`, `"im1"`, …) |
| **etiqueta por vista** (`label`) | Schatzker 1..6 = fractura, 7 = normal. | columna `label` |
| **`y_vista`** | Binario `1[label ≠ 7]`. | columna `y_vista` |
| **rodilla** (`knee_id`) | `(patient_id, "fx"|"normal")`. **Unidad de evaluación.** | columna `knee_id` |
| **logit por vista** | Salida del modelo por radiografía. | columna `logit` en `pred_df` |
| **predicción por rodilla** | **Media de logits** de las vistas de esa `knee_id`. | `eval.aggregate_views_to_knee` |

**Por qué `knee_id ≠ patient_id`.** La regla congelada (`data.knee_id_of`) es
`knee_id = f"{patient_id}:{'fx' if label<=6 else 'normal'}"`. Un paciente con vistas
de **ambos lados** (una rodilla fracturada y la contralateral normal) genera **dos**
`knee_id` distintos con el **mismo** `patient_id`. En PlaTiF eso ocurre exactamente en
4 pacientes (`data.MIXED_SIDE_IDS = {92, 112, 133, 147}`). Por eso hay 186 pacientes
pero **190 rodillas**.

**Por qué los splits van por paciente.** Si dos vistas de la misma persona cayeran en
train y test, el modelo podría memorizar rasgos del paciente (fuga). Entonces todas
las vistas y **ambas** rodillas de un paciente van al **mismo fold**
(`StratifiedGroupKFold` agrupando por `patient_id`).

**Por qué se agrega por rodilla y se remuestrean pacientes.** El modelo produce un
logit **por vista**; se promedian los logits **dentro de cada `knee_id`** (regla
congelada antes de mirar métricas). La métrica primaria (AUROC) se calcula **a nivel
rodilla**. Para la incertidumbre, el bootstrap **remuestrea pacientes completos**
(cada paciente arrastra sus 1–2 rodillas), porque las rodillas de un mismo paciente no
son observaciones independientes.

### 1.4 Métricas
* **Primaria:** AUROC out-of-fold (OOF) a nivel rodilla (`eval.knee_metrics`).
* **Contraste:** `ΔAUROC = AUROC(A) − AUROC(B)` con **bootstrap pareado por pacientes**
  (`eval.paired_bootstrap_delta_auroc`): efecto observado + IC95% percentil + p-valor
  bootstrap de dos colas. Secundaria: AUPRC y prevalencia.

### 1.5 Fuentes de variabilidad (dos, distintas)
* **Optimización** — se estima con **múltiples semillas** (`run_multiseed.py`): media
  ± desviación del AUROC entre semillas.
* **Muestreo** — se estima con el **bootstrap** por pacientes. Multi-semilla **no**
  sustituye al bootstrap; miden cosas distintas. El multi-semilla **agrupa** (pool)
  las distribuciones bootstrap de todas las semillas para el IC final.

### 1.6 Invariantes que NO debes cambiar
1. Unidad rodilla y agregación = **media de logits** por `knee_id`, congelada.
2. Cero fuga: `StratifiedGroupKFold(5)` por `patient_id`; val interno; el test nunca
   selecciona épocas ni umbral.
3. Backbone único ResNet-50 (aserción en `build_model`).
4. **Aserción de conteo de tensores** en cada carga de pesos; RadImageNet exige
   **exactamente 318**.
5. Ground-truth real congelado (`assert_real_index`: 186/190/421, mixtos
   `{92,112,133,147}`).
6. QC congelado antes de mirar métricas; solo se cuentan duplicados **cross-patient**.
7. Nada de números inventados: los resultados salen de las corridas reales.

*(Fuera de alcance de este baseline y por tanto de esta guía: MURA, MAE, segmentación,
nnU-Net y demás módulos de la tesis.)*

---

## 2. Arquitectura del proyecto

Flujo real del pipeline (ajustado tras leer las dependencias reales):

```text
config.yaml (utils.load_config)
    ↓
build_index / build_real_index        (data.py)     ── modo toy | real
    ↓  (solo real)
run_qc → apply_exclusions             (qc.py)       ── cohorte post-QC
    ↓
make_folds (+ check_no_leakage)       (splits.py)   ── folds por paciente + val interno
    ↓
ViewDataset → DataLoader              (data.py, train.py)
    ↓
build_model (random|imagenet|radimagenet, load_matched_weights)  (model.py)
    ↓
train_one_fold                        (train.py)    ── early stopping + umbral
    ↓
pred_df: logits OOF POR VISTA         (train.py)
    ↓
aggregate_views_to_knee               (eval.py)     ── media de logits por knee_id
    ↓
knee_metrics: AUROC / AUPRC           (eval.py)
    ↓
paired_bootstrap_delta_auroc          (eval.py)     ── ΔAUROC, IC95%, p (por pacientes)
    ↓
run_once                              (run_experiment.py)  ── 1 semilla, todos los comparadores
    ↓
run_multiseed  |  run_curve           ── robustez de optimización | curva de eficiencia
    ↓
failure_analysis · make_figures       ── matriz de confusión, tablas y figuras
```

Diferencia clave respecto al diagrama de la consigna: el **QC** solo interviene en
`mode: real`; en `mode: toy` el índice se usa directo (no hay QC). Y `run_once` es el
**núcleo reutilizado**: `run_multiseed.py` y `run_curve.py` lo envuelven (multiseed
literalmente llama `run_once`; curve reimplementa el bucle interno usando las mismas
piezas de `train`/`eval`/`splits`).

---

## 3. Mapa de dependencias y de llamadas

### 3.1 Dependencias entre archivos (import interno)

```text
utils.py        → (no importa módulos locales)
data.py         → utils (solo en su bloque __main__/CLI)
qc.py           → data (build_real_index, load_platif_struct, read_view_image), utils
splits.py       → qc (build_qc_cohort, en su CLI _main), utils
model.py        → utils (solo CLI)
train.py        → data (ViewDataset, subset_index)
eval.py         → (solo numpy/pandas/sklearn)
run_experiment  → data, eval, model, splits, train, utils, (qc si mode real)
run_multiseed   → run_experiment (run_once), eval, utils, (qc si mode real)
run_curve       → eval, model, splits, train, utils, qc
failure_analysis→ data (_find_col)
make_figures    → (solo json/pandas/matplotlib; lee los JSON de salida)
tests/test_splits → data, splits
```

### 3.2 Grafo de llamadas (funciones clave)

```text
run_experiment.main
└─ run_once
   ├─ set_seed, save_config, dump_versions            (utils)
   ├─ build_qc_cohort  → build_real_index → assert_real_index   (qc, data)
   │                    → run_qc → _phash/_hamming/_read_view_arrays/_flow_diagram
   │                    → apply_exclusions
   │   (o build_index → build_toy_index en modo toy)
   ├─ find_mixed_patients                              (data)
   ├─ make_folds → _fold_prevalence → _prev           (splits)
   ├─ check_no_leakage, save_manifest                 (splits)
   └─ por comparador × fold:
      ├─ set_seed(seed+fold)
      ├─ build_model → load_matched_weights → remap_radimagenet_keys / _torch_load_state_dict  (model)
      ├─ train_one_fold                                (train)
      │  ├─ subset_index                               (data)
      │  ├─ _make_loader → ViewDataset(__getitem__ → _toy_image | _real_image → _base_image → _finalize)
      │  ├─ _build_optimizer, _pos_weight
      │  ├─ (bucle) model.train / BCEWithLogitsLoss / backward / step
      │  ├─ _predict_logits (val) → roc_auc_score → early stopping
      │  └─ _choose_threshold, _predict_logits (test)  → pred_df
      ├─ aggregate_views_to_knee                       (eval)
      └─ knee_metrics                                  (eval)
   └─ paired_bootstrap_delta_auroc → pooled_ci, bootstrap_pvalue   (eval)

run_multiseed.main → (por semilla) run_once(collect_boot=True) → _mean_std, pooled_ci, bootstrap_pvalue
run_curve.main     → make_folds, nested_stratified_subsets, patient_positivity,
                     (por seed×frac×comp×fold) build_model + train_one_fold,
                     aggregate_views_to_knee, knee_metrics, paired_bootstrap_delta_auroc
make_figures.main  → figure_efficiency_curve, figure_forest_delta100, table_provenance_metrics, _save
failure_analysis.main → youden_threshold, _find_col
```

---

## 4. Bibliotecas vs scripts · entradas/salidas · artefactos

| Archivo | Tipo | Entradas | Salidas / artefactos |
|---|---|---|---|
| `utils.py` | biblioteca | — | `versions.json`, `config_used.yaml` (vía otros) |
| `data.py` | biblioteca (+CLI) | `config.yaml`, `.mat`, `xlsx` | `platif_index.csv` (CLI) |
| `qc.py` | biblioteca (+CLI) | índice real, `.mat` | `qc_exclusions.csv`, `qc_flow.json` |
| `splits.py` | biblioteca (+CLI) | cohorte post-QC | `folds_manifest.json` |
| `model.py` | biblioteca (+CLI) | `config.yaml`, `.pt` RadImageNet | modelo en memoria |
| `train.py` | biblioteca | índice, fold, modelo | `pred_df` por vista (en memoria) |
| `eval.py` | biblioteca | `pred_df` / knee df | dicts de métricas |
| `run_experiment.py` | **script** | `config.yaml` | `index.csv`, `folds_manifest.json`, `<comp>/{fold_k_preds.csv, oof_knee_preds.csv, metrics.json}`, `delta_auroc.json`, `summary.json`, `versions.json`, `config_used.yaml`, `run.log`, `qc_exclusions.csv` |
| `run_multiseed.py` | **script** | `config.yaml` | `seed_<s>/…` (como run_once) + `multiseed_summary.json`, `multiseed.log` |
| `run_curve.py` | **script** | `config.yaml` | `curve/seed_<s>/frac_<f>/<comp>_oof.csv`, `curve_summary.json`, `curve.log` |
| `failure_analysis.py` | **script** | un `oof_knee_preds.csv`, `xlsx` | reporte por stdout |
| `make_figures.py` | **script** | `curve_summary.json`, `multiseed_summary.json` | `fig_efficiency_curve.(png|pdf)`, `fig_forest_delta100.(png|pdf)`, `table_provenance_metrics.csv` |
| `tests/test_splits.py` | test | — (índice sintético) | asserts (pytest o standalone) |

Tipos de artefacto por extensión: **CSV** (índice, exclusiones QC, preds por
fold/OOF, tabla de métricas), **JSON** (folds, flujo QC, métricas, deltas, summaries,
versiones), **PNG/PDF** (figuras), **log** (`.log`). **No hay checkpoints en disco**:
el «mejor estado» se guarda en RAM con `copy.deepcopy(model.state_dict())` dentro de
`train_one_fold` y se restaura al final del fold; nada se serializa a `.pt`.

---

## 5. Orden de implementación (verificado contra el código)

El orden de la consigna es correcto salvo dos ajustes que el código impone:

* **`youden_threshold`** vive en `failure_analysis.py`, no es una primitiva de
  entrenamiento; la selección de umbral del pipeline es `train._choose_threshold`
  (que implementa Youden/F1 inline). No los confundas.
* **`subset_index`** (que la consigna pone en la Etapa 2) es usada por `train.py` y
  `run_curve.py`; puedes implementarla en la Etapa 2 pero solo se ejercita en la 6/13.

Orden recomendado (cada etapa produce algo ejecutable y comprobable):

0. Entender el experimento (`README`, `METHODS_as_run`, `config.yaml`, `requirements`).
1. `utils.py` (infra).
2. `data.py` puro: `knee_id_of`, `y_from_label`, `build_index_from_view_records`,
   `find_mixed_patients`, `build_toy_index`, `subset_index` + `ViewDataset` toy.
3. `splits.py` + `tests/test_splits.py` (**no avanzar sin cero fuga verde**).
4. `ViewDataset` real completo (`_base_image`, `_finalize`, etc.).
5. `model.py` (random → imagenet → radimagenet con aserción de conteo).
6. Primitivas de `train.py` + `train_one_fold` (overfit de un lote antes de todo).
7. `eval.py` (agregación, métricas, bootstrap) con datos sintéticos calculables a mano.
8. `run_experiment.py` end-to-end en **modo toy** (`--smoke`).
9. Lectura real de PlaTiF (`build_real_index`, `assert_real_index`, `cross_check_xlsx`).
10. `qc.py`.
11. Experimento real completo.
12. `run_multiseed.py`.
13. `run_curve.py` (`patient_positivity`, `nested_stratified_subsets`).
14. `failure_analysis.py` + `make_figures.py`.

---

## 6. Plantilla de explicación y funciones ancla

Cada función se explica con esta plantilla (la versión **completa** — con
implementación guiada de 8 pasos — se aplica a las **funciones ancla** de cada etapa,
que concentran los conceptos; para las funciones hermanas se da una ficha compacta con
los mismos campos esenciales y se indica «aplica la plantilla de su ancla»).

**Campos:** Ubicación · Responsabilidad · Razón de existencia · Clasificación ·
Entradas · Salidas · Estado/efectos · Concepto de programación · Concepto
matemático/estadístico · Correspondencia código–matemáticas · Librerías/símbolos ·
Pseudocódigo · Implementación guiada (firma vacía → pasos → pseudocódigo → TODO →
pistas → prueba → solución → explicación) · Ejemplo mínimo · Prueba · Errores
frecuentes · Criterio para continuar · Preguntas de comprensión · Ejercicio de
reconstrucción.

**Funciones ancla por etapa** (las que reciben la plantilla completa en §7):
`knee_id_of` (Etapa 2) · `make_folds` + `check_no_leakage` (Etapa 3) ·
`ViewDataset._base_image`/`_finalize` (Etapa 4) · `load_matched_weights` (Etapa 5) ·
`train_one_fold` (+ `_pos_weight`, `_choose_threshold`) (Etapa 6) ·
`aggregate_views_to_knee` + `paired_bootstrap_delta_auroc` (Etapa 7) · `_phash`
(Etapa 10).

---

## 7. Explicación función por función

> Convención: **[ANCLA]** = plantilla completa con implementación guiada; **[ficha]** =
> campos esenciales + remisión a su ancla. Las líneas son aproximadas.

### Etapa 1 — `utils.py`

#### `set_seed(seed)` [ficha]
* **Ubicación:** `utils.py:20`. Llamada por: `run_once`, `run_curve.main`,
  `splits._main`. Llama a: `random.seed`, `np.random.seed`, `torch.manual_seed`.
* **Responsabilidad:** fijar todas las semillas (Python, NumPy, PyTorch, CUDA) y poner
  cuDNN determinista.
* **Razón de existencia:** reproducibilidad; sin esto, dos comparadores no compartirían
  el mismo orden de lotes ni la misma init del head → el contraste de procedencia
  dejaría de ser limpio. Evita el error silencioso «resultados que no reproducen».
* **Clasificación:** infraestructura / reproducibilidad.
* **Entradas:** `seed:int`. **Salidas:** `None`.
* **Efectos:** escribe `os.environ["PYTHONHASHSEED"]`, muta estado global de RNGs,
  fija `cudnn.deterministic=True`, `benchmark=False`. `import torch` perezoso dentro de
  `try/except`.
* **Concepto de programación:** estado global, import perezoso, manejo de excepciones.
* **Matemática/estadística:** *Esta función no implementa una operación matemática
  central; su importancia es de ingeniería/reproducibilidad.*
* **Librerías:** `random`, `numpy.random.seed`, `torch.manual_seed`,
  `torch.cuda.manual_seed_all`, `torch.backends.cudnn`.
* **Prueba:** `set_seed(0); a=np.random.rand(3); set_seed(0); b=np.random.rand(3);
  assert (a==b).all()`.
* **Criterio para continuar:** dos llamadas con la misma semilla producen la misma
  secuencia aleatoria.

#### `get_logger(name, logfile)` [ficha]
* **Ubicación:** `utils.py:40`. **Responsabilidad:** logger idempotente a stdout (+
  archivo opcional). **Razón:** trazabilidad por corrida (`run.log`) sin duplicar
  handlers si se llama dos veces (`if logger.handlers: return`). **Clasificación:** I/O /
  infra. **Efectos:** crea directorio del logfile, añade handlers. **Sin matemática.**
* **Símbolos:** `logging.getLogger/Formatter/StreamHandler/FileHandler`,
  `os.makedirs(exist_ok=True)`. **Continuar:** llamarlo dos veces no duplica líneas.

#### `load_config(path)` / `save_config(cfg, path)` [ficha]
* **Ubicación:** `utils.py:62` / `:67`. **Responsabilidad:** leer/escribir `config.yaml`.
  **Razón:** toda la parametrización vive en YAML (una sola fuente de verdad); guardar
  `config_used.yaml` por corrida documenta *qué* se ejecutó. **Símbolos:**
  `yaml.safe_load` (evita ejecutar objetos arbitrarios), `yaml.safe_dump(sort_keys=False)`
  (conserva el orden legible). **Sin matemática.** **Prueba:** round-trip
  `save_config(load_config(p), q)` produce YAML equivalente.

#### `_git_hash()` / `dump_versions(path)` [ficha]
* **Ubicación:** `utils.py:76` / `:87`. **Responsabilidad:** capturar commit + versiones
  de librerías + GPU/CUDA a `versions.json`. **Razón:** reproducibilidad exacta; un
  resultado sin entorno registrado no es auditable. **Efectos:** subprocess `git
  rev-parse`, `__import__` dinámico, escribe JSON. **Símbolos:** `subprocess.check_output`,
  `platform.platform`, `datetime.now().isoformat`. **Auditoría:** sondea `h5py` aunque el
  código no lo use (ver §15). **Sin matemática.**

#### `resolve_device(pref)` [ficha]
* **Ubicación:** `utils.py:118`. **Responsabilidad:** devolver `"cuda"`/`"cpu"` según
  preferencia y disponibilidad. **Razón:** portabilidad CPU/GPU. **Símbolos:**
  `torch.cuda.is_available`. **Sin matemática.**

---

### Etapa 2 — `data.py` (modelo conceptual de datos)

#### `knee_id_of(patient_id, label)` **[ANCLA]**

**Ubicación:** `data.py:51`. Llamada por: `build_index_from_view_records`. Llama a: —.
Etapa: construcción del índice.

**Responsabilidad.** Derivar el identificador de rodilla a partir de `(patient_id,
label)`.

**Razón de existencia.** Es la **regla de unidad experimental** materializada en
código. Problema científico: la etiqueta clínica es por vista, pero la evaluación debe
ser por rodilla; sin una regla determinista y congelada, la agregación vista→rodilla no
tendría a qué agrupar. Si no existiera, un paciente con dos lados se colapsaría en una
sola unidad y perderías la rodilla contralateral normal (sesgo). Evita el error
silencioso de mezclar lados.

**Clasificación.** Transformación de datos / definición de dominio.

**Entradas.**
* `patient_id:int` — id de paciente; procedencia: nombre del `.mat` o el toy; se
  castea con `int()`.
* `label:int` — Schatzker 1..7; válido 1..7; se castea con `int()`; no se modifica el
  argumento.

**Salidas.** `str` con formato `"{patient_id}:{fx|normal}"` (p. ej. `"92:fx"`).
Consumidor: la columna `knee_id`, usada por `aggregate_views_to_knee`,
`check_no_leakage`, el bootstrap.

**Estado/efectos.** Ninguno (función **pura**).

**Concepto de programación.** Función pura; f-string; casteo defensivo.

**Concepto matemático/estadístico.** Función indicadora sobre el umbral de Schatzker:
`side = fx` si `label ≤ 6`, `normal` si `label = 7`.

**Correspondencia código–matemáticas.** `side = "fx" if int(label) <= 6 else "normal"`
↔ `lado(label) = 𝟙[label ≤ 6]`.

**Librerías/símbolos.** Ninguna externa (a propósito: la regla de dominio no debe
depender de librerías).

**Pseudocódigo.**
```
si label <= 6: lado = "fx"   ; si no: lado = "normal"
devolver f"{patient_id}:{lado}"
```

**Implementación guiada.**
1. Firma vacía:
   ```python
   def knee_id_of(patient_id: int, label: int) -> str:
       ...
   ```
2. Pasos: (a) castear ambos a int; (b) decidir lado por umbral 6; (c) formatear.
3. Pseudocódigo: arriba.
4. Ejercicio `TODO`:
   ```python
   def knee_id_of(patient_id, label):
       # TODO: lado = "fx" si label<=6 si no "normal"; devolver "pid:lado"
       raise NotImplementedError
   ```
5. Pistas: usa `int(...)` para robustez; el separador es `":"`.
6. Prueba que debe pasar:
   ```python
   assert knee_id_of(92, 3) == "92:fx"
   assert knee_id_of(92, 7) == "92:normal"
   ```
7. Solución de referencia: la de `data.py:51-54`.
8. Explicación: el umbral 6 codifica «Schatzker 1..6 = fractura, 7 = normal»; el
   `patient_id` en el prefijo garantiza que **rodillas de pacientes distintos nunca
   colisionan** (clave para `check_no_leakage`).

**Ejemplo mínimo.** Paciente 92 con vistas `[3, 7, 5]` → knee_ids `{"92:fx",
"92:normal"}` (¡dos rodillas!).

**Prueba.** ver paso 6.

**Errores frecuentes.** Usar `<` en vez de `<=` (perderías Schatzker 6);
olvidar `int()` y comparar strings; usar `patient_id` como knee_id (colapsa lados).

**Criterio para continuar.** `knee_id_of` distingue fx/normal y ambos casos casan con
los strings esperados.

**Preguntas de comprensión.** (1) ¿Por qué el umbral es 6 y no 7? (2) ¿Qué pasa con un
paciente cuyas vistas son `[2,7]`? (3) ¿Por qué el `patient_id` debe ir en el id?
(4) ¿Es pura? ¿por qué importa?

**Ejercicio de reconstrucción.** Reescríbela usando un diccionario
`{True:"fx", False:"normal"}` en vez de `if`; verifica que pasa la misma prueba.

---

#### `y_from_label(label)` [ficha]
* **Ubicación:** `data.py:57`. **Responsabilidad:** `y = int(label != 7)`.
  **Matemática:** función indicadora `y = 𝟙[label ≠ 7]`. **Pura.** **Prueba:**
  `y_from_label(7)==0 and y_from_label(1)==1`. Aplica la plantilla de `knee_id_of`.

#### `build_index_from_view_records(records)` [ficha]
* **Ubicación:** `data.py:61`. Llamada por: `build_toy_index`, `build_real_index`,
  el test sintético. **Responsabilidad:** de una lista `{patient_id, view, label}` a un
  DataFrame **una fila por vista** con `knee_id`, `side_class`, `label`, `y_vista`,
  ordenado por `(patient_id, view)`.
* **Razón:** una fila por vista permite (a) agregar por `knee_id`, (b) contar
  prevalencias a tres niveles, (c) mapear predicciones de vuelta por índice de fila.
* **Concepto:** comprensión/loop → `pd.DataFrame`; `sort_values(...).reset_index(drop=True)`.
* **Símbolos:** `pandas.DataFrame`, `DataFrame.sort_values`, `reset_index`.
* **Sin matemática central.** **Prueba:** ver test #3 (§10).

#### `find_mixed_patients(df)` [ficha]
* **Ubicación:** `data.py:80`. **Responsabilidad:** conjunto de `patient_id` con **>1**
  `side_class` (aportan 2 rodillas). **Concepto matemático:** cardinalidad de lados
  distintos por paciente `> 1`. **Símbolos:** `DataFrame.groupby(...).nunique()`.
  **Razón científica:** `run_once` **asserta** `len(mixed) >= 1` (el toy debe ejercitar
  la derivación de `knee_id`). **Prueba:** test #4 (§10).

#### `build_toy_index(cfg, seed)` [ficha]
* **Ubicación:** `data.py:89`. **Responsabilidad:** sintetizar un índice PlaTiF-forme
  (vistas variables 1..8, ≥`n_mixed` pacientes mixtos garantizados). **Concepto:** RNG
  con `np.random.default_rng(seed)`; construcción por casos (mixto/positivo/negativo).
  **Razón:** ejercer el pipeline entero **sin datos reales**. **Símbolos:**
  `numpy.random.default_rng`, `rng.choice/integers/random`. **Prueba:** genera un índice,
  `find_mixed_patients` devuelve ≥ `n_mixed`.

#### `subset_index(index_df, patient_ids)` [ficha]
* **Ubicación:** `data.py:425`. **Responsabilidad:** filtrar el índice a un conjunto de
  pacientes. **Razón:** construir train/val/test y los subconjuntos de la curva
  **por paciente** (nunca por vista → evita fuga). **Símbolos:** `DataFrame.isin`.
  **Sin matemática.** **Prueba:** `set(subset_index(df, {1,2}).patient_id) <= {1,2}`.

*(Las funciones de lectura real `_patient_id_from_path`, `list_platif_files`,
`load_platif_struct`, `_view_names`, `read_view_label`, `read_view_image`,
`build_real_index`, `assert_real_index`, `_find_col`, `cross_check_xlsx`, `build_index`
se documentan en la Etapa 9.)*

---

### Etapa 3 — `splits.py` (particiones y cero fuga)

#### `make_folds(index_df, n_folds, internal_val_folds, seed)` **[ANCLA]**

**Ubicación:** `splits.py:22`. Llamada por: `run_once`, `run_curve.main`,
`splits._main`, el test. Llama a: `StratifiedGroupKFold`, `_fold_prevalence`.
Etapa: particiones.

**Responsabilidad.** Producir una lista de folds; cada fold es un dict con
`train_patients`, `val_patients`, `test_patients` (listas de `patient_id`) y
`prevalence`.

**Razón de existencia.** Materializa la **cero fuga**: agrupa por `patient_id`
(externo, test) y vuelve a agrupar dentro del train para separar un **val interno**.
Sin val interno, el early stopping y el umbral se elegirían mirando el test (fuga de
selección). Evita el error científico silencioso de «tunear contra el test».

**Clasificación.** Estadística (validación cruzada) / transformación de datos.

**Entradas.**
* `index_df` — índice una-fila-por-vista (necesita `y_vista`, `patient_id`).
* `n_folds:int` — folds externos (config: 5).
* `internal_val_folds:int` — divide el train para el val (config: 5 → ~20% val).
* `seed:int` — reproducibilidad del barajado.

**Salidas.** `list[dict]`. Consumidor: `check_no_leakage`, `save_manifest`,
`train_one_fold` (usa las listas de pacientes).

**Estado/efectos.** Ninguno (no escribe archivos; `save_manifest` lo hace aparte).

**Concepto de programación.** Uso de `next(iterator)` para tomar **solo la primera**
partición interna; conjuntos para restar `train_all - val`.

**Concepto matemático/estadístico.** *k-fold estratificado y agrupado.* Los pacientes
se reparten en `k` folds disjuntos que cubren todo el conjunto; la **estratificación**
intenta preservar la prevalencia y el **agrupamiento** garantiza que un grupo
(paciente) no se parta entre folds.

**Correspondencia código–matemáticas.**
* `sgkf.split(X, y, groups)` con `y = df["y_vista"]`, `groups = df["patient_id"]` ↔ la
  restricción «mismo grupo ⇒ mismo fold» + estratificación por `y`.
* El split **interno** repite `StratifiedGroupKFold` sobre solo el train, con
  `random_state = seed + 1` (semilla distinta para no replicar la partición externa),
  y toma la **primera** división como (train, val).

**Librerías/símbolos.** `sklearn.model_selection.StratifiedGroupKFold` — provee la
partición estratificada-agrupada sin implementarla a mano; supuesto: hay suficientes
grupos por clase. Alternativa: `GroupKFold` (sin estratificar) — no elegida porque no
preserva prevalencia. `numpy.zeros(len(df))` como `X` ficticio (el split solo necesita
`y` y `groups`).

**Pseudocódigo.**
```
sgkf = StratifiedGroupKFold(n_folds, shuffle, seed)
para cada (tr_idx, te_idx) en sgkf.split(0, y_vista, patient_id):
    test_patients = pacientes en te_idx
    train_all     = pacientes en tr_idx
    inner = StratifiedGroupKFold(internal_val_folds, shuffle, seed+1)
    (i_tr, i_va) = primera partición de inner sobre solo train_all
    val_patients   = pacientes en i_va
    train_patients = train_all − val_patients
    guardar dict(fold, train/val/test, prevalence=_fold_prevalence(...))
```

**Implementación guiada.** (1) firma vacía; (2) construye `X,y,groups`; (3) itera el
split externo; (4) **subselecciona** el train y corre el split interno con `seed+1`;
(5) usa `next(...)` para la primera partición; (6) resta conjuntos; (7) adjunta
prevalencias. **TODO**: implementar el bloque interno. **Pista:** `tr_df =
df[df.patient_id.isin(train_all)]`. **Prueba:** las de `test_splits.py`. **Solución:**
`splits.py:22-55`. **Explicación:** `seed+1` evita que la partición interna coincida
con la externa; `next` basta porque solo necesitas **una** división train/val.

**Ejemplo mínimo.** 60 pacientes, `n_folds=5` → 5 folds; cada paciente aparece en
exactamente un `test_patients`.

**Prueba.** `test_no_patient_in_two_test_folds`, `test_train_val_test_disjoint`.

**Errores frecuentes.** Estratificar por paciente en vez de por vista (aquí `y` es por
**vista**; ver Auditoría sobre `_patient_labels`); reusar `seed` en el split interno
(replicaría la partición); olvidar `.reset_index(drop=True)` antes del split interno.

**Criterio para continuar.** `check_no_leakage(make_folds(...))` pasa sin excepción.

**Preguntas.** (1) ¿Por qué agrupar por `patient_id` y no por `knee_id`? (2) ¿Por qué
`seed+1` dentro? (3) ¿Por qué el val sale del train y no del pool completo? (4) ¿Qué
estratifica exactamente `y` aquí, vista o paciente? (5) ¿Por qué `next(...)`?

**Ejercicio de reconstrucción.** Cambia `internal_val_folds` a 4 y observa cómo cambia
el tamaño del val; verifica que la cero fuga sigue verde.

---

#### `check_no_leakage(folds, index_df)` **[ANCLA]**

**Ubicación:** `splits.py:103`. Llamada por: `run_once`, `run_curve.main`,
`splits._main`, el test. Etapa: particiones (guardia).

**Responsabilidad.** Levantar `AssertionError` si (1) algún paciente está en >1 fold de
test, (2) la cobertura de test no es total, (3) hay solape train/val/test dentro de un
fold, o (4) existe un `knee_id` compartido entre pacientes.

**Razón de existencia.** Es la **red de seguridad científica**: convierte la invariante
«cero fuga» en una aserción ejecutable. Sin ella, una fuga sutil (p. ej. una rodilla
huérfana) inflaría el AUROC de forma indetectable. Evita el peor error silencioso del
proyecto.

**Clasificación.** Control de calidad / estadística (integridad de particiones).

**Entradas.** `folds` (salida de `make_folds`), `index_df`. **Salidas.** `None` (o
excepción). **Efectos.** Ninguno salvo lanzar excepción.

**Concepto de programación.** Programación defensiva con `assert`; conjuntos para
detectar solapes; `groupby(...).nunique()` para detectar rodillas compartidas.

**Concepto matemático/estadístico.** Propiedades de **partición**: los
`test_patients` forman una partición de todos los pacientes (disjuntos + cobertura
total, `⋃ = P`, `∩ = ∅`); dentro de un fold, train/val/test disjuntos; la función
`knee_id → patient_id` es **inyectiva** (ninguna rodilla pertenece a dos pacientes).

**Correspondencia código–matemáticas.**
* `dupes = [p for p,c in test_counts if c>1]; assert not dupes` ↔ folds de test
  disjuntos.
* `assert covered == all_patients` ↔ cobertura total.
* `knee_owner = groupby("knee_id")["patient_id"].nunique(); assert (knee_owner==1).all`
  ↔ inyectividad `knee_id→patient`.

**Librerías/símbolos.** `set`, `dict.get`, `DataFrame.groupby(...).nunique()`.

**Pseudocódigo / guiada / prueba.** Implementa los tres chequeos en orden; **TODO**:
el chequeo de rodilla huérfana. **Pista:** `groupby("knee_id")["patient_id"].nunique()`
debe ser todo 1. **Prueba:** `test_check_no_leakage_passes`, `test_no_orphan_knee`.
**Solución:** `splits.py:103-132`.

**Errores frecuentes.** Comparar listas en vez de conjuntos; olvidar el chequeo de
cobertura (una fuga por omisión); no verificar rodillas compartidas (la más sutil).

**Criterio para continuar.** Con folds correctos no lanza; si inyectas a mano una
rodilla compartida, **debe** lanzar.

**Preguntas.** (1) ¿Qué invariante protege el chequeo (4)? (2) ¿Por qué la cobertura
total importa además de la disjunción? (3) ¿Podría pasar (1)–(3) y fallar (4)?

**Ejercicio.** Crea un índice donde dos pacientes compartan `knee_id` (a mano) y
comprueba que salta la aserción correcta.

---

#### `_prev(df, patients)` / `_fold_prevalence(df, train_p, val_p, test_p)` [ficha]
* **Ubicación:** `splits.py:58` / `:71`. **Responsabilidad:** contar pacientes/rodillas/
  vistas y **prevalencias** (paciente y rodilla) de un subconjunto. **Matemática:**
  prevalencia = media de `y_vista` sobre rodillas únicas `= n_pos / n_total`.
  **Símbolos:** `groupby("patient_id")["y_vista"].max()` (etiqueta paciente),
  `drop_duplicates("knee_id")`. **Consumidor:** `folds_manifest.json` y el reporte.

#### `patient_positivity(index_df)` [ficha]
* **Ubicación:** `splits.py:75`. **Responsabilidad:** `patient_id → bool` (positivo si
  alguna vista fx). **Matemática:** `max` de `y_vista` por paciente = OR lógico.
  **Usada por:** `run_curve` (para estratificar los subconjuntos anidados). **Prueba:**
  paciente con una vista fx → `True`.

#### `nested_stratified_subsets(patients, is_pos, fractions, seed)` **[ANCLA]**

**Ubicación:** `splits.py:81`. Llamada por: `run_curve.main`. Etapa: curva de eficiencia.

**Responsabilidad.** Para un conjunto de pacientes de train, devolver subconjuntos
**anidados** y **estratificados** por fracción: `{f: [patient_id, …]}` con
`10% ⊂ 25% ⊂ 50% ⊂ 100%`, ≥1 por clase.

**Razón de existencia.** La curva de eficiencia debe comparar **el mismo dato**
creciente entre comparadores; si los subconjuntos no fueran anidados, la variación
entre fracciones mezclaría «más datos» con «otros datos». Anidar aísla el efecto del
**tamaño** de etiquetas.

**Clasificación.** Estadística (muestreo estratificado anidado) / transformación.

**Entradas.** `patients` (lista), `is_pos` (dict `patient→bool`), `fractions` (lista
de floats), `seed`. **Salidas.** `dict{float: list[int]}`.

**Concepto matemático.** Se **barajan** positivos y negativos por separado (con la
misma semilla) y para cada fracción `f` se toma un **prefijo** de tamaño
`max(1, ⌈f·n_clase⌉)` de cada clase. Como los prefijos crecen monótonamente, los
subconjuntos son **anidados**; tomar de cada clase por separado **preserva la
prevalencia** (estratificación).

**Correspondencia código–matemáticas.** `n_pos = max(1, ceil(f*len(pos)))`;
`out[f] = pos[:n_pos] + neg[:n_neg]` ↔ prefijos crecientes ⇒ `S_{f1} ⊆ S_{f2}` si
`f1 ≤ f2`.

**Símbolos.** `numpy.random.default_rng(seed)`, `rng.shuffle`, `math.ceil`.

**Guiada / prueba.** **TODO:** baraja pos y neg, toma prefijos. **Pista:** ordena
`fractions` ascendente para razonar el anidamiento. **Prueba:**
`S_10 ⊆ S_25 ⊆ S_50 ⊆ S_100` y cada uno tiene ≥1 pos y ≥1 neg.

**Errores frecuentes.** Barajar el conjunto entero (rompe estratificación); no fijar
mínimo 1 (una fracción baja podría dejar una clase vacía → AUROC indefinido); usar
semillas distintas por clase (rompe reproducibilidad).

**Criterio para continuar.** Los cuatro subconjuntos anidan y preservan ambas clases.

**Preguntas.** (1) ¿Por qué prefijos y no muestreo independiente por fracción?
(2) ¿Qué rompería no estratificar? (3) ¿Por qué `max(1, …)`?

#### `save_manifest(folds, index_df, path, seed)` [ficha]
* **Ubicación:** `splits.py:135`. **Responsabilidad:** serializar folds + tamaños a
  `folds_manifest.json`. **Razón:** reproducibilidad/auditoría de particiones.
  **Símbolos:** `json.dump(indent=2)`. **Sin matemática.**

---

### Etapa 4 — `ViewDataset` (tensores y preprocesamiento)

Recorrido completo **fila del índice → tensor a ResNet-50**:
`__getitem__(i)` lee `self.df.iloc[i]` → si `toy`, `_toy_image`; si `real`,
`_real_image` → `_base_image` (carga `OriginalImage`, normaliza percentiles 1–99 a
`[0,1]`, resize bilineal a `S×S`, cachea) → (augment opcional) → `_finalize` (repite a
3 canales, estandariza con media/std de ImageNet) → devuelve `(img 3×S×S, y, i)`.

#### `ViewDataset.__init__ / __len__ / __getitem__` [ficha]
* **Ubicación:** `data.py:324 / 355 / 414`. **Responsabilidad:** implementar el
  **protocolo Dataset** de PyTorch (indexable + longitud) a nivel **vista**.
* **Concepto de programación:** protocolo `Dataset` (`__len__`+`__getitem__`), import
  perezoso de torch, caché por instancia (`_struct_cache` LRU de 8) + caché global
  (`_REAL_IMAGE_CACHE`). **Efectos:** llena cachés; mueve nada a device (eso lo hace el
  loop). **Salida de `__getitem__`:** `(tensor 3×S×S, tensor escalar y, int i)`; el
  tercer elemento `i` permite mapear predicciones de vuelta a filas.
* **Prueba:** test #9 (§10).

#### `ViewDataset._base_image(mat_path, view)` **[ANCLA]**

**Ubicación:** `data.py:377`. Llamada por: `_real_image`. Llama a: `_get_struct`,
`read_view_image`, `numpy.percentile/clip`, `F.interpolate`. Etapa: preprocesamiento.

**Responsabilidad.** Convertir `OriginalImage` (H×W float) en un tensor `1×S×S` en
`[0,1]`, con caché.

**Razón de existencia.** Estandariza el rango de intensidades y el tamaño **idéntico
para todos los comparadores** (invariante: misma preproc). Sin normalización robusta,
radiografías con exposiciones distintas darían escalas incomparables; sin resize, no
entran a ResNet-50 (espera `224×224`).

**Clasificación.** Aprendizaje profundo (preproc) / transformación / matemática (ligera).

**Entradas.** `mat_path:str`, `view:str`. **Salida.** tensor `torch.float32` de forma
`(1, S, S)` en `[0,1]`. **Consumidor:** `_finalize`.

**Estado/efectos.** Lee del `_REAL_IMAGE_CACHE` global; si falta, calcula y **escribe**
la caché (clave `(mat_path, view, img_size)`). El objeto cacheado es de solo lectura
(la augmentation crea tensores nuevos).

**Concepto matemático/estadístico.** *Normalización robusta por percentiles.* Con
`lo = P1(img)`, `hi = P99(img)`:
```
x̂ = clip( (x − lo) / (hi − lo), 0, 1 )
```
Los percentiles 1 y 99 recortan outliers (píxeles saturados) mejor que min/max. El
**resize bilineal** interpola en una malla `S×S`.

**Correspondencia código–matemáticas.**
`lo, hi = np.percentile(img, 1), np.percentile(img, 99)` ↔ `P1, P99`;
`np.clip((img-lo)/(hi-lo), 0, 1)` ↔ la fórmula `x̂`;
`F.interpolate(..., mode="bilinear")` ↔ remuestreo bilineal a `S×S`.

**Librerías/símbolos.** `numpy.percentile` (cuantiles empíricos), `numpy.clip`
(recorte), `torch.from_numpy` (comparte memoria host), `torch.nn.functional.interpolate`
(resize diferenciable) con `align_corners=False`. Alternativa a percentiles: min/max —
no elegida por sensibilidad a outliers.

**Guiada / prueba.** **TODO:** implementar normalización + resize. **Pista:** cuida
`hi<=lo` (imagen constante) → `hi = lo + 1e-6`. **Prueba:** salida en `[0,1]`, forma
`(1,S,S)`. **Solución:** `data.py:377-398`.

**Errores frecuentes.** Dividir por cero cuando `hi==lo`; olvidar `astype(float32)`;
pasar 2D a `interpolate` (espera `N×C×H×W`, por eso `[None, None]`); mutar el tensor
cacheado (contaminaría otras épocas).

**Criterio para continuar.** Dos llamadas con la misma `(mat_path, view)` devuelven el
**mismo** objeto (caché) y el rango es `[0,1]`.

**Preguntas.** (1) ¿Por qué P1–P99 y no min–max? (2) ¿Por qué la caché exige
`num_workers=0`? (3) ¿Por qué `[None, None]` antes de `interpolate`?

#### `ViewDataset._finalize(x_1hw)` [ficha]
* **Ubicación:** `data.py:400`. **Responsabilidad:** `1×S×S [0,1]` → `3×S×S`
  estandarizado con **media/std de ImageNet**. **Matemática:** `z = (x − μ)/σ` canal a
  canal con `μ=(0.485,0.456,0.406)`, `σ=(0.229,0.224,0.225)`; replicación de 1→3
  canales (`repeat`). **Razón:** los pesos preentrenados (ImageNet/RadImageNet) esperan
  esta estandarización; aplicarla a los **tres** comparadores mantiene la preproc
  idéntica. **Símbolos:** `tensor.repeat`, broadcasting `view(3,1,1)`. **Prueba:** salida
  `3×S×S`, media≈por-canal desplazada.

#### `_toy_image / _get_struct / _real_image` [ficha]
* `_toy_image` (`data.py:358`): ruido `torch.randn` **determinista por (patient_id,
  view)** con un `Generator` sembrado; añade `±0.5` en el canal 0 según `y` (señal débil
  para que el smoke aprenda algo distinto de azar). **No es realista**, es plumbing.
* `_get_struct` (`data.py:368`): caché LRU (máx 8) de structs `.mat` por path.
* `_real_image` (`data.py:408`): `_base_image` → (augment si activo) → `_finalize`.

---

### Etapa 5 — `model.py`

#### `load_matched_weights(model, source_sd, name, expect_exact, expect_min, logger)` **[ANCLA]**

**Ubicación:** `model.py:62`. Llamada por: `build_model`. Etapa: carga del modelo.

**Responsabilidad.** Copiar **in-place** los tensores de `source_sd` cuyo **nombre y
forma** coinciden con los del modelo; contar cuántos se cargaron; **abortar** si el
conteo no cumple lo esperado.

**Razón de existencia.** El *landmine* del proyecto: con `strict=False` un remap mal
hecho descartaría todo en silencio y **entrenarías `random` creyendo que es
RadImageNet**. La aserción de conteo convierte ese error silencioso en un abort ruidoso.
Es la guardia científica de la variable manipulada.

**Clasificación.** Aprendizaje profundo / control de calidad / I/O de pesos.

**Entradas.** `model:nn.Module`; `source_sd:OrderedDict` (state_dict fuente ya
remapeado); `name:str` (etiqueta para logs/errores); `expect_exact:int|None`,
`expect_min:int|None` (contrato de conteo). **Salida.** `int` = nº de tensores cargados.

**Estado/efectos.** **Modifica el modelo** (`model.load_state_dict(new_sd, strict=True)`);
loggea el desglose; puede lanzar `AssertionError`.

**Concepto de programación.** Comparación de `state_dict` por clave y forma; construir
un `new_sd` mezclando fuente y destino; `assert` como contrato.

**Concepto matemático.** *Esta función no implementa una operación matemática central;
su valor es de ingeniería/integridad.* (El «conteo» es cardinalidad de un conjunto de
tensores que casan nombre+forma.)

**Correspondencia código–conteo.** `loaded += 1` cada vez que `k in source_sd` **y**
`source_sd[k].shape == v.shape`; `assert loaded == expect_exact` (RadImageNet: 318) o
`assert loaded >= expect_min` (ImageNet: todo el extractor salvo `fc.`).

**Librerías/símbolos.** `nn.Module.state_dict()` (mapa nombre→tensor de parámetros
**y** buffers), `Tensor.shape`, `Tensor.clone`, `load_state_dict(strict=True)`.

**Pseudocódigo.**
```
new = {}
para (k, v) en model.state_dict():
    si k en source y source[k].shape == v.shape: new[k] = source[k].clone(); loaded++
    si no: new[k] = v            # conserva el valor del modelo (init)
loggear desglose
si expect_exact: assert loaded == expect_exact
si expect_min:   assert loaded >= expect_min
model.load_state_dict(new, strict=True)
devolver loaded
```

**Implementación guiada.** **TODO:** el bucle de casado + las dos aserciones. **Pista:**
compara **shape**, no solo nombre (un nombre que casa con forma distinta NO cuenta).
**Prueba:** tests #12–#14 (§10). **Solución:** `model.py:62-117`. **Explicación:**
`strict=True` al final garantiza que `new_sd` tiene **todas** las claves del modelo
(las no casadas conservan su init); el `fc.` reinicializado es esperado (la fuente es
solo extractor).

**Ejemplo mínimo.** Cargar ImageNet: `loaded` = nº de tensores del extractor;
`expect_min` = `len([k for k in model.state_dict() if not k.startswith("fc.")])`.

**Prueba.** Forzar RadImageNet con un remap roto → `AssertionError` (test #14).

**Errores frecuentes.** Casar por nombre sin comprobar forma (cargarías basura);
olvidar `.clone()` (aliasing con la fuente); usar `strict=False` sin contar
(**el error silencioso** que esta función existe para evitar).

**Criterio para continuar.** ImageNet carga `≥ expect_min`; RadImageNet carga
**exactamente 318**; un remap roto **aborta**.

**Preguntas.** (1) ¿Por qué comparar shape además de nombre? (2) ¿Qué diferencia hay
entre `expect_exact` y `expect_min` y por qué RadImageNet usa exact? (3) ¿Por qué está
bien que `fc.` quede sin cargar? (4) ¿Qué es un buffer vs un parámetro en el
state_dict?

**Ejercicio.** Cambia una clave del `source_sd` para que no case; verifica que el
conteo baja y (con `expect_exact`) aborta.

#### `remap_radimagenet_keys(state_dict)` [ficha]
* **Ubicación:** `model.py:29`. **Responsabilidad:** renombrar prefijos
  `backbone.{0,1,4,5,6,7}.` → `conv1./bn1./layer1..4.` (torchvision Sequential →
  nombres timm). **Razón:** sin el remap, **ninguna** clave casaría y la aserción de
  318 abortaría (correctamente). **Concepto:** diccionario de reemplazo + `str.startswith`.
  **Sin matemática.** **Prueba:** una clave `backbone.0.weight` → `conv1.weight`.

#### `_torch_load_state_dict(path, logger)` [ficha]
* **Ubicación:** `model.py:41`. **Responsabilidad:** `torch.load` robusto al cambio de
  `weights_only` en torch≥2.6 (intenta `True`, cae a `False` para el `.pt` confiable).
  **Concepto:** manejo de excepciones en cascada. **Símbolos:** `torch.load(map_location
  ="cpu")`. **Sin matemática.**

#### `build_model(cfg, weights, logger)` [ficha]
* **Ubicación:** `model.py:123`. **Responsabilidad:** construir ResNet-50 (timm) para el
  comparador `weights∈{random,imagenet,radimagenet}` con **un solo logit**
  (`num_classes=1`). **Aserción:** `backbone == "resnet50"`. **Flujo:** random = crear sin
  pesos; imagenet = crear + copiar state_dict de `timm.create_model(pretrained=True)` con
  `expect_min`; radimagenet = cargar `.pt`, remapear, `expect_exact=318`. **Símbolos:**
  `timm.create_model`. **Prueba:** tests #11–#14. Aplica la plantilla de
  `load_matched_weights`.

---

### Etapa 6 — `train.py`

#### `train_one_fold(cfg, full_index, fold, weights, model, device, logger, train_patients=None)` **[ANCLA]**

**Ubicación:** `train.py:93`. Llamada por: `run_once`, `run_curve.main`. Llama a:
`subset_index`, `_make_loader`, `_pos_weight`, `_build_optimizer`, `_predict_logits`,
`_choose_threshold`, `BCEWithLogitsLoss`, `roc_auc_score`. Etapa: entrenamiento por fold.

**Responsabilidad.** Entrenar **un** comparador en **un** fold con early stopping sobre
el AUROC de val interno, restaurar el mejor estado, elegir umbral (informativo) y
devolver `(pred_df por vista sobre el test, threshold)`.

**Razón de existencia.** Encapsula el bucle idéntico para todos los comparadores;
garantiza que el **test nunca** se usa para seleccionar época ni umbral. El parámetro
`train_patients` permite reusar exactamente el mismo bucle para la **curva** (subconjunto
de train) sin tocar val/test.

**Clasificación.** Aprendizaje profundo / orquestación.

**Entradas.** `cfg`; `full_index` (cohorte completa); `fold` (dict con listas de
pacientes); `weights` (etiqueta comparador); `model` (ya construido); `device`;
`logger`; `train_patients` (opcional, subconjunto de train). **Salida.** `(pred_df,
threshold)`: `pred_df` tiene `patient_id, view, knee_id, side_class, y_vista, logit,
fold, weights` (**una fila por vista** del test).

**Estado/efectos.** Muta el modelo (entrena, mueve a device, `train()/eval()`); guarda
el mejor `state_dict` en RAM (`copy.deepcopy`) y lo restaura; **no** escribe archivos.

**Concepto de programación.** Bucle de entrenamiento; `copy.deepcopy` para snapshot;
early stopping con contador de paciencia.

**Concepto matemático/estadístico (denso).**
* **Pérdida:** `BCEWithLogitsLoss` = sigmoide + binary cross-entropy en una operación
  numéricamente estable. Para logit `z`, etiqueta `y∈{0,1}`:
  `ℓ = −[ y·log σ(z) + (1−y)·log(1−σ(z)) ]`, con `σ(z)=1/(1+e^{−z})`.
* **`pos_weight`** = `n_neg/n_pos` (del train), que multiplica el término positivo para
  compensar el desbalance.
* **Optimización:** Adam actualiza los parámetros con el gradiente `∂ℓ/∂θ` obtenido por
  retropropagación (regla de la cadena).
* **Selección de modelo:** se guarda el estado con **mayor AUROC de val**; AUROC es el
  área bajo la curva ROC (ver §9).

**Correspondencia código–matemáticas.**
`criterion = BCEWithLogitsLoss(pos_weight=...)` ↔ la `ℓ` de arriba;
`opt.zero_grad(); loss.backward(); opt.step()` ↔ un paso de descenso;
`if score > best_auc: best_state = deepcopy(...)` ↔ argmax sobre épocas del AUROC de val.

**Librerías/símbolos.** `torch.nn.BCEWithLogitsLoss`, `torch.optim.Adam` (vía
`_build_optimizer`), `torch.no_grad` (en `_predict_logits`), `sklearn.metrics.
roc_auc_score`, `copy.deepcopy`.

**Pseudocódigo.**
```
tr = subset(train_patients | fold.train); va = subset(fold.val); te = subset(fold.test)
loaders(tr shuffle=True, va/te shuffle=False)
pos_weight = n_neg/n_pos (si class_weighted)
criterion = BCEWithLogitsLoss(pos_weight); opt = Adam(grupos)
best_auc=-1
para epoch en epochs:
    model.train(); para lote: zero_grad; loss=criterion(model(x), y); backward; step
    v_logits,v_y = predict(val); v_auc = roc_auc(v_y, v_logits)
    si v_auc > best_auc: best=deepcopy(state); patience=0
    si no: patience++ ; si patience>=max y epoch+1>=min_epochs: break
restaurar best
threshold = _choose_threshold(val); (t_logits,t_y,t_idx)=predict(test)
pred_df = te.iloc[t_idx][cols] + logit + fold + weights
```

**Implementación guiada.** **TODO:** el bucle de época + early stopping. **Pista:** usa
`copy.deepcopy(model.state_dict())` (no una referencia). **Prueba:** overfit de un
subconjunto diminuto → `train_loss` baja; el AUROC de val sube. **Solución:**
`train.py:93-166`.

**Ejemplo mínimo.** En toy con `--smoke`, un fold entrena y devuelve un `pred_df` con
tantas filas como vistas de test.

**Errores frecuentes.** Guardar `state_dict` por referencia (se sobrescribe → restauras
basura); usar el test para early stopping (fuga); olvidar `model.eval()` en predicción;
no mover `y` a device.

**Criterio para continuar.** Sobre un lote diminuto el modelo **sobreajusta** (loss→0);
en toy el AUROC de val sube por encima de 0.5.

**Preguntas.** (1) ¿Por qué el early stopping mira val y no test? (2) ¿Por qué
`deepcopy`? (3) ¿Por qué el umbral no afecta al AUROC? (4) ¿Qué mide `pos_weight`?
(5) ¿Por qué `pred_df` es por vista y no por rodilla?

**Ejercicio.** Cambia `early_stopping_patience` a 1 y observa cuántas épocas corre.

#### `_pos_weight(index_df, device)` [ficha]
* **Ubicación:** `train.py:47`. **Matemática:** `pos_weight = n_neg / n_pos` (a nivel
  **vista** del train). Devuelve `None` si alguna clase falta. **Razón:** compensa el
  desbalance en la BCE. **Prueba:** test #15. **Símbolos:** `torch.tensor(...,
  device=device)`.

#### `_choose_threshold(logits, y, metric)` [ficha]
* **Ubicación:** `train.py:68`. **Responsabilidad:** umbral sobre **probabilidades** del
  val que maximiza **Youden** (`sens+spec−1`) o **F1**. **Matemática:** barre todos los
  valores únicos de prob; para cada uno arma la matriz de confusión (TP/FP/FN/TN) y
  calcula el score. **Importante:** el umbral es **informativo** (matriz de confusión) y
  **no** afecta al AUROC (invariante a transformaciones monótonas). **Auditoría:**
  `order = np.argsort(-probs)` se calcula y **no se usa** (§15). **Prueba:** test #19.

#### `_make_loader / _build_optimizer / _predict_logits` [ficha]
* `_make_loader` (`train.py:18`): envuelve `ViewDataset` en `DataLoader`
  (`batch_size`, `shuffle`, `num_workers`, `drop_last=False`). **Auditoría:** el
  parámetro `full_index` **no se usa** dentro (§15). **Prueba:** test #10.
* `_build_optimizer` (`train.py:29`): **Adam** con dos grupos de parámetros —
  backbone (todo menos `fc.`) a `lr*backbone_lr_mult` y head a `lr`. Con `mult=1.0` (config)
  el LR es uniforme (contraste de procedencia limpio). **Concepto:** grupos de parámetros.
* `_predict_logits` (`train.py:55`): `model.eval()` + `torch.no_grad()`; devuelve
  `(logits, y, idx)` como arrays. **Efecto:** pone el modelo en `eval` y desactiva
  gradiente. **Prueba:** con un modelo fijo, dos llamadas dan el mismo logit.

---

### Etapa 7 — `eval.py`

#### `aggregate_views_to_knee(view_pred_df)` **[ANCLA]**

**Ubicación:** `eval.py:14`. Llamada por: `run_once`, `run_curve.main`. Etapa:
agregación vista→rodilla.

**Responsabilidad.** Colapsar los logits **por vista** a **una fila por `knee_id`**
usando la **media de logits**, recuperando `patient_id` y `y`, y añadiendo `prob`.

**Razón de existencia.** Materializa la regla de agregación **congelada**. Convierte
predicciones por vista (no independientes dentro de una rodilla) en la unidad de
evaluación correcta. Assertar `y` constante dentro de cada `knee_id` evita mezclar
lados.

**Clasificación.** Evaluación / transformación / matemática.

**Entradas.** `view_pred_df` con `knee_id, patient_id, side_class, y_vista, logit`.
**Salida.** DataFrame por rodilla con `knee_id, patient_id, side_class, y, logit
(media), n_views, prob`.

**Estado/efectos.** Ninguno salvo `assert` de consistencia de etiqueta.

**Concepto matemático.** *Media de logits.* Para una rodilla con vistas de logits
`z_1..z_m`: `z̄ = (1/m) Σ z_j`, y `prob = σ(z̄)`. **Promediar logits ≠ promediar
probabilidades**: `σ(mean(z)) ≠ mean(σ(z))` en general (σ es no lineal); promediar en el
espacio logit es la regla elegida y congelada.

**Correspondencia código–matemáticas.** `logit=("logit","mean")` ↔ `z̄`;
`out["prob"] = 1/(1+exp(−logit))` ↔ `σ(z̄)`; `y=("y_vista","first")` ↔ etiqueta de la
rodilla (constante).

**Librerías/símbolos.** `DataFrame.groupby("knee_id").agg(...)`, `groupby.nunique()`
(para la aserción), `numpy.exp`.

**Guiada / prueba.** **TODO:** el `groupby.agg` + la aserción `y` constante + `prob`.
**Prueba:** test #20. **Solución:** `eval.py:14-32`.

**Errores frecuentes.** Promediar probabilidades en vez de logits; agregar por
`patient_id` (colapsaría los dos lados de un mixto); olvidar la aserción de `y`
constante (mezclarías fx y normal).

**Criterio para continuar.** Una rodilla con logits `[2, −2, 0]` → `z̄=0`, `prob=0.5`,
`n_views=3`.

**Preguntas.** (1) ¿Por qué media de **logits** y no de probabilidades? (2) ¿Por qué
`first` para `y` y no `mean`? (3) ¿Qué protege la aserción `bad.empty`?

#### `knee_metrics(knee_df)` [ficha]
* **Ubicación:** `eval.py:35`. **Responsabilidad:** `n_knees, prevalence, auroc, auprc`.
  **Matemática:** AUROC sobre el **logit** (invariante a σ, monótona), AUPRC sobre
  `prob`; ambos `nan` si una sola clase. **Símbolos:** `roc_auc_score`,
  `average_precision_score`. **Prueba:** test #21. **Detalle clave:** usar el logit para
  AUROC es correcto porque el AUROC solo depende del **orden**.

#### `paired_bootstrap_delta_auroc(knee_a, knee_b, n_boot, ci, seed, return_deltas)` **[ANCLA]**

**Ubicación:** `eval.py:49`. Llamada por: `run_once`, `run_curve.main`. Etapa:
incertidumbre.

**Responsabilidad.** Estimar `ΔAUROC = AUROC(A) − AUROC(B)` y su incertidumbre por
**bootstrap pareado remuestreando pacientes completos** (cada paciente arrastra sus
1–2 rodillas), con el **mismo** resample para A y B.

**Razón de existencia.** Las rodillas de un paciente **no** son independientes;
remuestrear rodillas sueltas subestimaría la varianza. Remuestrear **pacientes** (un
cluster) da un IC honesto. Que el resample sea el **mismo** para A y B (pareado) cancela
la varianza común y aísla la **diferencia**.

**Clasificación.** Estadística (bootstrap cluster, pareado).

**Entradas.** `knee_a`, `knee_b` (mismos `knee_id`, mismas `y`); `n_boot` (config 2000);
`ci` (0.95); `seed`; `return_deltas` (si se conserva el array crudo para agrupar entre
semillas). **Salida.** dict con `delta_auroc_observed`, `ci_low/high`, `p_bootstrap`,
`n_boot_effective`, `n_patients` (+ `_deltas` si se pide).

**Estado/efectos.** Ninguno (usa un RNG local `default_rng(seed)`).

**Concepto matemático/estadístico.**
* **Estadístico observado:** `Δ̂ = AUROC(y, s_A) − AUROC(y, s_B)`.
* **Bootstrap por clusters:** en cada réplica se muestrean `n_pat` pacientes **con
  reemplazo**; se concatenan los índices de sus rodillas; se recalcula `Δ`. Si la
  muestra queda con una sola clase, se descarta.
* **IC percentil** (en `pooled_ci`): cuantiles `α/2` y `1−α/2` de la distribución de
  `Δ*`.
* **p-valor de dos colas** (en `bootstrap_pvalue`): `2·min(P(Δ*<0), P(Δ*>0))`, acotado a 1.

**Correspondencia código–matemáticas.**
`sampled = patients[rng.integers(0, n_pat, size=n_pat)]` ↔ muestreo de clusters con
reemplazo; `idx = concat(knees_by_pat[p] for p in sampled)` ↔ arrastrar rodillas del
cluster; `roc_auc_score(yy, sa[idx]) − roc_auc_score(yy, sb[idx])` ↔ `Δ*`.

**Librerías/símbolos.** `numpy.random.default_rng(seed).integers` (muestreo),
`numpy.where` (índices por paciente), `sklearn.metrics.roc_auc_score`,
`numpy.quantile` (en `pooled_ci`).

**Pseudocódigo.**
```
alinear A y B por knee_id (mismas y)
patients = únicos; knees_by_pat = {p: índices de sus rodillas}
obs = AUROC(y, sA) − AUROC(y, sB)
para _ en n_boot:
    sampled = elegir n_pat pacientes con reemplazo
    idx = concat(knees_by_pat[p] for p in sampled)
    si una sola clase en y[idx]: continуar
    deltas.append(AUROC(y[idx], sA[idx]) − AUROC(y[idx], sB[idx]))
devolver obs, pooled_ci(deltas), bootstrap_pvalue(deltas), ...
```

**Implementación guiada.** **TODO:** el bucle bootstrap. **Pista:** precomputa
`knees_by_pat` fuera del bucle; usa `np.concatenate`. **Prueba:** test #22 (pocos
pacientes sintéticos, comprobar que el IC contiene el observado). **Solución:**
`eval.py:49-104`.

**Errores frecuentes.** Remuestrear rodillas en vez de pacientes (IC demasiado
estrecho); usar resamples distintos para A y B (rompe el pareado); no descartar
réplicas de una sola clase (`roc_auc_score` lanzaría); recomputar `knees_by_pat` dentro
del bucle (lento).

**Criterio para continuar.** Con A=B el `ΔAUROC` observado es 0 y el IC lo rodea;
`p≈1`.

**Preguntas.** (1) ¿Por qué remuestrear pacientes y no rodillas? (2) ¿Por qué el
**mismo** resample para A y B? (3) ¿Qué hace `n_boot_effective` < `n_boot`?
(4) ¿Por qué el p-valor es `2·min(...)`?

**Ejercicio.** Con dos comparadores idénticos, verifica `Δ=0` y `p` cercano a 1.

#### `pooled_ci(deltas, ci)` / `bootstrap_pvalue(deltas)` [ficha]
* **Ubicación:** `eval.py:107` / `:115`. **Matemática:** IC percentil `[q_{α/2},
  q_{1−α/2}]` con `numpy.quantile`; p-valor `2·min(frac_neg, frac_pos)` acotado a 1.
  **Razón de existir aparte:** `run_multiseed` y `run_curve` **agrupan** (pool) los
  `_deltas` de varias semillas y reusan estas dos funciones sobre el pool. **Prueba:**
  test #22.

---

### Etapa 8 — `run_experiment.py`

#### `run_once(cfg, seed, out_dir, logger, device, smoke, collect_boot, index, excl_df)` [ficha ampliada]
* **Ubicación:** `run_experiment.py:49`. **Responsabilidad:** una corrida CV completa
  (todos los comparadores) para **una** semilla; escribe todos los artefactos y devuelve
  `{knee_oof, metrics, deltas, comparators}`. **Núcleo reutilizado** por multiseed.
* **Flujo:** set_seed → save_config/dump_versions → índice (QC si real, o toy, o
  reutilizar `index` pasado) → `find_mixed_patients` (+ assert ≥1) → `make_folds` +
  `check_no_leakage` + `save_manifest` → por comparador × fold: `set_seed(seed+fold)`,
  `build_model`, `train_one_fold`, `aggregate_views_to_knee` (por fold) → OOF por
  comparador → `knee_metrics` → `ΔAUROC` por par con `itertools.combinations`.
* **Detalle clave:** `set_seed(seed + fold["fold"])` **antes** de construir el modelo,
  igual para todos los comparadores → folds, orden de lotes e init del head **matcheados**.
* **`collect_boot`**: si `True`, `paired_bootstrap_delta_auroc(return_deltas=True)`
  conserva `_deltas` para que multiseed los agrupe.
* **Salidas:** ver §13. **Símbolos:** `itertools.combinations`, `json.dump`.

#### `parse_args()` / `_json_safe_deltas(deltas)` / `main()` [ficha]
* `parse_args` (`:34`): `--config`, `--smoke`. `_json_safe_deltas` (`:41`): quita el
  array crudo `_deltas` antes de serializar (no es JSON-friendly y es pesado). `main`
  (`:169`): carga config, arma `out_dir`, logger a `run.log`, resuelve device, importa
  torch (fallo temprano si falta) y llama `run_once`.

---

### Etapa 9 — Lectura real de PlaTiF (`data.py`)

* **`_patient_id_from_path(path)`** (`data.py:130`) [ficha]: regex `Patient_ID_(\d+)`
  del nombre de archivo. **Símbolos:** `re.search`.
* **`list_platif_files(root)`** (`:137`) [ficha]: `glob` de
  `Patient_Data_Part_*/Patient_ID_*.mat`, ordenado (186 esperados). **Símbolos:**
  `glob.glob`, `sorted`.
* **`load_platif_struct(mat_path)`** **[ANCLA breve]** (`:145`): `scipy.io.loadmat(...,
  squeeze_me=True, struct_as_record=False)` devuelve los structs de MATLAB v5 como
  objetos `mat_struct` con atributos = campos. Toma la única clave que no empieza por
  `__` (o la que empieza por `Patient_ID`). **Concepto clave:** con
  `struct_as_record=False`, cada struct expone `_fieldnames` y accedes a subcampos como
  atributos (`v.OriginalImage`, `v.label`, `v.BW`). Alternativa `h5py` sería para `.mat`
  v7.3 — **no** se usa aquí (ver §15). **Prueba:** test #24.
* **`_view_names(struct)`** (`:160`) [ficha]: campos que casan `^im\d+$`, ordenados por
  el entero; **ignora `Coronal_CT`** (fuera de protocolo). **Símbolos:** regex `VIEW_RE`.
* **`read_view_label(struct, view)`** (`:167`) / **`read_view_image(struct, view)`**
  (`:174`) [ficha]: extraen `label` (int) y `OriginalImage` (float32 2D) de una vista.
* **`build_real_index(cfg, seed=0, run_assertions=True)`** (`:181`) [ficha]: recorre los
  `.mat`, arma records `{patient_id, view, label, mat_path}`,
  `build_index_from_view_records`, re-adjunta `mat_path`, y (si `run_assertions`) llama
  `assert_real_index`. **Auditoría:** el parámetro `seed` **no se usa** (§15).
* **`assert_real_index(df)`** **[ANCLA breve]** (`:211`): imprime actual vs esperado y
  **asserta** 421 vistas / 186 pacientes / 190 rodillas / 128 fx / 62 normal y
  `mixed == {92,112,133,147}` **exacto**. **Razón:** congela el ground-truth; si el disco
  cambia, aborta antes de entrenar. **Prueba:** test #25.
* **`_find_col(cols, *needles)`** (`:245`) / **`cross_check_xlsx(df, xlsx_path)`**
  (`:253`) [ficha]: localiza columnas del xlsx por subcadenas y **verifica** (no filtra)
  que `Normal(xlsx) ⇔ todas las vistas label==7`; reporta discrepancias como supuesto
  marcado. **Símbolos:** `pandas.read_excel(engine="openpyxl")`, `merge`, `groupby.max`.
  **Auditoría:** `side_col` se calcula y solo se imprime (§15).
* **`build_index(cfg, seed)`** (`:302`) [ficha]: despacha a `build_toy_index` o
  `build_real_index` según `cfg["mode"]`.

---

### Etapa 10 — `qc.py`

#### `_phash(img, hash_size=8, highfreq_factor=4)` **[ANCLA]**

**Ubicación:** `qc.py:45`. Llamada por: `run_qc`. Etapa: control de calidad.

**Responsabilidad.** Calcular un **hash perceptual** de 64 bits de una imagen vía DCT.

**Razón de existencia.** Detectar **near-duplicates** (misma radiografía repetida) para
excluir duplicados **entre pacientes** (riesgo de fuga). Un hash perceptual es robusto a
pequeños cambios de intensidad/escala, a diferencia de un hash criptográfico.

**Clasificación.** Control de calidad / matemática (transformada).

**Entradas.** `img:np.ndarray` 2D; `hash_size=8` (→ 64 bits), `highfreq_factor=4`.
**Salida.** `int` (64 bits). **Efectos.** Ninguno.

**Concepto matemático (denso).**
1. Normaliza a `[0,1]` por percentiles 1–99.
2. Redimensiona a `n×n` con `n = hash_size*highfreq_factor = 32` por muestreo en malla.
3. Aplica **DCT-II 2D** (`dct` por filas y por columnas, `norm="ortho"`).
4. Toma el bloque de **baja frecuencia** `hash_size×hash_size` (esquina superior
   izquierda) — ahí se concentra la estructura.
5. Umbraliza por la **mediana** del bloque **excluyendo el término DC** `[0,0]` (el DC
   es el promedio global; incluirlo sesgaría el umbral).
6. Cada bit = `1` si el coeficiente `>` mediana; se empaquetan con desplazamientos.

**Correspondencia código–matemáticas.**
`d = dct(dct(small, axis=0, norm="ortho"), axis=1, norm="ortho")` ↔ DCT-II 2D
separable; `dlow = d[:8,:8]` ↔ baja frecuencia; `med = median(dlow.flatten()[1:])` ↔
mediana sin el DC; `bits = (dlow > med)` ↔ umbralización; `h = (h<<1)|bit` ↔ empaquetado.

**Librerías/símbolos.** `scipy.fftpack.dct` (DCT sin implementar la FFT a mano),
`numpy.percentile/clip/median`, `numpy.linspace + np.ix_` (muestreo en malla). La DCT
concentra energía en pocas frecuencias bajas (base del hash perceptual y de JPEG).

**Guiada / prueba.** **TODO:** el pipeline DCT→umbral→bits. **Pista:** DCT separable =
DCT por un eje y luego por el otro. **Prueba:** `_hamming(_phash(im), _phash(im)) == 0`
(idéntica) y una imagen muy distinta da hamming grande (test #26). **Solución:**
`qc.py:45-65`.

**Errores frecuentes.** Incluir el DC en la mediana (umbral sesgado); tomar alta
frecuencia en vez de baja; olvidar `norm="ortho"` (escala inconsistente).

**Criterio para continuar.** Imagen consigo misma → hamming 0; copia ligeramente
alterada → hamming pequeño; imagen distinta → hamming grande.

**Preguntas.** (1) ¿Por qué DCT y baja frecuencia? (2) ¿Por qué excluir el DC de la
mediana? (3) ¿Por qué un hash **perceptual** y no `md5`?

#### `_hamming(a, b)` [ficha]
* **Ubicación:** `qc.py:68`. **Matemática:** distancia de Hamming = nº de bits distintos
  = `popcount(a XOR b)`. **Código:** `bin(a ^ b).count("1")`. **Razón:** near-duplicate si
  `hamming ≤ dup_hamming_threshold`. **Prueba:** `_hamming(0b1010, 0b1000)==1`.

#### `run_qc(cfg, index_df)` [ficha ampliada]
* **Ubicación:** `qc.py:83`. **Responsabilidad:** aplicar los criterios **congelados** y
  devolver `(excl_df, flow)`. **Criterios (en orden, con `continue` al excluir):**
  (1) **resolución** `min(H,W) < min_resolution`; (2) **concordancia de máscara** `BW`
  ausente / forma distinta / trivial (`frac_fg ≤ 0` o `≥ 1`); (3) **duplicado** por pHash
  con alcance `cross_patient` (un match **intra**-paciente NO se excluye — es dato
  legítimo, no hay fuga porque los folds son por paciente); (4) **manuales** desde
  `manual_exclusions_csv` si se provee. **Razón científica:** criterios congelados
  ANTES de mirar métricas; no se excluyen casos difíciles porque el modelo falle.
* **Símbolos:** `DataFrame.groupby("mat_path")` (reusar el struct cargado), `_phash`,
  `_hamming`. **Prueba:** test #26.

#### `_read_view_arrays / _flow_diagram / apply_exclusions / build_qc_cohort` [ficha]
* `_read_view_arrays` (`:75`): devuelve `(img, bw)` de una vista (`BW` = máscara).
* `_flow_diagram` (`:160`): cuenta `full` vs `after_qc` (pacientes/rodillas/vistas/
  knee_pos/knee_neg), vistas excluidas por criterio, rodillas y pacientes caídos.
  **Regla:** una vista excluida no tira su rodilla si quedan otras vistas; la rodilla
  cae si TODAS sus vistas caen. **Símbolos:** `DataFrame.apply` sobre pares
  `(patient_id, view)`.
* `apply_exclusions` (`:186`): devuelve el índice sin las vistas excluidas.
* `build_qc_cohort(cfg, verbose)` (`:196`): `build_real_index` → `run_qc` →
  `apply_exclusions`; devuelve `(cohort, excl_df)`. Es la entrada de datos reales de
  `run_once`, `run_multiseed`, `run_curve`, `splits._main`.

---

### Etapas 12–14 — orquestadores y análisis

#### `run_multiseed.main()` / `_mean_std(xs)` [ficha]
* **Ubicación:** `run_multiseed.py:34` / `:29`. **Responsabilidad:** por cada semilla
  llama `run_once(collect_boot=True)`; acumula AUROC/AUPRC por comparador y los `_deltas`
  por par. **Agregación:** por comparador → `_mean_std` (media, `std(ddof=1)`) =
  variabilidad de **optimización**; por par → media±desv del efecto entre semillas +
  IC95% **agrupado** (`pooled_ci` sobre `np.concatenate` de los `_deltas`) + p agrupado.
  **Detalle:** en modo real construye la cohorte QC **una vez** y la reutiliza en todas
  las semillas (evita recomputar QC). **Salida:** `multiseed_summary.json`. **Matemática:**
  desviación muestral con `ddof=1`.

#### `run_curve.main()` / `_mean_std` [ficha ampliada]
* **Ubicación:** `run_curve.py:46`. **Responsabilidad:** curva de eficiencia. Por
  semilla: `make_folds`+`check_no_leakage`; `nested_stratified_subsets` por fold (mismos
  para todos los comparadores y fracciones, sembrados con `seed+fold`); por
  `fracción × comparador × fold`: `set_seed(seed+fold)`, `build_model`, `train_one_fold`
  con `train_patients = subconjunto`, val/test intactos; agrega, `knee_metrics`.
  **Agregación:** `curve_table[fraction][comp]` (media±desv AUROC entre semillas);
  `interaction` (ΔAUROC por par y fracción, IC agrupado); `data_equivalence` (fracción
  mínima que iguala `random@100%`). **Auditoría:** `data_equivalence` se calcula y
  escribe pero la guía del paper lo marca engañoso (§15). **Salida:** `curve_summary.json`.

#### `failure_analysis.main()` / `youden_threshold(y, prob)` [ficha]
* **Ubicación:** `failure_analysis.py:31` / `:24`. **Responsabilidad:** sobre un
  `oof_knee_preds.csv`, elige umbral **Youden** (`argmax(tpr−fpr)` con `roc_curve`),
  construye matriz de confusión a nivel rodilla, sens/spec, nº medio de vistas en
  aciertos vs fallos, y desglosa **falsos negativos** por tipo Schatzker (join con xlsx).
  **Símbolos:** `sklearn.metrics.roc_curve`, `_find_col` (reusado de `data.py`).
  **Auditoría:** importa una función **privada** de otro módulo (`_find_col`) — acoplamiento
  (§15).

#### `make_figures.py`: `figure_efficiency_curve / figure_forest_delta100 / table_provenance_metrics / _save / main` [ficha]
* **Ubicación:** `make_figures.py`. **Responsabilidad:** leer `curve_summary.json` y
  `multiseed_summary.json` y producir la **curva de eficiencia** (`errorbar` AUROC vs
  fracción, escala log en x, línea de azar 0.5), el **forest** de ΔAUROC@100% (IC
  agrupado; color según si el IC excluye 0) y la **tabla** de métricas por comparador.
  **Símbolos:** `matplotlib.pyplot.subplots/errorbar/axhline/axvline`, `pandas.DataFrame.
  to_csv`. **Sin matemática nueva** (solo dibuja lo ya calculado). **Prueba:** test #28.

---

## 8. Imports y librerías por archivo

Clasificación: **[std]** biblioteca estándar · **[ext]** dependencia externa ·
**[local]** módulo del proyecto.

### `utils.py`
| Símbolo | Tipo | Dónde | Para qué | Por qué | Alternativa | ¿Dominar ya? |
|---|---|---|---|---|---|---|
| `json` | std | dump_versions | serializar versiones | JSON estándar | — | sí |
| `logging` | std | get_logger | logs | infra estándar | loguru | sí |
| `os` | std | rutas/env | FS y env | portable | pathlib | sí |
| `platform` | std | dump_versions | SO | reproducibilidad | — | no |
| `random` | std | set_seed | semilla Python | RNG base | — | sí |
| `subprocess` | std | _git_hash | commit git | trazabilidad | GitPython | no |
| `sys` | std | logger/versions | stdout/version | infra | — | no |
| `datetime` | std | dump_versions | timestamp | auditoría | — | no |
| `numpy` | ext | set_seed | RNG numpy | base numérica | — | sí |
| `yaml` | ext | load/save_config | leer/escribir YAML | config | json/toml | sí |
| `torch` | ext (perezoso) | set_seed/resolve_device | semillas/device | DL | — | sí |

### `data.py`
| Símbolo | Tipo | Dónde | Para qué | Por qué |
|---|---|---|---|---|
| `glob` [std] | list_platif_files | listar `.mat` | patrón de archivos |
| `os`,`re` [std] | rutas / regex ids-vistas | parseo | — |
| `numpy` [ext] | percentiles/clip/RNG toy | preproc y toy | base numérica |
| `pandas` [ext] | índice DataFrame | tabla una-fila-por-vista | agrupaciones |
| `scipy.io.loadmat` [ext] | load_platif_struct | leer `.mat` v5 | structs MATLAB |
| `torch`,`torch.nn.functional` [ext] | ViewDataset | tensores/interpolate | DL |
| `torchvision.transforms` [ext] | augment (si enabled) | augmentation | **solo si `augment.enabled`** |
| `utils.load_config` [local] | CLI | config | — |

### `qc.py`
`argparse,json,os` [std]; `numpy,pandas` [ext]; `scipy.fftpack.dct` [ext] (pHash);
`data.{build_real_index,load_platif_struct,read_view_image}` [local];
`utils.load_config` [local].

### `splits.py`
`json,os` [std]; `numpy,pandas` [ext]; `sklearn.model_selection.StratifiedGroupKFold`
[ext]; en CLI `qc.build_qc_cohort`, `utils.{load_config,set_seed}` [local]; `math`
(dentro de `nested_stratified_subsets`).

### `model.py`
`collections.OrderedDict` [std]; `timm,torch,torch.nn` [ext]; en CLI
`utils.{get_logger,load_config}` [local].

### `train.py`
`copy` [std]; `numpy,pandas,torch` [ext]; `sklearn.metrics.roc_auc_score` [ext];
`torch.utils.data.DataLoader` [ext]; `data.{ViewDataset,subset_index}` [local].

### `eval.py`
`numpy,pandas` [ext]; `sklearn.metrics.{average_precision_score,roc_auc_score}` [ext].

### `run_experiment.py`
`argparse,itertools,json,os` [std]; `pandas` [ext]; `data,eval,model,splits,train,utils`
[local]; `qc` [local, si real]; `torch` [ext, import de guardia].

### `run_multiseed.py`
`argparse,itertools,json,os,collections.defaultdict` [std]; `numpy` [ext];
`eval.{bootstrap_pvalue,pooled_ci}`, `run_experiment.run_once`, `utils.*` [local].
*(Nota: `itertools` se importa pero el pareado lo hace `run_once`; ver §15.)*

### `run_curve.py`
`argparse,itertools,json,os,collections.defaultdict` [std]; `numpy,pandas` [ext];
`eval.*, model.build_model, splits.*, train.train_one_fold, utils.*, qc.build_qc_cohort`
[local].

### `failure_analysis.py`
`argparse,os` [std]; `numpy,pandas` [ext]; `yaml` [ext]; `sklearn.metrics.roc_curve`
[ext]; `data._find_col` [local, **privada**].

### `make_figures.py`
`argparse,json,os` [std]; `matplotlib` [ext]; `pandas` [ext]; `yaml` [ext, opcional].

### Dependencias declaradas en `requirements.txt`
`torch, torchvision, timm, scikit-learn, pandas, numpy, scipy, openpyxl, h5py, pyyaml,
matplotlib, pytest`. **Verificación de uso:**
* `h5py` — **no se usa** para cargar datos (todo es `scipy.io.loadmat`); solo se sondea
  su versión en `dump_versions`. Ver §15.
* `torchvision` — solo se usa si `augment.enabled=true` (config actual: `false`).
* Todos los demás se usan directamente.

---

## 9. Matemáticas y estadística que aparecen

**Datos/preproc.** Función indicadora `y = 𝟙[label ≠ 7]`; percentiles P1/P99;
normalización `x̂=clip((x−lo)/(hi−lo),0,1)`; resize bilineal; replicación 1→3 canales;
estandarización `z=(x−μ)/σ` (ImageNet); formas `C×H×W` (vista) y `B×C×H×W` (lote).

**QC.** DCT-II 2D separable; término DC `[0,0]`; umbral por mediana (sin DC); pHash de
64 bits; XOR + popcount = Hamming; duplicado **visual** (contenido casi igual) ≠ **fuga**
(el mismo contenido en **otro** paciente cruzando folds).

**Particiones.** k-fold estratificado y agrupado; folds externos e internos; prevalencia
(media de `y_vista` sobre rodillas); disjunción + cobertura; predicciones OOF (cada
rodilla se predice desde el fold donde fue test); subconjuntos anidados estratificados.

**Modelo.** Convolución; ResNet-50 (conexiones residuales `y=f(x)+x`); backbone +
clasificador lineal de **un logit**; transferencia = copiar pesos del extractor;
`state_dict` = nombre→tensor (parámetros **y** buffers como running stats de BatchNorm);
casado nombre+forma.

**Entrenamiento.** `σ(z)=1/(1+e^{−z})`; BCE `ℓ=−[y logσ(z)+(1−y)log(1−σ(z))]`;
`BCEWithLogitsLoss` combina σ+BCE de forma numéricamente estable (log-sum-exp);
`pos_weight=n_neg/n_pos`; gradiente + regla de la cadena + retropropagación; Adam (lr,
weight decay, grupos de parámetros); `train`/`eval`; `torch.no_grad`; early stopping;
restauración del mejor estado.

**Umbral.** Matriz de confusión (TP/FP/FN/TN); sensibilidad `TP/(TP+FN)`; especificidad
`TN/(TN+FP)`; F1 `2TP/(2TP+FP+FN)`; Youden `J=sens+spec−1`; un umbral es una métrica
**dependiente del punto de operación**, AUROC no.

**Evaluación.** Media de logits `z̄`; `σ(mean z) ≠ mean σ(z)`; AUROC (área ROC),
**invariante a transformaciones estrictamente monótonas** (por eso se calcula sobre el
logit); AUPRC; prevalencia; `ΔAUROC` entre comparadores.

**Incertidumbre.** Bootstrap; bootstrap **pareado** (mismo resample A/B); remuestreo por
**clusters de pacientes**; IC percentil `[q_{α/2}, q_{1−α/2}]`; distribución bootstrap;
p-valor de dos colas `2·min(P(Δ*<0),P(Δ*>0))`; variabilidad de **optimización** (entre
semillas, `std ddof=1`) vs de **muestreo** (bootstrap); pooling de distribuciones.

---

### 9.1 Verificación de cada concepto: dónde estudiarlo y cómo checarlo

Para cada concepto matemático real del baseline: **fórmula**, **dónde estudiarlo**
(referencia canónica que puedes consultar) y **cómo checarlo** (una comprobación
numérica que confirma que el código implementa la fórmula). Las referencias son
fuentes reales; las comprobaciones son ejecutables tal cual.

| Concepto | Fórmula | Dónde estudiarlo (referencia) | Cómo checarlo (numéricamente) |
|---|---|---|---|
| Indicadora `y=𝟙[label≠7]` | `y=1` si `label≠7` | cualquier texto de lógica/probabilidad básica | `y_from_label(7)==0 and y_from_label(3)==1` |
| Normalización P1–P99 | `x̂=clip((x−P1)/(P99−P1),0,1)` | `numpy.percentile` (docs); Efron & Tibshirani cap. cuantiles | ver bloque **C1** |
| Estandarización ImageNet | `z=(x−μ)/σ` | docs de `torchvision` (transforms.Normalize) | media/σ por canal ≈ 0/1 tras estandarizar |
| DCT-II 2D | `X_k=Σ_n x_n cos[π/N (n+½)k]` | Ahmed, Natarajan & Rao (1974), *IEEE T. Computers*; docs `scipy.fftpack.dct` (type-2, norm='ortho') | `dct` de una señal constante concentra energía en el DC `[0]` |
| pHash + Hamming | `bit=𝟙[coef>mediana]`; `H=popcount(a⊕b)` | Zauner (2010), *Impl. & Benchmarking of Perceptual Hash*; pHash.org | ver bloque **C2** |
| k-fold estratificado-agrupado | partición por grupo, estratos ≈ prevalencia | docs scikit-learn `StratifiedGroupKFold`; *Elements of Statistical Learning* §7.10 (CV) | `check_no_leakage` verde + prevalencias por fold similares |
| Sigmoide | `σ(z)=1/(1+e^{−z})` | cualquier texto de regresión logística | `σ(0)=0.5`; monótona creciente |
| BCE con logits | `ℓ=−[y logσ(z)+(1−y)log(1−σ(z))]` (estable por log-sum-exp) | docs PyTorch `BCEWithLogitsLoss`; *Deep Learning* (Goodfellow) §6.2 | ver bloque **C3** |
| `pos_weight` | `w=n_neg/n_pos` | docs `BCEWithLogitsLoss` (arg `pos_weight`) | `n_pos=2,n_neg=6 ⇒ 3.0` |
| Adam | momentos 1º/2º con corrección de sesgo | Kingma & Ba (2015), arXiv:1412.6980 | tras `opt.step()` un parámetro cambia (test #18) |
| Youden J | `J=sens+spec−1=TPR−FPR` | Youden (1950), *Cancer* | ver bloque **C4** |
| AUROC | `P(score⁺>score⁻)` = U de Mann–Whitney normalizada | Fawcett (2006), *Pattern Recognition Letters*; Hanley & McNeil (1982), *Radiology*; docs sklearn `roc_auc_score` | ver bloque **C5** |
| AUROC invariante a σ | monótona ⇒ mismo orden ⇒ mismo AUROC | Fawcett (2006) | ver bloque **C5** |
| AUPRC (AP) | `AP=Σ_n (R_n−R_{n−1})·P_n` | Davis & Goadrich (2006), *ICML*; docs sklearn `average_precision_score` | `average_precision_score` de un caso separable = 1.0 |
| Media de logits ≠ media de probs | `σ(mean z)≠mean σ(z)` | desigualdad de Jensen (σ no lineal) | ver bloque **C6** |
| Bootstrap (percentil) | `IC=[q_{α/2},q_{1−α/2}]` de `Δ*` | Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, cap. 13 | `pooled_ci` == `np.quantile(deltas,[α/2,1−α/2])` |
| Bootstrap por clusters | remuestrear **grupos** (pacientes) con reemplazo | Davison & Hinkley (1997) §3.8; Field & Welsh (2007), *JRSS-B* | con A=B, `Δ*≈0` y el IC rodea 0 (test #22) |
| p-valor bootstrap 2 colas | `p=2·min(P(Δ*<0),P(Δ*>0))` | Davison & Hinkley (1997) | A=B ⇒ `p≈1`; efecto grande ⇒ `p` pequeño |
| Desv. muestral (multi-semilla) | `s=√(Σ(x−x̄)²/(n−1))` | docs `numpy.std(ddof=1)` | `_mean_std([1,2,3])==(2.0, 1.0)` |

**Bloques de comprobación ejecutables** (pégalos en un intérprete con el entorno
`ropec` activo; validan que el código = la fórmula):

```python
# C1 — normalización P1–P99 (data.ViewDataset._base_image)
import numpy as np
img = np.array([[0.,10.,1000.],[2.,3.,4.]])          # 1000 es outlier
lo, hi = np.percentile(img,1), np.percentile(img,99)
xhat = np.clip((img-lo)/(hi-lo),0,1)
assert xhat.min()>=0 and xhat.max()<=1                # rango correcto
```

```python
# C2 — pHash estable + Hamming (qc._phash, qc._hamming)
from qc import _phash, _hamming
im = np.random.default_rng(0).random((256,256))
assert _hamming(_phash(im), _phash(im)) == 0          # misma imagen -> 0 bits
im2 = im.copy(); im2[0,0]+=0.01
assert _hamming(_phash(im), _phash(im2)) <= 5          # cambio mínimo -> hamming pequeño
```

```python
# C3 — BCEWithLogitsLoss == fórmula manual con pos_weight (train._pos_weight + criterion)
import torch, torch.nn as nn
z  = torch.tensor([0.5,-1.2]); y = torch.tensor([1.,0.]); pw = torch.tensor([3.0])
s  = torch.sigmoid(z)
man = -(pw*y*torch.log(s) + (1-y)*torch.log(1-s)).mean()
lib = nn.BCEWithLogitsLoss(pos_weight=pw)(z,y)
assert torch.allclose(man, lib, atol=1e-6)             # coinciden
```

```python
# C4 — Youden = argmax(TPR-FPR) (failure_analysis.youden_threshold / train._choose_threshold)
from sklearn.metrics import roc_curve
y = np.array([0,0,1,1]); p = np.array([0.1,0.4,0.35,0.8])
fpr,tpr,thr = roc_curve(y,p); j = tpr-fpr
assert thr[int(np.argmax(j))] > 0                      # el umbral maximiza sens+spec-1
```

```python
# C5 — AUROC = U de Mann-Whitney, e invariante a la sigmoide (eval.knee_metrics)
from sklearn.metrics import roc_auc_score
y = np.array([0,0,1,1]); s = np.array([0.1,0.4,0.35,0.8])
pos, neg = s[y==1], s[y==0]
u = np.mean([(a>b)+0.5*(a==b) for a in pos for b in neg])   # def. de Mann-Whitney
assert abs(u - roc_auc_score(y,s)) < 1e-12                    # AUROC == U normalizada
sig = 1/(1+np.exp(-s))
assert roc_auc_score(y,s) == roc_auc_score(y,sig)            # invariante a σ (monótona)
```

```python
# C6 — promediar logits != promediar probabilidades (eval.aggregate_views_to_knee)
# (usa un vector ASIMÉTRICO: con [2,-2,0] ambos dan 0.5 por simetría y no se ve la diferencia)
z = np.array([2.,1.,0.])
prob_de_media_logits = 1/(1+np.exp(-z.mean()))               # σ(1)   ≈ 0.731 (regla del baseline)
media_de_probs       = (1/(1+np.exp(-z))).mean()             # ≈ 0.704
assert abs(prob_de_media_logits - media_de_probs) > 1e-3     # difieren (Jensen)
```

**Dónde "checar" dentro del propio repo (sin escribir código nuevo):** las pruebas de
`tests/test_splits.py` verifican la cero fuga; `python model.py --config config.yaml
--weights all` confirma el conteo de tensores; los `metrics.json`/`delta_auroc.json`
de una corrida contienen los valores que estas fórmulas producen. Para AUROC/AUPRC y el
bootstrap, corre los bloques **C5/C6** y compáralos con el `oof_knee_preds.csv` real.

---

## 10. Pruebas incrementales obligatorias

> Todas corren **sin PlaTiF** salvo #24–#26 (necesitan `.mat`) y son rápidas. Para cada
> una: qué valida · por qué importa · código · esperado · causa de fallo · qué NO
> ejecutar si falla.

1. **`knee_id_of`** — `assert knee_id_of(5,3)=="5:fx"; assert knee_id_of(5,7)=="5:normal"`.
   *Si falla:* no sigas a splits (todo depende de la unidad rodilla).
2. **`y_from_label`** — `assert y_from_label(7)==0 and y_from_label(6)==1`.
3. **Índice sintético** — construir records y `build_index_from_view_records`; comprobar
   columnas `{patient_id,view,knee_id,side_class,label,y_vista}` y orden.
4. **Paciente mixto** — un paciente con labels `[3,7]` ⇒ `find_mixed_patients` lo incluye.
5. **Unicidad de `knee_id`** — `groupby("knee_id")["patient_id"].nunique()` todo 1
   (`test_no_orphan_knee`).
6. **Cobertura de folds** — unión de `test_patients` == todos (`test_no_patient_in_two_test_folds`).
7. **Disjunción train/val/test** (`test_train_val_test_disjoint`).
8. **Mixto en un solo fold** (`test_mixed_patient_knees_same_fold`).
9. **Imagen toy** — `ViewDataset(...,'toy')[0][0].shape==(3,S,S)`; rango finito.
10. **Batch del loader** — `next(iter(DataLoader(ds,batch_size=4)))[0].shape==(4,3,S,S)`.
11. **ResNet-50 un logit** — `build_model(cfg,'random')(x).shape[-1]==1`.
12. **Carga random** — `build_model(cfg,'random')` no lanza; params ~25.6M.
13. **Carga ImageNet** — `loaded >= expect_min` (loggeado).
14. **Fallo RadImageNet** — con un `source_sd` sin remapear, `expect_exact=318` ⇒
    `AssertionError`. *Si NO falla:* revisa el remap antes de entrenar (podrías entrenar
    random creyendo RadImageNet).
15. **`pos_weight`** — índice con `n_pos=2,n_neg=6` ⇒ tensor `[3.0]`.
16. **`BCEWithLogitsLoss`** — comparar `criterion(logit,y)` con la fórmula manual.
17. **`backward` genera gradientes** — tras `loss.backward()`, `p.grad is not None`.
18. **`step` cambia parámetros** — copiar un `p`, `opt.step()`, `assert cambió`.
19. **F1/Youden manual** — con un vector pequeño, `_choose_threshold` coincide con el
    cálculo a mano.
20. **Agregación por rodilla** — logits `[2,-2,0]` ⇒ `logit_mean=0, prob=0.5, n_views=3`.
21. **AUROC/AUPRC** — ejemplo separable ⇒ AUROC=1.0.
22. **Bootstrap pareado** — A=B ⇒ `Δ=0`, IC rodea 0, `p≈1`.
23. **Smoke completo** — `python run_experiment.py --config config.yaml --smoke` (toy)
    genera todos los artefactos del fold 0.
24. **Lectura `.mat`** — `load_platif_struct(path)` expone `_fieldnames` con `imN`.
25. **`assert_real_index`** — sobre PlaTiF real imprime 421/186/190 y no lanza.
26. **QC** — `run_qc` sobre real: resolución/máscara/duplicados; comprueba `qc_flow`.
27. **Manifiesto de folds** — `save_manifest` produce JSON con `folds` y tamaños.
28. **Artefactos finales** — `make_figures.py` genera PNG/PDF y `table_provenance_metrics.csv`.

---

## 11. Sesiones de estudio

Cada sesión: **objetivo · archivos · funciones · conceptos previos · docs · implementar ·
pruebas · resultado observable · criterio de aceptación · ejercicios · autoevaluación.**
Usa 3 niveles: **N1** idea mínima · **N2** equivalente al baseline · **N3** extensiones
(separadas del baseline).

* **S0 — Entender el experimento.** `README`, `METHODS_as_run`, `config.yaml`,
  `requirements`. Resultado: puedes enunciar pregunta, comparadores, unidad, métrica,
  invariantes. Aceptación: llenas §1 de memoria.
* **S1 — Infra (`utils.py`).** Funciones: todas. Docs: `logging`, `yaml`, `torch`
  determinismo. Prueba: seed reproducible. N3: añadir hash del `config` al `versions.json`.
* **S2 — Datos puros (`data.py`).** `knee_id_of`…`subset_index` + `ViewDataset` toy.
  Prueba: #1–#4, #9. Aceptación: índice toy con ≥1 mixto.
* **S3 — Splits + cero fuga.** `splits.py` + `tests/test_splits.py`. **No avanzar sin
  verde.** Aceptación: #5–#8.
* **S4 — Dataset real.** `_base_image/_finalize/_real_image`. Docs: `F.interpolate`.
  Prueba: rango `[0,1]`, forma `3×S×S`.
* **S5 — Modelo.** `model.py`. Docs: `timm`, `state_dict`. Prueba: #11–#14.
* **S6 — Entrenamiento.** primitivas + `train_one_fold`. Antes de todo: **overfit de un
  lote** (loss→0). Prueba: #15–#19. Aceptación: AUROC de val > 0.5 en toy.
* **S7 — Evaluación.** `eval.py` con ejemplos a mano. Prueba: #20–#22.
* **S8 — Smoke end-to-end.** `run_experiment.py --smoke` (toy). Prueba: #23.
* **S9 — PlaTiF real.** Etapa 9 + `cross_check_xlsx`. Prueba: #24–#25.
* **S10 — QC.** `qc.py`. Prueba: #26.
* **S11 — Real completo.** `run_experiment.py` (real).
* **S12 — Multi-semilla.** `run_multiseed.py`.
* **S13 — Curva.** `run_curve.py`.
* **S14 — Análisis y figuras.** `failure_analysis.py`, `make_figures.py`. Prueba: #28.

---

## 12. Comandos de ejecución

```bash
# Entorno (fresco)
conda create -n ropec python=3.11 -y && conda activate ropec
cd ropec_baseline && pip install -r requirements.txt

# Cero fuga (sin datos reales)
pytest tests/                 # o: python tests/test_splits.py

# Smoke toy (plumbing)
python run_experiment.py --config config.yaml --smoke

# CV completa (según mode: en config.yaml)
python run_experiment.py --config config.yaml

# Multi-semilla y curva
python run_multiseed.py --config config.yaml
python run_curve.py --config config.yaml

# Piezas aisladas (CLIs)
python data.py --config config.yaml                 # platif_index.csv (real)
python qc.py --config config.yaml                    # qc_exclusions.csv, qc_flow.json
python splits.py                                     # folds_manifest.json (post-QC)
python model.py --config config.yaml --weights all   # prueba de carga de pesos

# Análisis y figuras
python failure_analysis.py --config config.yaml --oof outputs/multiseed_100/seed_1337/radimagenet/oof_knee_preds.csv
python make_figures.py --curve outputs/curve/curve_summary.json \
                       --multiseed outputs/multiseed_100/multiseed_summary.json
```

---

## 13. Artefactos esperados

Bajo `outputs/<run_name>/` (de `run_once`): `index.csv`, `folds_manifest.json`,
`config_used.yaml`, `versions.json`, `run.log`, `qc_exclusions.csv` (real), y por
comparador `<comp>/{fold_k_preds.csv, oof_knee_preds.csv, metrics.json}`, más
`delta_auroc.json` y `summary.json`.
Multi-semilla añade `seed_<s>/…` y `multiseed_summary.json`, `multiseed.log`.
Curva: `curve/seed_<s>/frac_<f>/<comp>_oof.csv`, `curve_summary.json`, `curve.log`.
Figuras: `outputs/figures/fig_efficiency_curve.(png|pdf)`,
`fig_forest_delta100.(png|pdf)`, `table_provenance_metrics.csv`.
**No hay checkpoints `.pt`** (el mejor estado vive en RAM por fold).

---

## 14. Errores frecuentes

* **Dimensiones:** pasar 2D a `F.interpolate` (necesita `N×C×H×W`); `squeeze(-1)` mal
  puesto (el head da `(B,1)`).
* **Índices:** mapear predicciones sin el `idx` que devuelve `__getitem__`.
* **Tipos:** comparar `label` como string por no castear a int.
* **Dispositivos:** olvidar `y.to(device)` en el loop.
* **Agrupación:** agregar por `patient_id` en vez de `knee_id` (colapsa lados);
  promediar probabilidades en vez de logits.
* **Fuga:** splittear por vista; usar el test para early stopping/umbral; duplicado
  intra-paciente tratado como fuga.
* **Clases:** AUROC con una sola clase presente (→ `nan`); no fijar mínimo 1 por clase
  en la curva.
* **Carga de pesos:** `strict=False` sin contar tensores (el gran landmine); casar por
  nombre sin comprobar forma; olvidar el remap RadImageNet (aborta a 318).
* **Serialización:** intentar `json.dump` del array `_deltas` (por eso `_json_safe_deltas`).
* **Reproducibilidad:** no llamar `set_seed(seed+fold)` antes de construir el modelo.

---

## 15. Auditoría para comprender, no para cambiar silenciosamente

> Para cada hallazgo: **dónde · naturaleza · ¿afecta resultados? · cómo se corregiría ·
> por qué NO cambiarlo antes de reproducir el baseline.**

1. **`_patient_labels` (splits.py:17) — función muerta.** Definida y **nunca llamada**
   (la estratificación de `make_folds` usa `y_vista` **por vista**, no la etiqueta de
   paciente). *Naturaleza:* deuda técnica. *¿Afecta?* No. *Corrección:* borrarla o usarla
   si se quisiera estratificar por paciente. *No cambiar aún:* cambiar la estratificación
   alteraría los folds y por tanto los números; primero reproduce.
2. **`XLSX_BILATERAL_IDS` (data.py:41) — constante muerta.** Documenta bilaterales del
   xlsx pero no se usa. *Deuda/documentación.* No afecta. Dejar como nota.
3. **`full_index` en `_make_loader` (train.py:18) — parámetro no usado.** El loader solo
   usa `index_df`. *Deuda.* No afecta. Corrección: quitar el parámetro (y su call site).
4. **`order = np.argsort(-probs)` (train.py:73) — variable calculada y no usada** en
   `_choose_threshold`. *Deuda.* No afecta.
5. **`side_col` (data.py:262) — calculada y solo impresa** en `cross_check_xlsx`. *Menor.*
   No afecta (es diagnóstico).
6. **`build_real_index(seed=…)` — parámetro no usado.** El índice real es determinista
   desde disco. *Deuda cosmética.* No afecta.
7. **`h5py` en `requirements.txt` — dependencia aparentemente no usada.** Solo se sondea
   su versión en `dump_versions`; la carga real usa `scipy.io.loadmat` (v5). *Naturaleza:*
   dependencia sobrante (o reservada para `.mat` v7.3). No afecta. No quitar aún: podría
   ser necesaria si algún `.mat` fuese v7.3.
8. **`torchvision` — usada solo si `augment.enabled=true`** (config actual `false`, y
   `METHODS_as_run` documenta que el augment degradó los CIs). *Decisión válida*, no
   error: el código queda listo por si se reactiva.
9. **`data_equivalence` en `run_curve.py` (líneas 156–176) vs guía del paper.** El código
   **calcula y escribe** `data_equivalence` (fracción que iguala `random@100%`), pero el
   documento de encuadre del paper indica **no reportarlo** por engañoso (random@100%
   está en el azar, 0.526). *Naturaleza:* discrepancia código↔documentación (no bug).
   *¿Afecta?* No a los números; sí a la **interpretación** si se cita. *No cambiar aún:*
   el JSON es un artefacto; basta con **no** usar ese bloque en el texto.
10. **`_find_col` importado desde `data.py` en `failure_analysis.py` — función privada
    importada entre módulos.** *Acoplamiento / deuda.* No afecta. Corrección: promoverla a
    utilidad pública si se reusa.
11. **`itertools` importado en `run_multiseed.py`** pero el pareado de comparadores lo
    hace `run_once`; el import queda sin uso evidente en el cuerpo de `main`. *Menor.*
    Verifícalo antes de quitarlo.
12. **Aserciones que protegen la validez científica (NO tocar):**
    `check_no_leakage`; `assert_real_index` (ground-truth congelado);
    `load_matched_weights` (conteo de tensores, RadImageNet=318);
    `aggregate_views_to_knee` (`y` constante por rodilla);
    `run_once` (`len(mixed) >= 1`); `paired_bootstrap_delta_auroc` (mismos `knee_id`/`y`).
13. **Excepciones que podrían ocultar problemas:** el `try/except Exception` de
    `cross_check_xlsx` y de `failure_analysis` degrada el cruce con xlsx a aviso — bien
    para no bloquear, pero **enmascara** un xlsx mal formado; revisa el stdout.
14. **Caché global `_REAL_IMAGE_CACHE` + `num_workers=0`.** Decisión válida **para este
    baseline** (dataset pequeño, preproc determinista sin augment → cacheable en un solo
    proceso). **No generalizable:** con `num_workers>0` o augment activo la caché no se
    compartiría/serviría igual. No cambiar sin re-medir.
15. **`n_boot`.** `config.yaml`, `METHODS_as_run.md` y el default de `eval` = **2000**
    (consistentes). El **borrador del paper** menciona 10 000 en un punto → discrepancia
    **documentación↔documentación** a verificar antes de redactar; el **código** usa 2000.

---

## 16. Glosario

* **Vista / `view`:** una radiografía `imN` de un paciente.
* **`knee_id`:** `(patient_id, fx|normal)`; unidad de evaluación.
* **Paciente mixto:** aporta 2 rodillas (fx + contralateral normal).
* **OOF (out-of-fold):** predicción de cada rodilla hecha por el modelo del fold donde
  esa rodilla fue **test**.
* **Logit:** salida pre-sigmoide del head; `prob = σ(logit)`.
* **AUROC:** área bajo ROC; solo depende del **orden** de los scores.
* **AUPRC:** área bajo precisión-recall; sensible a la prevalencia.
* **`pos_weight`:** `n_neg/n_pos`, compensa desbalance en la BCE.
* **`state_dict`:** mapa nombre→tensor (parámetros **y** buffers).
* **Bootstrap por clusters:** remuestrear **pacientes** con reemplazo (arrastran sus
  rodillas).
* **Pareado:** mismo resample para A y B ⇒ aísla la diferencia.
* **Pooling (multi-semilla):** concatenar las distribuciones bootstrap de todas las
  semillas para el IC final.
* **Variabilidad de optimización vs de muestreo:** entre semillas vs por el bootstrap.
* **pHash / Hamming:** hash perceptual DCT / nº de bits distintos.

---

## 17. Tabla maestra de funciones

| # | Archivo | Función/clase | Propósito | Entrada | Salida | Mate/estadística | Librería | Prueba mínima | Depende de |
|--:|---|---|---|---|---|---|---|---|---|
| 1 | utils | `set_seed` | fijar semillas | `seed` | `None` | — (infra) | random/np/torch | #(seed reproducible) | — |
| 2 | utils | `get_logger` | logger idempotente | name/logfile | Logger | — | logging | no duplica handlers | — |
| 3 | utils | `load_config` | leer YAML | path | dict | — | yaml | round-trip | — |
| 4 | utils | `save_config` | escribir YAML | cfg,path | archivo | — | yaml | round-trip | — |
| 5 | utils | `_git_hash` | commit actual | — | str | — | subprocess | devuelve hash o 'unknown' | — |
| 6 | utils | `dump_versions` | entorno→JSON | path | dict+archivo | — | import dinámico | JSON con keys | _git_hash |
| 7 | utils | `resolve_device` | cpu/cuda | pref | str | — | torch | 'cpu' sin GPU | — |
| 8 | data | `knee_id_of` | id de rodilla | pid,label | str | 𝟙[label≤6] | — | #1 | — |
| 9 | data | `y_from_label` | binariza | label | int | 𝟙[label≠7] | — | #2 | — |
| 10 | data | `build_index_from_view_records` | índice/vista | records | DataFrame | — | pandas | #3 | knee_id_of,y_from_label |
| 11 | data | `find_mixed_patients` | pacientes 2 rodillas | df | set | nunique>1 | pandas | #4 | — |
| 12 | data | `build_toy_index` | índice sintético | cfg,seed | DataFrame | RNG | numpy | ≥n_mixed | build_index_from_view_records |
| 13 | data | `_patient_id_from_path` | id de nombre | path | int | — | re | regex | — |
| 14 | data | `list_platif_files` | listar .mat | root | list | — | glob | 186 archivos | — |
| 15 | data | `load_platif_struct` | leer .mat v5 | path | mat_struct | — | scipy.io | #24 | — |
| 16 | data | `_view_names` | campos imN | struct | list | — | re | ignora Coronal_CT | — |
| 17 | data | `read_view_label` | label de vista | struct,view | int | — | numpy | int 1..7 | — |
| 18 | data | `read_view_image` | OriginalImage | struct,view | ndarray | — | numpy | float32 2D | — |
| 19 | data | `build_real_index` | índice real | cfg | DataFrame | — | pandas | #25 | list_platif_files,assert_real_index |
| 20 | data | `assert_real_index` | congelar GT | df | None/abort | conteos | pandas | #25 | find_mixed_patients |
| 21 | data | `_find_col` | localizar columna | cols,needles | str/None | — | — | subcadena | — |
| 22 | data | `cross_check_xlsx` | verificar xlsx | df,xlsx | DataFrame | — | pandas/openpyxl | discrepancias | _find_col |
| 23 | data | `build_index` | toy/real switch | cfg,seed | DataFrame | — | — | despacha | build_toy/real |
| 24 | data | `ViewDataset.__init__` | dataset vista | df,cfg | — | — | torch | construye | — |
| 25 | data | `ViewDataset.__len__` | nº vistas | — | int | — | — | == len(df) | — |
| 26 | data | `ViewDataset._toy_image` | ruido determinista | pid,view,y | tensor | RNG sembrado | torch | #9 | — |
| 27 | data | `ViewDataset._get_struct` | caché struct | path | struct | LRU | — | reusa | load_platif_struct |
| 28 | data | `ViewDataset._base_image` | preproc [0,1] | path,view | tensor 1×S×S | P1/P99,clip,bilinear | numpy/torch | rango [0,1] | read_view_image |
| 29 | data | `ViewDataset._finalize` | 3ch+ImageNet | 1×S×S | 3×S×S | (x−μ)/σ | torch | forma 3×S×S | — |
| 30 | data | `ViewDataset._real_image` | pipeline real | path,view | tensor | — | torch | — | _base_image,_finalize |
| 31 | data | `ViewDataset.__getitem__` | (img,y,i) | i | tuple | — | torch | #9 | _toy/_real_image |
| 32 | data | `subset_index` | filtrar pacientes | df,ids | DataFrame | — | pandas | ⊆ ids | — |
| 33 | qc | `_phash` | hash perceptual | img | int | DCT+mediana | scipy.fftpack | #26 | — |
| 34 | qc | `_hamming` | distancia bits | a,b | int | XOR+popcount | — | ==1 | — |
| 35 | qc | `_read_view_arrays` | (img,BW) | struct,view | tuple | — | numpy | — | read_view_image |
| 36 | qc | `run_qc` | criterios QC | cfg,index | (excl,flow) | umbrales | pandas | #26 | _phash,_hamming |
| 37 | qc | `_flow_diagram` | conteos QC | index,excl | dict | — | pandas | full/after | — |
| 38 | qc | `apply_exclusions` | quitar vistas | index,excl | DataFrame | — | pandas | sin excluidas | — |
| 39 | qc | `build_qc_cohort` | cohorte post-QC | cfg | (cohort,excl) | — | — | 419 vistas | build_real_index,run_qc,apply_exclusions |
| 40 | splits | `_patient_labels` | etiqueta paciente | df | Series | max | pandas | **muerta** | — |
| 41 | splits | `make_folds` | folds+val interno | df,n,iv,seed | list[dict] | k-fold estrat. agrup. | sklearn | #6–#8 | _fold_prevalence |
| 42 | splits | `_prev` | prevalencias | df,patients | dict | media y | pandas | — | — |
| 43 | splits | `_fold_prevalence` | prev por split | df,tr,va,te | dict | — | pandas | — | _prev |
| 44 | splits | `patient_positivity` | pid→bool | df | dict | max=OR | pandas | — | — |
| 45 | splits | `nested_stratified_subsets` | subconjuntos anidados | patients,is_pos,fr,seed | dict | prefijos estratif. | numpy/math | ⊆ anidado | — |
| 46 | splits | `check_no_leakage` | guardia fuga | folds,df | None/abort | partición+inyectiva | pandas | #5–#8 | — |
| 47 | splits | `save_manifest` | folds→JSON | folds,df,path,seed | archivo | — | json | #27 | — |
| 48 | model | `remap_radimagenet_keys` | renombrar claves | state_dict | OrderedDict | — | — | backbone.0→conv1 | — |
| 49 | model | `_torch_load_state_dict` | load robusto | path | dict | — | torch | carga .pt | — |
| 50 | model | `load_matched_weights` | casar+contar | model,src,exp | int/abort | conteo tensores | torch | #13,#14 | — |
| 51 | model | `build_model` | ResNet-50/comp | cfg,weights | nn.Module | — | timm | #11–#14 | load_matched_weights |
| 52 | train | `_make_loader` | DataLoader | df,… | DataLoader | — | torch | #10 | ViewDataset |
| 53 | train | `_build_optimizer` | Adam 2 grupos | model,cfg | Optimizer | grupos LR | torch | #18 | — |
| 54 | train | `_pos_weight` | n_neg/n_pos | df,device | tensor/None | ratio clases | torch | #15 | — |
| 55 | train | `_predict_logits` | inferencia | model,loader | (logit,y,idx) | — | torch.no_grad | determinista | — |
| 56 | train | `_choose_threshold` | Youden/F1 | logits,y | float | matriz confusión | numpy | #19 | — |
| 57 | train | `train_one_fold` | entrenar fold | cfg,…,model | (pred_df,thr) | BCE+Adam+ES+AUROC | torch/sklearn | overfit lote | _make_loader,_pos_weight,_predict_logits,_choose_threshold |
| 58 | eval | `aggregate_views_to_knee` | media logits | pred_df | DataFrame | z̄, σ(z̄) | pandas | #20 | — |
| 59 | eval | `knee_metrics` | AUROC/AUPRC | knee_df | dict | AUROC/AUPRC/prev | sklearn | #21 | — |
| 60 | eval | `paired_bootstrap_delta_auroc` | ΔAUROC IC | knee_a,knee_b | dict | bootstrap cluster pareado | numpy/sklearn | #22 | pooled_ci,bootstrap_pvalue |
| 61 | eval | `pooled_ci` | IC percentil | deltas,ci | (lo,hi) | cuantiles | numpy | rodea obs | — |
| 62 | eval | `bootstrap_pvalue` | p dos colas | deltas | float | 2·min(P<0,P>0) | numpy | A=B→≈1 | — |
| 63 | run_experiment | `parse_args` | CLI | — | args | — | argparse | — | — |
| 64 | run_experiment | `_json_safe_deltas` | quitar _deltas | deltas | dict | — | — | serializable | — |
| 65 | run_experiment | `run_once` | CV 1 semilla | cfg,seed,… | dict+artefactos | — | (todo) | #23 | build_index/qc,make_folds,build_model,train_one_fold,eval.* |
| 66 | run_experiment | `main` | script | — | artefactos | — | — | #23 | run_once |
| 67 | run_multiseed | `_mean_std` | media/std | xs | (m,s) | media, std ddof=1 | numpy | — | — |
| 68 | run_multiseed | `main` | barrido semillas | cfg | multiseed_summary | pooling IC | numpy | — | run_once,pooled_ci,bootstrap_pvalue |
| 69 | run_curve | `_mean_std` | media/std | xs | (m,s) | — | numpy | — | — |
| 70 | run_curve | `main` | curva eficiencia | cfg | curve_summary | interacción proc×frac | (todo) | — | nested_stratified_subsets,train_one_fold,eval.* |
| 71 | failure_analysis | `youden_threshold` | umbral Youden | y,prob | float | argmax(tpr−fpr) | sklearn | — | — |
| 72 | failure_analysis | `main` | análisis fallos | cfg,oof | stdout | matriz confusión | pandas/sklearn | — | youden_threshold,_find_col |
| 73 | make_figures | `_save` | png+pdf | fig,path | archivos | — | matplotlib | — | — |
| 74 | make_figures | `figure_efficiency_curve` | curva | curve,out | figura | — | matplotlib | #28 | _save |
| 75 | make_figures | `figure_forest_delta100` | forest ΔAUROC | multiseed,out | figura | — | matplotlib | #28 | _save |
| 76 | make_figures | `table_provenance_metrics` | tabla CSV | multiseed,out | CSV | — | pandas | #28 | — |
| 77 | make_figures | `main` | script figuras | args | artefactos | — | — | #28 | las tres anteriores |
| 78 | tests | `_synthetic_index` + `test_*` | cero fuga | — | asserts | — | numpy | #5–#8 | make_folds,check_no_leakage |

---

## Anexo A — `eval.py` con plantilla completa de 8 pasos

Expansión del **grupo `eval.py`** (el núcleo estadístico) a la plantilla completa:
para cada función, los 8 pasos (firma vacía → pasos → pseudocódigo → ejercicio `TODO`
→ pistas → prueba → solución → explicación) más **Concepto matemático · dónde
verificarlo · cómo checarlo**. Reconstruye estas cinco funciones **en este orden**
(cada una depende de la anterior) y no uses predicciones reales hasta que los bloques
de comprobación pasen.

### A.1 `aggregate_views_to_knee(view_pred_df)`  (`eval.py:14`)

**Concepto matemático.** Media de logits por rodilla: `z̄=(1/m)Σ_j z_j`, `prob=σ(z̄)`.
**Dónde verificarlo:** desigualdad de Jensen (σ convexa/cóncava por tramos ⇒
`σ(mean z)≠mean σ(z)`). **Cómo checarlo:** bloque **C6** de §9.1.

1. **Firma vacía**
   ```python
   def aggregate_views_to_knee(view_pred_df):
       ...
   ```
2. **Pasos.** (a) agrupar por `knee_id`; (b) por grupo: `patient_id`/`side_class`/`y`
   con `first`, `logit` con `mean`, `n_views` con `size`; (c) assertar que `y_vista` es
   constante dentro de cada `knee_id`; (d) añadir `prob=σ(logit)`.
3. **Pseudocódigo**
   ```
   g = groupby(knee_id)
   out = g.agg(patient_id=first, side_class=first, y=first, logit=mean, n_views=size)
   assert todas las rodillas tienen y_vista único
   out.prob = 1/(1+exp(-out.logit)); devolver out
   ```
4. **Ejercicio `TODO`**
   ```python
   def aggregate_views_to_knee(df):
       # TODO g=groupby('knee_id'); agg(...); assert y constante; prob=sigmoid(logit)
       raise NotImplementedError
   ```
5. **Pistas.** `groupby.agg(col=("orig","func"))`; la aserción usa
   `g["y_vista"].nunique()` y comprueba que todos son 1.
6. **Prueba** (test #20)
   ```python
   import pandas as pd, numpy as np
   df = pd.DataFrame({"knee_id":["9:fx"]*3,"patient_id":[9]*3,"side_class":["fx"]*3,
                      "y_vista":[1,1,1],"logit":[2.,-2.,0.]})
   out = aggregate_views_to_knee(df)
   assert out.loc[0,"n_views"]==3 and abs(out.loc[0,"logit"])<1e-9
   assert abs(out.loc[0,"prob"]-0.5)<1e-9
   ```
7. **Solución de referencia:** `eval.py:14-32`.
8. **Explicación.** La agregación en el espacio **logit** (no probabilidad) es la regla
   congelada; `first` para `y` es válido **porque** la aserción garantiza que `y` es
   constante en la rodilla — si no lo fuera (bug de `knee_id`), la aserción lo caza.

### A.2 `knee_metrics(knee_df)`  (`eval.py:35`)

**Concepto matemático.** AUROC `=P(score⁺>score⁻)` (U de Mann–Whitney normalizada),
**invariante a transformaciones monótonas** (por eso se calcula sobre `logit`); AUPRC
`=Σ(R_n−R_{n−1})P_n`; prevalencia `=mean(y)`. **Dónde verificarlo:** Fawcett (2006);
Davis & Goadrich (2006); docs sklearn. **Cómo checarlo:** bloque **C5** de §9.1.

1. **Firma vacía**
   ```python
   def knee_metrics(knee_df):
       ...
   ```
2. **Pasos.** (a) `y=int`; (b) `score=logit`, `prob=prob`; (c) si hay 2 clases: AUROC
   sobre `score`, AUPRC sobre `prob`; si no, `nan`; (d) devolver dict con `n_knees`,
   `prevalence`, `auroc`, `auprc`.
3. **Pseudocódigo**
   ```
   y=knee.y; score=knee.logit; prob=knee.prob
   auroc = roc_auc_score(y,score) si |unique(y)|==2 si no nan
   auprc = average_precision_score(y,prob) si |unique(y)|==2 si no nan
   devolver {n_knees, prevalence=mean(y), auroc, auprc}
   ```
4. **Ejercicio `TODO`.** Implementa el dict; cuida el caso de una sola clase (→ `nan`).
5. **Pistas.** `len(np.unique(y))==2` como guarda; AUROC sobre **logit** (no `prob`) es
   equivalente y evita perder precisión numérica de σ.
6. **Prueba** (test #21)
   ```python
   import pandas as pd
   knee = pd.DataFrame({"y":[0,0,1,1],"logit":[-1,-0.5,0.5,1.0],
                        "prob":[0.27,0.38,0.62,0.73]})
   m = knee_metrics(knee)
   assert m["auroc"]==1.0 and m["n_knees"]==4 and m["prevalence"]==0.5
   ```
7. **Solución de referencia:** `eval.py:35-46`.
8. **Explicación.** Usar el `logit` para AUROC es correcto porque el AUROC solo depende
   del **orden** de los scores y σ es estrictamente creciente (comprobado en **C5**). La
   guarda de una sola clase evita que `roc_auc_score` lance.

### A.3 `paired_bootstrap_delta_auroc(knee_a, knee_b, n_boot, ci, seed, return_deltas)`  (`eval.py:49`)

**Concepto matemático.** Bootstrap **por clusters (pacientes)** y **pareado**:
`Δ̂=AUROC(y,s_A)−AUROC(y,s_B)`; en cada réplica se muestrean pacientes con reemplazo y
se recalcula `Δ*`. **Dónde verificarlo:** Efron & Tibshirani (1993) cap. 13 (bootstrap);
Davison & Hinkley (1997) §3.8 y Field & Welsh (2007) (datos agrupados). **Cómo checarlo:**
con A=B, `Δ*≈0` y el IC rodea 0 (test #22).

1. **Firma vacía**
   ```python
   def paired_bootstrap_delta_auroc(knee_a, knee_b, n_boot=2000, ci=0.95, seed=0,
                                    return_deltas=False):
       ...
   ```
2. **Pasos.** (a) alinear A y B por `knee_id` (mismas `y`); (b) mapear
   `patient→índices de sus rodillas`; (c) `obs=ΔAUROC`; (d) `n_boot` réplicas: muestrear
   `n_pat` pacientes con reemplazo, concatenar índices, saltar si una sola clase,
   recalcular `Δ`; (e) devolver `obs`, `pooled_ci`, `bootstrap_pvalue`, contadores.
3. **Pseudocódigo** (ver el del §7, ANCLA).
4. **Ejercicio `TODO`.** Implementa el bucle de réplicas; precomputa `knees_by_pat`
   **fuera** del bucle.
5. **Pistas.** `patients[rng.integers(0,n_pat,size=n_pat)]` = muestreo de clusters;
   `np.concatenate([knees_by_pat[p] for p in sampled])`.
6. **Prueba** (test #22)
   ```python
   import pandas as pd, numpy as np
   base = pd.DataFrame({"knee_id":["1:fx","1:normal","2:fx","3:normal"],
                        "patient_id":[1,1,2,3],"y":[1,0,1,0],
                        "logit":[1.2,-0.3,0.8,-1.1]})
   d = paired_bootstrap_delta_auroc(base, base.copy(), n_boot=500, seed=0)
   assert abs(d["delta_auroc_observed"])<1e-12        # A==B -> 0
   assert d["ci_low"]<=0<=d["ci_high"]                # IC rodea 0
   ```
7. **Solución de referencia:** `eval.py:49-104`.
8. **Explicación.** Remuestrear **pacientes** (no rodillas) respeta que las dos rodillas
   de un mixto no son independientes; el **mismo** resample para A y B cancela la
   varianza común y aísla la diferencia (pareado). Réplicas de una sola clase se
   descartan porque el AUROC no está definido ahí.

### A.4 `pooled_ci(deltas, ci)`  (`eval.py:107`)

**Concepto matemático.** Intervalo **percentil**: `[q_{α/2}, q_{1−α/2}]` con
`α=1−ci`. **Dónde verificarlo:** Efron & Tibshirani (1993) cap. 13. **Cómo checarlo:**
`pooled_ci(d,0.95)` == `tuple(np.quantile(d,[0.025,0.975]))`.

1. **Firma vacía**
   ```python
   def pooled_ci(deltas, ci=0.95):
       ...
   ```
2. **Pasos.** (a) si vacío → `(nan,nan)`; (b) `alpha=(1−ci)/2`; (c) devolver los dos
   cuantiles.
3. **Pseudocódigo**
   ```
   si len(deltas)==0: return nan,nan
   alpha=(1-ci)/2; return quantile(deltas,alpha), quantile(deltas,1-alpha)
   ```
4. **Ejercicio `TODO`.** Implementa los dos cuantiles con `numpy.quantile`.
5. **Pistas.** `ci=0.95 ⇒ alpha=0.025 ⇒ [q_0.025, q_0.975]`.
6. **Prueba**
   ```python
   import numpy as np
   d = np.random.default_rng(0).normal(size=100000)
   lo,hi = pooled_ci(d,0.95)
   assert abs(lo-np.quantile(d,0.025))<1e-9 and abs(hi-np.quantile(d,0.975))<1e-9
   ```
7. **Solución de referencia:** `eval.py:107-112`.
8. **Explicación.** Es la forma **percentil** del bootstrap (la más simple); se separa
   en su propia función porque `run_multiseed`/`run_curve` la aplican sobre el **pool**
   de `_deltas` de todas las semillas.

### A.5 `bootstrap_pvalue(deltas)`  (`eval.py:115`)

**Concepto matemático.** p-valor bootstrap de **dos colas** para `H0: ΔAUROC=0`:
`p=2·min(P(Δ*<0), P(Δ*>0))`, acotado a 1. **Dónde verificarlo:** Davison & Hinkley
(1997). **Cómo checarlo:** A=B ⇒ `p≈1`; una distribución toda positiva ⇒ `p≈0`.

1. **Firma vacía**
   ```python
   def bootstrap_pvalue(deltas):
       ...
   ```
2. **Pasos.** (a) si vacío → `nan`; (b) `frac_neg=mean(deltas<0)`, `frac_pos=mean(>0)`;
   (c) `p=min(1, 2·min(frac_neg,frac_pos))`.
3. **Pseudocódigo**
   ```
   si vacío: nan
   return min(1, 2*min(mean(deltas<0), mean(deltas>0)))
   ```
4. **Ejercicio `TODO`.** Implementa las dos fracciones y la combinación de dos colas.
5. **Pistas.** `mean(bool_array)` = proporción; el `2·min(...)` reparte la masa a ambos
   lados.
6. **Prueba**
   ```python
   import numpy as np
   assert bootstrap_pvalue(np.array([1.,2.,3.]))==0.0        # todo positivo
   d = np.random.default_rng(0).normal(size=100000)
   assert abs(bootstrap_pvalue(d)-1.0)<0.05                   # centrado en 0 -> ~1
   ```
7. **Solución de referencia:** `eval.py:115-121`.
8. **Explicación.** Cuenta cuánta masa bootstrap cae del lado «equivocado» respecto a 0;
   al ser cluster-robusto (los `Δ*` vienen del bootstrap por pacientes), hereda la
   dependencia intra-paciente correctamente.

> **Cierre del anexo.** Con A.1–A.5 reconstruidas y sus bloques de comprobación en
> verde, tienes todo `eval.py` sin mirar la solución. Si quieres, aplica esta misma
> plantilla de 8 pasos al siguiente grupo (primitivas de `train.py`:
> `_pos_weight`, `_choose_threshold`, `_predict_logits`, `_build_optimizer`).

---

## Anexo B — primitivas de `train.py` con plantilla completa

Las cuatro primitivas que alimentan `train_one_fold`, cada una con los 8 pasos y
**Concepto matemático · dónde verificarlo · cómo checarlo**. Impleméntalas y prueba
cada una **por separado** antes de `train_one_fold`. (`_make_loader` es plumbing puro:
envuelve `ViewDataset` en `DataLoader`; su único punto fino es el parámetro
`full_index` no usado — ver §15 — y se prueba con el test #10.)

### B.1 `_pos_weight(index_df, device)`  (`train.py:47`)

**Concepto matemático.** Peso de clase para la BCE: `w = n_neg / n_pos` (a nivel
**vista** del train). Multiplica el término de los positivos en la pérdida para
compensar el desbalance; si `w>1`, un falso negativo «pesa» más. **Dónde verificarlo:**
docs de `torch.nn.BCEWithLogitsLoss` (argumento `pos_weight`). **Cómo checarlo:** con
`n_pos=2, n_neg=6` debe dar `tensor([3.0])`; con una clase ausente, `None`.

1. **Firma vacía**
   ```python
   def _pos_weight(index_df, device):
       ...
   ```
2. **Pasos.** (a) contar `n_pos`, `n_neg` desde `y_vista`; (b) si alguna es 0 → `None`;
   (c) devolver `tensor([n_neg/n_pos])` en `device`.
3. **Pseudocódigo**
   ```
   n_pos = sum(y_vista==1); n_neg = sum(y_vista==0)
   si n_pos==0 o n_neg==0: return None
   return tensor([n_neg/n_pos], float32, device)
   ```
4. **Ejercicio `TODO`.** Implementa el conteo y la guarda de clase ausente.
5. **Pistas.** `int((df["y_vista"]==1).sum())`; devuelve `None` (no 0) si falta una
   clase — así `train_one_fold` sabe pasar `pos_weight=None`.
6. **Prueba** (test #15)
   ```python
   import pandas as pd, torch
   df = pd.DataFrame({"y_vista":[1,1,0,0,0,0,0,0]})   # n_pos=2, n_neg=6
   assert float(_pos_weight(df,"cpu")) == 3.0
   df1 = pd.DataFrame({"y_vista":[1,1,1]})             # sin negativos
   assert _pos_weight(df1,"cpu") is None
   ```
7. **Solución de referencia:** `train.py:47-52`.
8. **Explicación.** Se calcula sobre el **train** (no val/test) para no filtrar
   información; a nivel vista porque la pérdida opera por vista. `None` cuando falta una
   clase evita dividir por cero y desactiva el reponderado en ese fold.

### B.2 `_build_optimizer(model, cfg)`  (`train.py:29`)

**Concepto matemático.** **Adam** con **grupos de parámetros**: el backbone (todo menos
`fc.`) recibe `lr·backbone_lr_mult` y el head `lr`. Con `mult=1.0` (config actual) el LR
es **uniforme** — decisión deliberada para un contraste de procedencia limpio. **Dónde
verificarlo:** Kingma & Ba (2015), arXiv:1412.6980 (Adam); docs `torch.optim.Adam`
(param groups). **Cómo checarlo:** el optimizador tiene **2 grupos** y sus `lr` son
`lr·mult` y `lr`.

1. **Firma vacía**
   ```python
   def _build_optimizer(model, cfg):
       ...
   ```
2. **Pasos.** (a) leer `lr`, `weight_decay`, `backbone_lr_mult`; (b) separar parámetros
   entrenables en `head` (nombre empieza por `fc.`) y `backbone` (resto); (c) construir
   dos grupos con sus `lr`; (d) devolver `Adam(groups, lr, weight_decay)`.
3. **Pseudocódigo**
   ```
   para (name,p) en model.named_parameters() si p.requires_grad:
       (head si name.startswith('fc.') si no backbone).append(p)
   groups = [{'params':backbone,'lr':lr*mult}, {'params':head,'lr':lr}]
   return Adam(groups, lr=lr, weight_decay=wd)
   ```
4. **Ejercicio `TODO`.** Implementa la separación por nombre y los dos grupos.
5. **Pistas.** `model.named_parameters()` da `(nombre, tensor)`; filtra
   `p.requires_grad`; el head de ResNet-50 en timm se llama `fc.*`.
6. **Prueba** (necesita torch)
   ```python
   opt = _build_optimizer(build_model(cfg,"random"), cfg)
   assert len(opt.param_groups) == 2
   lr = float(cfg["train"]["lr"]); mult = float(cfg["train"].get("backbone_lr_mult",1.0))
   assert abs(opt.param_groups[0]["lr"] - lr*mult) < 1e-12   # backbone
   assert abs(opt.param_groups[1]["lr"] - lr) < 1e-12        # head
   ```
7. **Solución de referencia:** `train.py:29-44`.
8. **Explicación.** Los **grupos de parámetros** permiten un LR discriminativo
   (típico en fine-tuning: backbone más lento). Aquí `mult=1.0` lo iguala **a propósito**:
   si el backbone y el head aprendieran a ritmos distintos, la comparación de procedencia
   mezclaría «mejor init» con «mejor esquema de LR».

### B.3 `_predict_logits(model, loader, device)`  (`train.py:55`)

**Concepto matemático.** *No implementa una operación matemática central; es inferencia
(plumbing) con dos garantías clave:* `model.eval()` (BatchNorm/Dropout en modo
evaluación) y `torch.no_grad()` (sin construir el grafo de gradientes → memoria y
velocidad). **Dónde verificarlo:** docs PyTorch `Module.eval` y `torch.no_grad`. **Cómo
checarlo:** dos llamadas con el mismo modelo y el mismo orden de datos dan **idénticos**
logits.

1. **Firma vacía**
   ```python
   def _predict_logits(model, loader, device):
       ...
   ```
2. **Pasos.** (a) `model.eval()`; (b) bajo `torch.no_grad()`, iterar el loader; (c)
   `out = model(img).squeeze(-1)` (de `(B,1)` a `(B,)`); (d) acumular logits, `y`, `idx`;
   (e) `np.concatenate` de los tres.
3. **Pseudocódigo**
   ```
   model.eval()
   with no_grad:
     para (img,y,idx) en loader:
        out = model(img.to(device)).squeeze(-1)
        acumular out.cpu().numpy(), y.numpy(), idx
   return concat(logits), concat(ys), concat(idxs)
   ```
4. **Ejercicio `TODO`.** Implementa el bucle sin gradiente; recuerda `.detach().cpu()`.
5. **Pistas.** El tercer elemento del batch (`idx`) es el índice de fila que devuelve
   `ViewDataset.__getitem__`; **no lo pierdas**: mapea las predicciones de vuelta a `te_df`.
6. **Prueba** (necesita torch)
   ```python
   l1,y1,i1 = _predict_logits(model, loader, "cpu")
   l2,y2,i2 = _predict_logits(model, loader, "cpu")
   import numpy as np; assert np.allclose(l1,l2)     # determinista en eval
   assert l1.ndim==1                                 # (N,), no (N,1)
   ```
7. **Solución de referencia:** `train.py:55-65`.
8. **Explicación.** `eval()` es imprescindible: en `train()` la BatchNorm usaría las
   estadísticas del batch y las predicciones dependerían del orden/tamaño del lote.
   `no_grad()` no cambia el resultado pero evita gastar memoria en el grafo. `squeeze(-1)`
   colapsa el head de un logit `(B,1)→(B,)` para alinear con `y`.

### B.4 `_choose_threshold(logits, y, metric)`  (`train.py:68`)

**Concepto matemático.** Umbral óptimo sobre **probabilidades** que maximiza **Youden**
`J=sens+spec−1=TPR−FPR` o **F1** `=2·TP/(2·TP+FP+FN)`, barriendo todos los valores
únicos de `prob`. Es un punto de **operación**: informa la matriz de confusión pero
**no** afecta al AUROC (que integra sobre todos los umbrales). **Dónde verificarlo:**
Youden (1950), *Cancer*; docs sklearn (F1, ROC). **Cómo checarlo:** en un caso separable
`[-2,-0.5,0.5,2]` con `y=[0,0,1,1]`, ambos criterios devuelven `σ(0.5)=0.6225`.

1. **Firma vacía**
   ```python
   def _choose_threshold(logits, y, metric):
       ...
   ```
2. **Pasos.** (a) `probs=σ(logits)`; (b) si una sola clase → `0.5`; (c) para cada `t` en
   `unique(probs)`: `pred=probs≥t`, contar TP/FP/FN/TN, calcular `score` (f1 o youden);
   (d) devolver el `t` de mayor `score`.
3. **Pseudocódigo**
   ```
   probs=1/(1+exp(-logits)); si |unique(y)|<2: return 0.5
   best_t,best=0.5,-1
   para t en unique(probs):
     pred=probs>=t; tp,fp,fn,tn=confusion(pred,y)
     score = 2tp/(2tp+fp+fn)  si f1  ;  sens+spec-1  si youden
     si score>best: best,best_t=score,t
   return best_t
   ```
4. **Ejercicio `TODO`.** Implementa el barrido + matriz de confusión + ambos scores.
5. **Pistas.** Barre solo los **valores únicos** de `prob` (los cambios de decisión solo
   ocurren ahí); cuida denominadores cero (`if (tp+fn) else 0`).
6. **Prueba** (test #19, verificada numéricamente)
   ```python
   import numpy as np
   logits = np.array([-2.,-0.5,0.5,2.]); y = np.array([0,0,1,1])
   assert abs(_choose_threshold(logits,y,"youden") - 0.6225) < 1e-3
   assert abs(_choose_threshold(logits,y,"f1")     - 0.6225) < 1e-3
   assert _choose_threshold(logits, np.array([1,1,1,1]), "youden") == 0.5  # 1 clase
   ```
7. **Solución de referencia:** `train.py:68-90`.
8. **Explicación.** El umbral se elige en el **val interno**, nunca en test; es
   «informativo» porque el resultado primario (AUROC) es invariante a él. **Auditoría:**
   la línea `order = np.argsort(-probs)` se calcula y no se usa (§15) — puedes omitirla en
   tu reconstrucción sin cambiar el resultado.

> **Cierre.** Con B.1–B.4 en verde, `train_one_fold` (§7, ANCLA) es solo el pegamento:
> loaders → pos_weight → optimizador → bucle de época con early stopping en el AUROC de
> val → restaurar mejor estado → umbral → predicción OOF de test.

---

## Anexo C — método general para replicar experimentos/papers

> Plantilla reutilizable (independiente de este proyecto) para reproducir un paper o
> montar tu propia ablación rápido y sin engañarte. Es el patrón que sigue
> `ropec_baseline`, destilado en pasos.

**1. Congela la pregunta y la unidad de análisis ANTES de tocar datos.** Escribe en una
frase: qué variable manipulas, qué mantienes fijo, cuál es la **unidad** (aquí:
rodilla) y la **métrica primaria** (aquí: AUROC OOF). Si no puedes escribirlo, aún no
entiendes el experimento.

**2. Una sola fuente de verdad para la configuración.** Todo parámetro en un
`config.yaml`; guarda `config_used.yaml` + `versions.json` (git hash, librerías, GPU)
por corrida. Sin esto no hay reproducibilidad.

**3. Construye un modo *toy* que imite la estructura real.** Un dataset sintético que
reproduzca las rarezas del real (aquí: nº de vistas variable, pacientes mixtos) te deja
correr el pipeline **entero** en segundos y atrapar bugs de plumbing sin datos.

**4. Ataca la fuga de datos primero y con tests.** Define el split por la unidad
correcta (aquí: paciente), y escribe **tests de cero fuga** que fallen si algo se cruza.
No entrenes hasta que estén en verde. La fuga es el error #1 que invalida papers.

**5. Convierte cada invariante en una aserción.** Conteo de tensores al cargar pesos,
ground-truth congelado, etiqueta constante por unidad. Una aserción ruidosa hoy evita un
resultado falso silencioso mañana.

**6. Sube por escalones comprobables** (el orden de esta guía): infra → datos puros →
splits → dataset → modelo → una pasada/overfit de un lote → evaluación con ejemplos a
mano → smoke end-to-end → datos reales. Cada escalón produce algo ejecutable.

**7. Separa las dos variabilidades.** Optimización (varias semillas) y muestreo
(bootstrap). Repórtalas distinto; una no sustituye a la otra.

**8. Reporta efecto + intervalo, no significancia binaria.** ΔMétrica con IC (bootstrap
por la unidad de agrupación correcta). Guarda las predicciones OOF y los arrays crudos
para poder re-agregar.

**9. Preprocesamiento idéntico entre comparadores.** Si comparas A vs B, cualquier
diferencia que no sea la variable manipulada contamina el contraste. (En una ablación,
un preprocesamiento subóptimo pero **igual** para todos sigue dando un contraste válido.)

**10. Escribe primero lo que NO vas a afirmar.** Una lista «NO afirmar» (superioridad
clínica, números proyectados, etc.) te protege de sobre-vender. Redacta los resultados
solo desde los artefactos reales.

**Checklist exprés para replicar un paper ajeno:** (a) ¿cuál es su unidad y su split?
(b) ¿qué exclusiones/QC aplican y en qué orden? (c) ¿qué está fijo y qué varían?
(d) ¿qué métrica y qué test de incertidumbre? (e) ¿publican folds/semillas/pesos?
Si alguna no está clara en el paper, ese es exactamente el hueco que tu réplica (o tu
propio paper) puede llenar.

---

## 18. Checklist: «Puedo reconstruir ROPEC baseline sin IA»

Marca solo lo que puedes **explicar, probar, modificar y reconstruir** (no solo
ejecutar):

- [ ] Explico la pregunta, los 3 comparadores y por qué la única variable es la
  procedencia de pesos.
- [ ] Distingo paciente / vista / rodilla y explico por qué `knee_id ≠ patient_id` y por
  qué hay 190 rodillas y 186 pacientes.
- [ ] Construyo el índice (`build_index_from_view_records`) y detecto pacientes mixtos.
- [ ] Implemento `make_folds` y **demuestro cero fuga** con `check_no_leakage` +
  `tests/test_splits.py`.
- [ ] Explico cada transformación de imagen: P1/P99 → `[0,1]` → resize bilineal → 3
  canales → estandarización ImageNet.
- [ ] Cargo y **verifico** los 3 modelos y entiendo la aserción de conteo (RadImageNet =
  318) y por qué `strict=False` sin contar es un error científico silencioso.
- [ ] Entreno un fold (`train_one_fold`) con early stopping en val y restauración del
  mejor estado; hago overfit de un lote como sanity.
- [ ] Explico `BCEWithLogitsLoss`, `pos_weight=n_neg/n_pos`, Adam y por qué el umbral es
  informativo pero AUROC no depende de él.
- [ ] Obtengo predicciones OOF por vista y las **agrego por rodilla** (media de logits).
- [ ] Calculo AUROC/AUPRC a nivel rodilla.
- [ ] Implemento el **bootstrap pareado por pacientes**, el IC percentil y el p-valor de
  dos colas; explico por qué se remuestrean pacientes.
- [ ] Corro múltiples semillas y distingo variabilidad de optimización vs de muestreo.
- [ ] Construyo la curva de eficiencia con subconjuntos **anidados y estratificados** e
  interpreto la interacción procedencia×fracción.
- [ ] Ejecuto el análisis de fallos y genero tablas y figuras.
- [ ] Reproduzco el experimento completo **solo desde `config.yaml`** y sé qué artefacto
  produce cada script.
- [ ] Identifico el código muerto y las discrepancias de la §15 **sin cambiarlos** antes
  de reproducir el baseline.

> Objetivo cumplido cuando puedes, para **cualquier** función de la tabla maestra,
> escribir su firma, su pseudocódigo, una prueba mínima y explicar qué invariante
> científica protege — sin consultar la IA.
