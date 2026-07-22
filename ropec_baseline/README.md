# ROPEC 2026 — baseline de procedencia de preentrenamiento

Ablación controlada de **procedencia de preentrenamiento** con backbone único
**ResNet-50**: **ImageNet vs RadImageNet vs random** sobre **PlaTiF** (fractura de
platillo tibial, presencia/ausencia a **nivel rodilla**).

- Métrica primaria: **AUROC OOF**.
- Comparación: **ΔAUROC con bootstrap pareado remuestreando PACIENTES completos**
  (efecto + IC95%), + prevalencia y AUPRC. **No** significancia binaria.
- MURA está **fuera** de P1 (es el experimento MAE de la tesis; no se toca).

## Invariantes (heredados por todo el pipeline)

1. `StratifiedGroupKFold(5)` sobre `patient_id`; todas las vistas/rodillas de un
   paciente en el mismo fold. Split interno para early stopping y umbral.
2. Backbone único ResNet-50 (`timm`). Todos los comparadores comparten folds,
   augs, optimizador, épocas y presupuesto de tuning.
3. **Toda carga de pesos con aserción de conteo de tensores** (aborta si falla).
   RadImageNet ResNet-50 exige **318 tensores** o aborta.
4. Predicción a nivel rodilla: `knee_id = (patient_id, "fx" if label<=6 else
   "normal")`. Agregación vista→rodilla = **media de logits** dentro de cada
   `knee_id`, congelada antes de ver métricas.
5. Primaria = AUROC OOF; comparación = ΔAUROC bootstrap pareado por pacientes.

## Entorno (crea uno FRESCO)

Ningún env conda existente trae el stack completo. Crea uno nuevo:

```bash
conda create -n ropec python=3.11 -y
conda activate ropec
pip install -r requirements.txt
```

## Fase 0 — smoke test sobre dataset juguete (este paso)

El toy reproduce la estructura real de PlaTiF: nº de vistas variable por paciente
(1–8) y ≥1 paciente mixto (rodilla fx + rodilla normal) para ejercitar la
derivación de `knee_id` y las aserciones.

```bash
conda activate ropec
cd ropec_baseline
python run_experiment.py --config config.yaml --smoke   # 1 fold, ~rápido
# o el pipeline completo (5 folds):
python run_experiment.py --config config.yaml
```

Debe generar, bajo `outputs/<run_name>/`:

- `index.csv`, `folds_manifest.json`
- `<comparador>/fold_k_preds.csv` (OOF **a nivel rodilla**), `oof_knee_preds.csv`,
  `metrics.json`
- `delta_auroc.json`, `summary.json`, `versions.json`, `config_used.yaml`, `run.log`

## Estructura

| Archivo | Rol |
|---|---|
| `config.yaml` | Toda la parametrización (rutas absolutas a `/mnt/llm-storage`, salidas en el repo). |
| `utils.py` | Semillas, logging, dump de versiones, IO de config. |
| `data.py` | Índice + Dataset. Toy en Fase 0; PlaTiF real en Fase 1. |
| `splits.py` | `StratifiedGroupKFold(5)` por paciente + chequeo de cero fuga + manifiesto. |
| `model.py` | ResNet-50 (timm) + carga de pesos con **aserción de conteo**. |
| `train.py` | Bucle de entrenamiento por fold (early stopping + umbral). |
| `eval.py` | Agregación vista→rodilla, AUROC/AUPRC, ΔAUROC bootstrap pareado. |
| `run_experiment.py` | Orquestador end-to-end. |

## Roadmap de fases

- **Fase 0** (este repo): andamiaje + smoke toy. ← estás aquí
- **Fase 1**: conectar PlaTiF real (`.mat` v5) + QC + splits — reescribe
  `build_real_index()` en `data.py`.
- **Fase 2 (contingencia)**: RadImageNet (318 tensores) + ImageNet vs RadImageNet
  @100% → **CHECKPOINT / paper mínimo** (congelar + `git tag`).
- **Fase 3**: curva de eficiencia de etiquetas (aditivo).
- **Fase 5**: figuras, tablas, Methods "as-run".

**Nada de resultados inventados.** El bloque de Methods se redacta con números de
corridas reales del usuario.
