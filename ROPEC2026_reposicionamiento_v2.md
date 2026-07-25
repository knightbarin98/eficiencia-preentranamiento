# ROPEC v2 — Reposicionamiento tras el preprint + protocolo revisado

Actualiza el framing del paper de ROPEC después de dos hallazgos que estrechan tu novedad. Reemplaza la sección de alcance/novedad del plan base.

**Ajuste 14 jul 2026 (§4):** análisis primario = **interacción procedencia×fracción**; random como **curva de referencia**; punto al **10%**; agregación **vista→rodilla**; tamaño de efecto + CV repetida; prevalencia/AUPRC; **frontera Brier→P2**; **go/no-go MURA día 0**.

---

## 0. Veredicto sobre la crítica que subiste

Es la mejor revisión externa del proceso. **Adóptala casi entera.** Acertó en: hallar y leer bien el preprint, estrechar la novedad, exigir RadImageNet como comparador, corregir la unidad experimental, volver la limpieza parte del método, exigir ΔAUROC pareado, y detectar la contradicción en mi propio plan sobre el número de papers. Donde agrego valor es en el punto que ella no enfatizó (sección 1b) y en reconciliar el conteo (sección 5).

---

## 1. El paisaje cambió dos veces, no una

**1a. Apareció un vecino directo (el preprint).**
Elnakib, Saad & Al-Kabbany (2026), *Phenotyping Tibial Plateau Fractures via Self-Supervised Learning* (arXiv:2606.17295, 15 jun 2026): RadImageNet-ResNet-50 + SimCLR sobre PlaTiF (186→154 tras limpieza de 8 pasos), UMAP + k-means → 4 fenotipos, validación experta ciega. Tarea = **fenotipado no supervisado**, no detección. → Mata el framing "primera aplicación de SSL a radiografía de TPF".

**1b. (Lo más importante) La detección supervisada de TPF ya está resuelta.**
- **van der Gaast et al. (Knee 54:81-89, 2025):** GoogleNet+ResNet, 1506 radiografías / 753 pacientes, detección + clasificación Schatzker; sensibilidad 92.7%.
- **Huo et al. (Radiology Advances 2(3):umaf020, 2025):** MobileNetV3-YOLOv8 multicéntrico **con validación externa**, incluye fracturas ocultas.
- **Liu (RetinaNet):** exactitud 0.91, casi a la par de ortopedistas (0.92).

→ **No puedes vender "detección de TPF" como contribución.** Ni detección ni "primer SSL" están disponibles.

---

## 2. Tu celda abierta (más estrecha y más precisa)

> **¿Cómo gobierna la procedencia del preentrenamiento —imagen natural (ImageNet), radiología general (RadImageNet) y radiografía musculoesquelética de otra anatomía (MURA-SSL)— la eficiencia de etiquetas para detectar fractura de platillo tibial, bajo evaluación agrupada por paciente y comparación pareada?**

Por qué sigue abierta:
- van der Gaast/Huo **no** ablacionan procedencia de preentrenamiento ni curvas de eficiencia de etiquetas.
- El preprint **afirma** RadImageNet > ImageNet en cohortes pequeñas, pero **sin ablación controlada** (lo toma de la literatura de RadImageNet + una corrida preliminar).
- Tú provees exactamente esa prueba controlada. Esa es toda tu contribución — y es suficiente para ROPEC si se ejecuta limpia.

---

## 3. El preprint es un regalo, no un golpe

Úsalo activamente:
- **Cítalo explícitamente** en Related Work (lo contrario sería sospechoso: es de junio 2026 y usa tu dataset).
- **Construye sobre su protocolo de limpieza** (8 pasos, 186→154) — no lo copies a ciegas; para *detección* algunos criterios cambian (p. ej., excluir hardware posoperatorio tiene sentido, pero tu definición de positivo/negativo difiere de la de fenotipado).
- **Testea su afirmación no verificada** (RadImageNet > ImageNet) como tu contraste principal.
- **Posiciónate** como "la evaluación supervisada controlada de procedencia × eficiencia que el trabajo reciente asumió pero no realizó".

---

## 4. Protocolo revisado (incorpora la crítica)

**Comparadores (un solo backbone, ResNet-50):**
1. ImageNet
2. RadImageNet
3. MURA-SSL — solo si obtienes/entrenas pesos con la MISMA arquitectura
4. Random — solo como sanity check

**Contraste principal:**
- **Análisis primario = interacción procedencia × fracción de etiquetas:** ¿la ventaja de RadImageNet sobre ImageNet **se ensancha** al reducir etiquetas? (no dos curvas comparadas a ojo).
- **Garantizado:** RadImageNet vs ImageNet a través de **10/25/50/100 %** de etiquetas.
- **Random como curva de referencia** (no solo sanity): expresa la eficiencia en equivalencia de datos ("RadImageNet al 25 % iguala a random al 100 %").
- **Si el brazo MURA llega:** MURA-SSL vs RadImageNet a 25 % (mide si la especialización MSK compensa la distancia anatómica hombro/mano → rodilla). **Go/no-go el día 0:** solo si hay pesos SSL de MURA sobre ResNet-50 descargables; no se entrena SSL desde cero.

**Condición de justicia:** mismos folds, augmentations, optimizador, épocas y presupuesto de tuning para todas las inicializaciones. **MAE-ViT fuera** (va a la tesis); comparar MAE-ViT contra ImageNet-ResNet confundiría fuente con arquitectura.

**Unidad experimental (formulación correcta):**
- Asignación a folds: **paciente**.
- Predicción: **rodilla / lado radiográfico**.
- Remuestreo estadístico: **paciente**.
- Restricción: todas las vistas y rodillas de un paciente en el mismo fold.
- **Agregación vista→rodilla:** regla congelada antes de ver resultados (p. ej. media de logits de las vistas de una rodilla → una predicción por rodilla/lado).

**Definición de tarea:** *presencia vs. ausencia de fractura de platillo tibial* (no "fractura vs. normal": algunos casos con platillo normal pueden tener otras fracturas; el negativo no es "radiografía sana").

**Limpieza = método:** criterios de QC congelados **antes** de ver resultados; diagrama `186 pacientes → X excluidos por criterio → N pacientes, M rodillas`. No excluyas los casos difíciles porque el modelo falla.

**Métricas:**
- Primaria: **AUROC** con predicciones out-of-fold.
- Comparación: **ΔAUROC = AUROC(fuente) − AUROC(baseline)** con **bootstrap pareado remuestreando pacientes completos** (no basta con reportar dos IC solapados). Reporta **tamaño de efecto + IC, no significancia binaria**; con n≈150-186 los IC serán anchos → considera **CV repetida (5×5)** para estabilizar el estimador.
- **Prevalencia de clase reportada** (sale del QC); AUPRC prominente si hay desbalance.
- Secundarias: AUPRC, sensibilidad, especificidad, balanced accuracy, Brier, F1 (umbral elegido solo en validación). **Brier/calibración NO se reclama aquí** — es la contribución de P2 (SPIE); si lo reportas, enmárcalo como proper scoring rule combinado y difiere el análisis a P2.

**Variabilidad de submuestreo:** subconjuntos anidados **10⊂25⊂50⊂100 %** (el codo bajo es donde vive el efecto). ≥3 semillas para 10/25/50 %; si no alcanza el tiempo, subconjunto anidado predefinido y **declara la curva como exploratoria**.

**Reproducibilidad:** manifiesto de folds versionado, log de exclusiones, config por corrida, commit, predicciones OOF, semillas, versiones de PyTorch/CUDA/pesos.

**Auditoría de datasets:** degrádala a **tabla de motivación** (justifica PlaTiF, documenta incompatibilidad anatomía/modalidad). No es una "segunda contribución" salvo que sea una revisión sistemática con protocolo — no lo es aquí.

---

## 5. Reconciliación honesta del conteo de papers

La crítica tiene razón y mis dos documentos se contradecían:
- El de "programa" decía **2-3 papers fuertes**.
- El "quincenal" proyectaba ~6 papers de 12 experimentos.

La cadencia quincenal era de **experimentos**, no de **papers** — pero la presenté de forma que invitaba a confundirlos. **Corrección: 3-4 trabajos portantes; los micro-experimentos se agregan.**

| # | Paper portante | Contenido | Sede |
|---|---|---|---|
| 1 | **ROPEC 2026** | Procedencia × eficiencia de etiquetas (ImageNet/RadImageNet/MURA/random) | ROPEC |
| 2 | **Percepción confiable** | Calibración + abstención (P2+P3 juntos, no separados); idealmente 2º dataset o validación externa | SPIE MI / ISBI |
| 3 | **Segmentación + incertidumbre** | nnU-Net + tu método + Dice/ASSD/HD95 + MC/ensemble (P4+P5 juntos). **Requiere etiquetas de fragmentos — PlaTiF solo no basta** (sus máscaras son tibia/región, no fragmentos) | ISBI / EMBC |
| 4 | **Principal integrado** | Percepción→SSM/MAP→SE(3)→propagación→hexápodo + validación sintética/fantoma | MICCAI o RA-L/IROS |

**P7 (Jacobiano vs Monte Carlo) NO es un paper de ICRA por sí solo.** Una linealización que coincide con Monte Carlo en perturbaciones pequeñas es validación interna, no contribución de main track. Para volverse paper necesita el modelo cinemático real del hexápodo, mapa SE(3)→struts, singularidades, límites mecánicos, first-order vs unscented vs Monte Carlo, perturbaciones realistas del registro, y validación en simulador/fantoma → eso es el paper de robótica (P7+P8 juntos) o parte del principal. ICRA 15-sep es irreal salvo que la cinemática y el simulador ya existan.

---

## 6. Señal competitiva para tu contribución PRINCIPAL

La **Fase 2** del preprint apunta explícitamente a un encoder **ViT-MAE** sobre un corpus mayor de TPF. Tu tesis usa MAE. Es decir: **ese grupo se mueve hacia tu terreno**. Tu diferenciación real: tú haces **CT + segmentación + propagación de incertidumbre + planeación de reducción con hexápodo**; ellos hacen radiografía + fenotipado. Distinto, pero:
1. Vigila a ese grupo (Elnakib/Al-Kabbany).
2. Refuerza el consejo del documento de puntos ciegos: **arranca el núcleo duro (SE(3)/incertidumbre) ya** y corre la prueba de 3 resultados sobre el principal antes de noviembre. La ventana de novedad de la parte de MAE se está cerrando; la de la propagación de incertidumbre en SE(3) para reducción sigue abierta y es más tuya.

---

## 7. Qué congelas hoy (Día 1 actualizado)

Reescribe el párrafo de hueco así (rellena al ejecutar):

> "La detección de fractura de platillo tibial en radiografía alcanza sensibilidad experta (van der Gaast 2025; Huo 2025), y trabajo reciente asume que el preentrenamiento radiológico (RadImageNet) supera a ImageNet en cohortes ortopédicas pequeñas (Elnakib 2026), pero ningún estudio aísla cómo la procedencia del preentrenamiento gobierna la eficiencia de etiquetas para esta tarea bajo evaluación agrupada por paciente. Presentamos una ablación controlada de [ImageNet/RadImageNet/MURA-SSL] sobre PlaTiF, con particiones por paciente y ΔAUROC pareado, caracterizando [hallazgo]."

Contribuciones (tres, honestas):
1. Primera ablación controlada de procedencia de preentrenamiento × eficiencia de etiquetas para detección de TPF, agrupada por paciente.
2. Prueba directa de la afirmación RadImageNet > ImageNet en este régimen (antes asumida, no ablacionada).
3. Protocolo reproducible (folds, exclusiones, predicciones OOF, ΔAUROC pareado) sobre un dataset público.
