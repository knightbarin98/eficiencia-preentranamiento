# ROPEC 2026 — Encuadre y esqueleto del paper (para redactar)

**Estado (22 jul 2026):** experimentos terminados y **validados** (no es bug: RadImageNet cargó 318 tensores; random se queda en el azar y los preentrenados suben → el pipeline aprende señal real). Resultado = **nulo sobre procedencia**, que es un aporte honesto. Deadline efectivo: **23 jul**. Este doc es tu andamiaje: encuadre, números reales para citar tal cual, abstract semi-armado, esqueleto de intro con dónde entra cada lectura, y la lista "NO afirmar". El texto del paper va en **inglés**; las notas de guía van en español y marcadas.

---

## 0. El encuadre en una frase (léelo antes de escribir nada)

Tu hipótesis original (RadImageNet > ImageNet) **se refuta** — y eso **es** el paper, no un fracaso. Elnakib 2026 *asumió* esa ventaja sin ablación controlada; tú la probaste y no existe. El claim honesto:

> *In a small-cohort regime for tibial plateau fracture detection, pretraining provenance (ImageNet vs. RadImageNet) does not govern label efficiency: radiology-specific pretraining confers no advantage over generic ImageNet, contrary to assumptions in recent work. Both exceed random initialization only once labels are sufficient.*

**No vendes** detección (ya resuelta) ni superioridad. Vendes una **ablación controlada que corrige una suposición del campo** + una caracterización honesta de dónde ayuda el preentrenamiento.

---

## 1. Hallazgos reales (números para citar exactamente)

**Cohorte / QC (de `outputs/qc_flow.json`):**
- 186 pacientes · 190 rodillas · 421 vistas → tras QC: **419 vistas** (2 excluidas por concordancia de máscara), 190 rodillas, 186 pacientes.
- Prevalencia a nivel rodilla: **128 pos / 62 neg = 0.674 positivo** (tarea desbalanceada; baseline AUPRC ≈ 0.67).

**Rendimiento @100% de etiquetas (5 semillas, `multiseed_summary.json`):**

| Fuente | AUROC (media ± sd) | AUPRC |
|---|---|---|
| RadImageNet | **0.679 ± 0.038** (máx por semilla 0.716) | 0.800 |
| ImageNet | **0.664 ± 0.028** | 0.784 |
| Random | **0.526 ± 0.020** | 0.698 |

**ΔAUROC pareado @100% (bootstrap 10 000, remuestreo por paciente):**
- RadImageNet − ImageNet = **+0.015**, IC95% **[−0.109, +0.136]**, p=0.80 → **nulo** (indistinguibles).
- RadImageNet − Random = **+0.154**, IC95% **[+0.009, +0.286]** → preentrenar ayuda.
- ImageNet − Random = **+0.139**, IC95% **[+0.011, +0.267]** → preentrenar ayuda.

**Interacción procedencia × fracción (RadImageNet − ImageNet por fracción, `curve_summary.json`):**
- 10%: −0.043 · 25%: −0.003 · 50%: −0.013 · 100%: +0.025 → **todos los IC cruzan 0**: la procedencia **nunca** separa.

**Curva de eficiencia (AUROC media, 3 semillas):**

| Fracción | RadImageNet | ImageNet | Random |
|---|---|---|---|
| 10% | 0.506 | 0.550 | 0.511 |
| 25% | 0.545 | 0.547 | 0.535 |
| 50% | 0.621 | 0.634 | 0.544 |
| 100% | 0.688 | 0.663 | 0.526 |

→ A **≤25% todo está en el azar** (ni el preentrenamiento ayuda); el beneficio sobre random **emerge solo con más etiquetas** (~0.08 a 50%, ~0.15 a 100%, IC excluye 0 solo a 100%).

> ⚠️ **Arreglo de integridad:** NO uses el bloque `data_equivalence` ("RadImageNet@25% ≈ random@100%"). Es engañoso: random@100% está en el azar (0.526), así que "igualarlo" es un listón trivial. Bórralo del texto.

---

## 2. Contribuciones (tres, honestas)

1. **Primera ablación controlada** de procedencia de preentrenamiento (ImageNet/RadImageNet/random) × eficiencia de etiquetas para detección de TPF, agrupada por paciente y con ΔAUROC pareado.
2. **Prueba directa —y refutación— de la ventaja asumida RadImageNet > ImageNet** en este régimen: no la hay a ninguna fracción de etiquetas.
3. **Protocolo reproducible** (manifiesto de folds, exclusiones QC, predicciones OOF, configs, semillas, versiones) sobre un dataset público.

---

## 3. Abstract — semi-armado (rellena solo los `[ ]`)

> Recent work assumes radiology-specific pretraining (RadImageNet) outperforms generic ImageNet pretraining in small orthopedic cohorts, but this has not been tested under controlled, patient-grouped evaluation. We ablate pretraining provenance — ImageNet, RadImageNet, and random initialization — for tibial plateau fracture detection on PlaTiF (419 radiographs from 186 patients; 190 knees, 67% fracture-positive), using a single fixed ResNet-50 backbone with identical folds, optimizer, and budget across initializations. Evaluation is patient-grouped (StratifiedGroupKFold); predictions are made per knee by averaging view logits; and sources are compared by paired ΔAUROC bootstrapped over patients. Across label fractions (10/25/50/100%) and 5 seeds, RadImageNet and ImageNet are statistically indistinguishable (ΔAUROC = +0.015, 95% CI [−0.11, +0.14]); provenance does not affect label efficiency at any budget. Both pretrained sources exceed random initialization only once labels are sufficient (ΔAUROC ≈ +0.15 at 100%, CI excludes zero; no benefit at ≤25%, where all initializations remain near chance, AUROC ≈ 0.5). Maximum AUROC is 0.68. Our controlled result refutes the assumed advantage of radiological pretraining for this task and delimits where pretraining helps, motivating in-domain self-supervised representations. We release folds, out-of-fold predictions, and configurations. [1 frase clínica de contexto extraída de una lectura, si sobra espacio.]

*(≈210 palabras. Recórtalo a la longitud del abstract IEEE si hace falta quitando la última frase de release.)*

---

## 4. Esqueleto de Introducción (¾ pág) — con SLOTS de lecturas

Cada párrafo trae [LECTURA→] marcando qué debes extraer al leer mañana.

**P1 — La detección de TPF ya está resuelta (no es tu contribución).**
- Idea: la detección de fractura de platillo en radiografía alcanza sensibilidad experta.
- [LECTURA→ van der Gaast 2025 (Knee): sens 92.7%; Huo 2025 (Radiology Advances): multicéntrico + validación externa. Extrae 1 cifra de cada uno.]

**P2 — La suposición sobre procedencia del preentrenamiento.**
- Idea: en cohortes médicas pequeñas se asume que el preentrenamiento de dominio (RadImageNet) supera a ImageNet; trabajo reciente en TPF lo adopta sin ablación controlada. Pero la evidencia de que el transfer de ImageNet aporta poco en imagen médica deja la pregunta abierta.
- [LECTURA→ Mei 2022 (RadImageNet, Radiology:AI e210315): supera a ImageNet 4–9% en MSK pequeño. Elnakib 2026 (arXiv:2606.17295): usa RadImageNet-ResNet50 + SimCLR sobre PlaTiF, **asume** la ventaja. Raghu 2019 Transfusion (arXiv:1902.07208): ImageNet transfer aporta poco → **respalda tu nulo**.]

**P3 — El hueco.**
- Idea: ningún estudio aísla cómo la procedencia gobierna la **eficiencia de etiquetas** para detección de TPF bajo evaluación agrupada por paciente.
- [LECTURA→ Elnakib como vecino directo: su tarea es fenotipado no supervisado, no detección ni ablación → ese es exactamente tu hueco.]

**P4 — Qué hacemos + contribuciones.**
- Idea: presentamos una ablación controlada (ImageNet/RadImageNet/random), agrupada por paciente, con ΔAUROC pareado y curva de eficiencia; enumera las 3 contribuciones de §2.
- [LECTURA→ PlaTiF (Zenodo 18007397) como dataset; CLAIM 2024 + Metrics Reloaded como marco de reporte.]

---

## 5. Related Work (½ pág) — la diferencia EXACTA

- **Detección de TPF:** van der Gaast, Huo, Liu → resuelta; tú no compites ahí.
- **Procedencia / SSL de preentrenamiento:** Mei (RadImageNet), Raghu (Transfusion), Azizi 2021 (SSL médico); molde MSK: Hinterwimmer 2022, Thorat 2024, Spahr (MURA-SSL).
- **Vecino directo — di la diferencia en una frase cada uno:**
  - *Elnakib 2026:* fenotipado **no supervisado** (UMAP+k-means), **asume** RadImageNet>ImageNet sin ablación → tú haces la ablación supervisada controlada.
  - *Yue 2025 (arXiv:2502.02862):* MAE + platillo tibial en **CT + segmentación** → distinta modalidad y tarea que tu radiografía + detección + procedencia.

---

## 6. Mapa lectura → qué extraer → sección (tu lista de mañana)

| Lectura | Qué extraer | Va a |
|---|---|---|
| van der Gaast 2025 (Knee) | sens 92.7%, tamaño cohorte | Intro P1, Related |
| Huo 2025 (Radiology Advances) | multicéntrico + validación externa | Intro P1, Related |
| Mei 2022 (RadImageNet) | +4–9% MSK pequeño; pesos ResNet-50 | Intro P2, Related |
| Elnakib 2026 (arXiv:2606.17295) | usa PlaTiF+RadImageNet+SimCLR; asume ventaja; fenotipado | Intro P2/P3, Related (diferencia) |
| Raghu 2019 Transfusion | transfer ImageNet aporta poco | Intro P2 + **Discussion** (respalda nulo) |
| Azizi 2021 (SSL médico) | justifica SSL/MAE | Related + Future work |
| Yue 2025 | MAE+TPF en CT+seg | Related (diferencia) |
| PlaTiF (Zenodo 18007397) | 421 rx / 186 pac; una institución; 2D | Methods, Limitations |
| CLAIM 2024 / Metrics Reloaded | checklist de reporte, AUROC+AUPRC | Methods |
| **Moldes de escritura** (no de dominio): García-Peraza-Herrera ODSI-DB (arXiv:2303.08252), Liu plafond tibial (arXiv:2102.11684), Ghose 2021, Liu masked sinogram (arXiv:2209.01356) | estructura de "preliminary study": comparación controlada, CV, código liberado, cero superioridad clínica | plantilla general del paper |

---

## 7. Lista "NO afirmar" (blindaje)

Evita: superioridad clínica · superar a médicos · "primera detección/SSL de TPF" (ya existe) · **que RadImageNet sea "inútil"** (solo mostraste este régimen/tarea/dataset) · calibración (es P2/SPIE — no reclames Brier aquí) · segmentación · utilidad para reducción/hexápodo · validar MAE para la tesis. **Cero resultados proyectados o "esperados".**

Sí puedes: "caracterizamos el efecto de la procedencia sobre la eficiencia de etiquetas, con particiones por paciente y comparación pareada; la procedencia no gobierna la eficiencia en este régimen."

---

## 8. Discussion + Future Work — la conexión con MAE (½ pág)

- **Interpretación honesta:** el preentrenamiento ayuda solo con etiquetas suficientes; la procedencia no separa. A ≤25% **nada supera el azar** → a pocas etiquetas el cuello **no es la representación, es el dato/tarea de fine-tuning**. (Alinéalo con Raghu/Transfusion.)
- **Límites (sé explícito):** AUROC absoluto modesto (~0.68), una institución, radiografía 2D, n≈186, sin validación externa; no es un detector desplegable.
- **Future work → tu tesis (MAE):** el nulo motiva SSL **en-dominio** (MAE sobre radiografía de rodilla/MSK) como la dirección que podría sacar del azar el régimen de pocas etiquetas donde ImageNet/RadImageNet no pueden. **Criterio de éxito para MAE:** ¿supera el azar a 10–25%? ¿supera 0.68 a 100%? Si no, el límite es el dato, no la representación. (No sobrevendas MAE aquí; es *future work*.)

---

## 9. Estructura 6 páginas IEEE (checklist de armado)

1. **Introduction** (¾) — §4 de este doc.
2. **Related Work** (½) — §5.
3. **Materials & Methods** (1½) — cohorte por paciente, QC (diagrama 186→419 vistas), comparadores, ResNet-50, folds `StratifiedGroupKFold`, agregación media de logits por rodilla, ΔAUROC pareado. Reusa `METHODS_as_run.md`.
4. **Results** (1½) — tabla @100% (§1) + figura de curva/interacción (`outputs/figures/fig_efficiency_curve.pdf`) + forest ΔAUROC (`fig_forest_delta100.pdf`).
5. **Discussion** (¾) — §8.
6. **Conclusion** (¼) — limitada a lo demostrado + **decisión accionable**: la procedencia no justifica preferir RadImageNet aquí; el pipeline de la tesis puede usar ImageNet y el foco de eficiencia se traslada a SSL en-dominio.

> **Recuerda:** al terminar, pasa el checklist CLAIM 2024 y confirma que el abstract trae N pacientes, comparadores, ΔAUROC con IC y conclusión limitada.
