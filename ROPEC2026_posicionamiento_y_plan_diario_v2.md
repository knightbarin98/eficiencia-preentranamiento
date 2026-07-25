# ROPEC — Posicionamiento (v2) + plan día por día

**Cambio v2:** el hueco ya no es "transferencia cross-anatomía"; es **procedencia de preentrenamiento × eficiencia de etiquetas** (tras Elnakib 2026 y la detección de TPF ya resuelta).

**Ajuste 14 jul 2026:** análisis primario = **interacción procedencia×fracción**; random como **curva de referencia**; curva **10/25/50/100%**; **go/no-go MURA día 0**; métricas con tamaño de efecto; **Brier→P2**.

---

## Parte 1 — Tu posición en el estado del arte (v2)

**Principio:** publicable = hueco de la literatura, no novedad desde cero.

### Lo que ya está ocupado (no lo reclames)
- **Detección de TPF en radiografía:** resuelta y con validación externa — van der Gaast et al. (Knee 2025, sens. 92.7%), Huo et al. (Radiology Advances 2025, multicéntrico + externa), Liu (RetinaNet ~a la par de médicos).
- **SSL sobre PlaTiF:** ocupada por Elnakib et al. (arXiv:2606.17295, jun 2026): RadImageNet-ResNet-50 + SimCLR para **fenotipado no supervisado**.

### Tu celda abierta
> Ablación **controlada** de procedencia de preentrenamiento (ImageNet / RadImageNet / MURA-SSL / random) sobre la **eficiencia de etiquetas** para detección de TPF, **agrupada por paciente**, con **ΔAUROC pareado**.

Sigue abierta porque: van der Gaast/Huo no ablacionan procedencia ni eficiencia; Elnakib **afirma** RadImageNet > ImageNet **sin ablación controlada**. Tú provees esa prueba.

### Andamiaje de citas
- **Anclas:** Raghu 2019 (Transfusion, arXiv:1902.07208); Azizi 2021 (SSL médico, ICCV); **Mei 2022 (RadImageNet, Radiology: AI e210315)**.
- **Detección resuelta:** van der Gaast 2025 (Knee); Huo 2025 (Radiology Advances).
- **Vecino directo:** Elnakib 2026 (arXiv:2606.17295).
- **Molde/MSK:** Hinterwimmer 2022; Thorat 2024; Spahr (arXiv:2102.09895, MURA-SSL).
- **Dominio:** Yue 2025 (arXiv:2502.02862, CT+segmentación); PlaTiF (Zenodo 18007397).

### Riesgo y mitigación
"RadImageNet > ImageNet" podría verse asumido → tu valor es hacerlo **controlado + eficiencia de etiquetas + pareado**. Sé explícito: "no afirmamos superioridad clínica; caracterizamos el efecto de la procedencia bajo escasez".

---

## Parte 2 — Prueba de novedad de 3 resultados (sin cambios, reutilizable)

1. Busca el reclamo exacto (Scholar/Semantic Scholar/arXiv, 2-3 redacciones).
2. Clasifica:
   - (a) Respondido en tu escenario exacto → no es paper; pivotea (régimen/población/incertidumbre).
   - **(b) Respondido en escenarios adyacentes, no el tuyo → tu hueco.** ← apunta aquí. (ROPEC v2 vive aquí: procedencia se estudió en general, no en TPF-detección-eficiencia-pareado.)
   - (c) Nadie lo preguntó → revisa surveys por si fue descartado.
3. Mapea con surveys. 4. Encadena "cited by" sobre el vecino (Elnakib, Yue).

---

## Parte 3 — Plan día por día (12-24 jul, v2)

**Día 0 · Sáb 12 — Datos y logística**
- [ ] PlaTiF (Zenodo 18007397) + **pesos RadImageNet ResNet-50** (CC BY 4.0). → en disco.
- [ ] **Go/no-go MURA:** ¿hay pesos SSL de MURA sobre ResNet-50 descargables? Si no, MURA fuera (no entrenar SSL desde cero).
- [ ] Deadline/plantilla en EasyChair.

**Día 1 · Dom 13 — Posicionamiento (cierra framing ANTES de codear)**
- [ ] Leer Raghu (arXiv:1902.07208) + hojear Mei RadImageNet (e210315). → párrafo "procedencia importa".
- [ ] Leer Elnakib (arXiv:2606.17295): su afirmación RadImageNet>ImageNet sin ablación = tu hueco; su limpieza de 8 pasos = tu base.
- [ ] Hojear van der Gaast + Huo. → párrafo "detección ya resuelta".
- [ ] Correr prueba de 3 resultados. → confirmar celda (b).
- [ ] Redactar párrafo de hueco + 3 contribuciones. → **framing congelado.**

**Día 2 · Lun 14 — Auditoría + limpieza (método)**
- [ ] QC con criterios congelados (anatomía, lado, hardware, vista, resolución, duplicados); diagrama 186 → exclusiones → N/M. *Recurso:* protocolo de 8 pasos de Elnakib como referencia (adáptalo a detección).
- [ ] Tabla de motivación de datasets.

**Día 3 · Mar 15 — Harness sobre datos juguete**
- [ ] Training loop + métricas + split en MedMNIST/CIFAR. *Recurso:* `timm`, `sklearn`.

**Día 4 · Mié 16 — PlaTiF + partición sin fuga**
- [ ] `StratifiedGroupKFold` por `patient_id`; todas las rodillas/vistas de un paciente juntas.

**Día 5 · Jue 17 — Ablación de procedencia**
- [ ] ResNet-50 con **ImageNet vs RadImageNet** (+ random sanity), 5 folds. *Nota:* RadImageNet no es flag de `timm`; carga los pesos descargados en un ResNet-50. → `fold_k_preds.csv`.

**Día 6 · Vie 18 — Métricas + checkpoint**
- [ ] AUROC OOF + **ΔAUROC pareado** (bootstrap por paciente); **tamaño de efecto + IC, no significancia**; prevalencia de clase; sens/esp/AUPRC/balanced/(Brier → difiere calibración a P2).
- [ ] **CHECKPOINT:** ¿reproducible? Si no → ImageNet vs RadImageNet a 100%, sin curva.

**Día 7 · Sáb 19 — Eficiencia de etiquetas (+ MURA-SSL si pasó el go/no-go)**
- [ ] Curva **10/25/50/100%** (subconjuntos anidados, mismos para todos). ≥3 semillas para 10/25/50% o declara exploratoria. **Análisis primario = interacción procedencia×fracción** + **random como curva de referencia**.
- [ ] *Stretch:* brazo MURA-SSL **solo si hubo pesos descargables (día 0)**. *Recurso:* Spahr (arXiv:2102.09895). Si no converge, fuera.

**Día 8 · Dom 20 — Figuras/errores + FREEZE**
- [ ] Tabla (procedencia × métricas) + curva. Fallos (FN, calidad, Schatzker). *Recurso:* Metrics Reloaded.

**Día 9 · Lun 21 — Escritura: Methods + Results**
- [ ] Plantilla IEEE. *Recurso:* CLAIM 2024; García-Peraza-Herrera (arXiv:2303.08252) como molde.

**Día 10 · Mar 22 — Intro + Related Work + Discussion**
- [ ] Intro: detección resuelta (van der Gaast/Huo) + RadImageNet asumido (Elnakib) → tu pregunta. Related Work: diferencia exacta vs Elnakib y Yue.

**Día 11 · Mié 23 — Abstract + CLAIM + blindaje**
- [ ] Abstract con N pacientes, comparadores, ΔAUROC, conclusión limitada. Lista "no afirmar".

**Día 12 · Jue 24 — Asesor + envío.**

---

### Nota de secuencia (sin cambios)
Día 1 (posicionamiento) antes de todo código: si el framing no cierra en (b), mejor saberlo el domingo.
