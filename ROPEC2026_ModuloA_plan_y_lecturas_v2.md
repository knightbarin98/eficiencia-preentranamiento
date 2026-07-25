# ROPEC 2026 — Módulo A: plan de entrega y lecturas (v2, corregido)

**Fecha:** 12 jul 2026 · **Deadline:** ~24 jul 2026 (verificar en EasyChair) · **Sesión:** Workshop on Biomedical Applications · **Formato:** full paper IEEE ≤6 pág, oral, IEEE Xplore.
**Cambio v2:** el framing pasó de "transferencia cross-anatomía / primera SSL en TPF" a **ablación de procedencia de preentrenamiento × eficiencia de etiquetas**, tras el preprint Elnakib 2026 y la evidencia de que la detección de TPF ya está resuelta (van der Gaast 2025, Huo 2025).
**Ajuste 14 jul 2026:** análisis primario = **interacción procedencia×fracción**; random como **curva de referencia** (no solo sanity); punto al **10%**; agregación **vista→rodilla** congelada; prevalencia/AUPRC desde QC; **frontera Brier→P2**; conclusión accionable; **go/no-go MURA día 0**.

---

## 0. Veredicto sobre las conversaciones iniciales (sin cambios)

| Conversación | Estatus | Uso |
|---|---|---|
| Doc 1 (auditoría rigurosa) | **Confiable** | Esqueleto. |
| Doc 2 (WIP genérico) | Con cuidado | Descartar "resultados esperados". |
| Doc 3 (arquetipos) | Solo la idea | **Citas probablemente inventadas.** |

**Regla de integridad:** nada de gráficas proyectadas ni "resultados esperados" numéricos. Un resultado pequeño real > uno grande inventado.

---

## 1. Decisión de alcance (v2)

**Pregunta primaria (reemplaza la anterior):**
> ¿Cómo gobierna la **procedencia del preentrenamiento** —ImageNet, RadImageNet y radiografía MSK de otra anatomía (MURA-SSL)— la **eficiencia de etiquetas** para detectar fractura de platillo tibial, bajo evaluación **agrupada por paciente** y comparación **pareada**?

- **Dataset principal:** PlaTiF (421 radiografías / 186 pacientes, Schatzker, múltiples vistas).
- **Backbone único:** **ResNet-50** (es el que tiene pesos RadImageNet).
- **Comparadores:** ImageNet · RadImageNet · MURA-SSL (si alcanza) · **random como curva de referencia** (no solo sanity: da la eficiencia de datos "desde cero", que vuelve concreta la afirmación — p. ej. "RadImageNet al 25% iguala a random al 100%").
- **Tarea:** **presencia vs. ausencia de fractura de platillo** (no "fractura vs normal": algunos casos con platillo normal tienen otras fracturas).
- **Unidad:** folds y remuestreo = **paciente**; predicción = **rodilla/lado**; todas las vistas/rodillas de un paciente en el mismo fold. **Agregación vista→rodilla congelada** (p. ej. media de logits por rodilla; documentar la regla antes de ver resultados).
- **Análisis primario = interacción `procedencia × fracción de etiquetas`** (¿la brecha RadImageNet−ImageNet se ensancha al bajar las etiquetas?), no dos curvas comparadas a ojo.
- **Métrica primaria:** AUROC out-of-fold. **Comparación:** ΔAUROC con bootstrap pareado remuestreando pacientes; reportar **tamaño de efecto + IC, no significancia binaria**. Reportar **prevalencia de clase** y AUPRC (n≈150-186 → IC anchos; considerar CV repetida 5×5).
- **Auditoría de datasets:** **tabla de motivación**, no "segunda contribución".
- **Fuera de este paper:** MAE-ViT (va a la tesis), FPN, nnU-Net, segmentación, MC Dropout, GPMM, SE(3), hexápodo.

**Por qué:** detección de TPF y "primera SSL en TPF" ya están ocupadas. El hueco real es la **ablación controlada de procedencia × eficiencia** que el trabajo reciente asumió pero no ejecutó.

---

## 2. Lecturas directamente relacionadas (v2, verificadas)

### 2.1 Vecino directo — imprescindible (nuevo)
**Elnakib, Saad & Al-Kabbany (2026).** *Phenotyping Tibial Plateau Fractures via Self-Supervised Learning.* arXiv:2606.17295.
- Usa PlaTiF + RadImageNet-ResNet-50 + SimCLR para **fenotipado no supervisado**. **Afirma** RadImageNet > ImageNet en cohortes pequeñas **sin ablación controlada**. Tu paper testea exactamente eso. Cítalo y construye sobre su limpieza de 8 pasos.

### 2.2 Detección de TPF ya resuelta — cítalos para acotar tu contribución (nuevo)
- **van der Gaast et al. (Knee 54:81-89, 2025):** GoogleNet+ResNet, 1506 radiografías/753 pacientes, detección + Schatzker; sensibilidad 92.7%.
- **Huo et al. (Radiology Advances 2(3):umaf020, 2025):** MobileNetV3-YOLOv8 multicéntrico **con validación externa**.
- → Establecen que la *detección* está hecha; tu contribución es la *ablación de procedencia × eficiencia*, no detectar.

### 2.3 Anclas de preentrenamiento (nuevo/actualizado)
- **RadImageNet — Mei et al. (Radiology: AI 4(5):e210315, 2022).** 1.35M imágenes radiológicas; supera a ImageNet en tareas MSK pequeñas por 4-9% AUC. Justifica el brazo RadImageNet. Pesos ResNet-50 **descargables (CC BY 4.0)**; no hay ViT.
- **Raghu et al. (2019), Transfusion** (arXiv:1902.07208): la transferencia ImageNet aporta poco en imagen médica → justifica cuestionar ImageNet.
- **Azizi et al. (2021),** *Big Self-Supervised Models Advance Medical Image Classification* (ICCV): justifica el brazo SSL.

### 2.4 Molde metodológico (sin cambios)
- **Hinterwimmer et al. (2022),** *From SSL to transfer learning with MSK radiographs.*
- **Thorat et al. (2024),** detección de fractura de muñeca con SSL + ResNet.
- **Spahr et al.** (arXiv:2102.09895): MURA como fuente de SSL.

### 2.5 Dominio / apoyo (sin cambios)
- **Yue et al. (2025)** arXiv:2502.02862 — MAE + platillo tibial en **CT+segmentación** (distinto de tu radiografía+detección+procedencia).
- **PlaTiF (2026)** Sci Data, Zenodo 18007397 — tu dataset.
- **FracAtlas** (Abedeen 2023) — secundario/exploratorio; "leg" ≠ tibia.
- **MURA** (Rajpurkar 2017) — fuente de SSL, extremidad superior.

---

## 3. Guías para subir el rigor (sin cambios)
- **CLAIM 2024** (checklist de reporte).
- **Metrics Reloaded** (Nature Methods 2024): AUROC + AUPRC + sens/esp + balanced acc + Brier.
- **nnU-Net Revisited** (Isensee 2024): baselines bajo mismo presupuesto (aplica a papers futuros).

---

## 4. Correcciones críticas al texto (v2)
- [ ] **Comparadores:** ImageNet / RadImageNet / MURA-SSL / random, **mismo ResNet-50**, mismos folds/augs/optimizador/épocas/tuning.
- [ ] **Tarea:** presencia/ausencia de fractura de platillo.
- [ ] **Unidad:** paciente (folds/remuestreo), rodilla/lado (predicción).
- [ ] **CV:** 5 folds externos agrupados por paciente; dentro del 80% una validación para early stopping/umbral.
- [ ] **Limpieza = método:** criterios QC congelados antes de ver resultados; diagrama 186 → exclusiones → N pacientes, M rodillas.
- [ ] **Comparación:** ΔAUROC con bootstrap pareado por paciente (no dos IC solapados).
- [ ] **Análisis primario:** interacción procedencia × fracción de etiquetas (no solo efectos principales); random como curva de referencia.
- [ ] **Agregación vista→rodilla** congelada y documentada (p. ej. media de logits).
- [ ] **Prevalencia de clase** reportada (sale del QC); AUPRC prominente si hay desbalance; tamaño de efecto + IC, no significancia binaria.
- [ ] **Frontera con P2:** no reclamar calibración aquí; si reportas Brier, enmárcalo como proper scoring rule combinado y difiere el análisis de calibración a P2 (SPIE).
- [ ] **Conclusión accionable:** declarar qué inicialización adopta el pipeline de la tesis y por qué.
- [ ] **MAE fuera** de este paper.

---

## 5. Plan por días (v2, comparadores corregidos)

**12-13 jul — Alcance y datos**
- [ ] Congelar pregunta (procedencia × eficiencia), título, 3 contribuciones.
- [ ] Descargar PlaTiF (Zenodo 18007397); descargar pesos RadImageNet ResNet-50. **Go/no-go MURA (día 0):** ¿hay pesos SSL de MURA sobre ResNet-50 descargables? Si no, MURA fuera — no se entrena SSL desde cero (se come el cronograma).
- [ ] Confirmar deadline/plantilla en EasyChair.

**14-16 jul — Auditoría + limpieza (método)**
- [ ] QC de PlaTiF con criterios congelados; diagrama de exclusiones; conteos por paciente/rodilla/clase.
- [ ] Tabla de motivación de datasets.

**17-20 jul — Ablación (núcleo)**
- [ ] Folds agrupados por paciente, semilla fija, subconjuntos anidados **10⊂25⊂50⊂100%** (el codo bajo es donde vive el efecto de procedencia; el 100% es lo menos informativo).
- [ ] Entrenar ResNet-50 con ImageNet, RadImageNet (+ random sanity); MURA-SSL si alcanza.
- [ ] AUROC OOF + ΔAUROC pareado (bootstrap por paciente); secundarias sens/esp/AUPRC/balanced/Brier/F1.
- [ ] **Checkpoint 18-19:** ¿reproducible? Si no → contingencia (ImageNet vs RadImageNet a 100%, sin curva ni MURA).

**21 jul — Figuras/errores + FREEZE**
- [ ] Tabla (procedencia × métricas) + curva de eficiencia. Análisis de fallos (FN, calidad, Schatzker).

**22-23 jul — Escritura IEEE**
- [ ] Redactar; pasar CLAIM; abstract con N pacientes, comparadores, ΔAUROC, conclusión limitada.

**24 jul — Asesor + envío.**

---

## 6. Lista de "NO afirmar" (v2)
Evita: superioridad clínica · superar a médicos · **"primera detección/SSL de TPF"** (ya existe) · segmentación de fragmentos · utilidad para reducción/hexápodo · validar MAE para toda la tesis.
Sí puedes: "caracterizamos el efecto de la procedencia del preentrenamiento sobre la eficiencia de etiquetas… con particiones por paciente y comparaciones pareadas."

---

### Estructura de 6 páginas (v2)
1. **Introduction** (¾): detección ya resuelta (van der Gaast/Huo) + RadImageNet asumido sin ablación (Elnakib) → tu pregunta + 3 contribuciones.
2. **Related Work** (½): detección TPF, SSL/procedencia de preentrenamiento, diferencia exacta vs Elnakib y Yue.
3. **Materials & Methods** (1½): cohorte por paciente, limpieza, comparadores, folds, métricas, ΔAUROC pareado.
4. **Results** (1½): tabla procedencia × métricas + curva de eficiencia.
5. **Discussion** (¾): límites (n≈186 pre-limpieza, una institución, solo platillo, sin prueba externa).
6. **Conclusion** (¼): limitada a lo demostrado + **decisión accionable** (qué inicialización adopta el pipeline de la tesis y por qué).
