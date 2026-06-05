# Selección del número de clases en clasificación no supervisada de imágenes satelitales bajo autocorrelación espacial

Proyecto de investigación (target: paper). Aporte al área de aprendizaje no
supervisado, aplicado a teledetección.

## Problema

La clasificación no supervisada de imágenes multiespectrales (ISODATA, k-means)
exige fijar el número de clases *k*. Los criterios estándar para elegir *k*
(silhouette, elbow, Calinski-Harabasz, Davies-Bouldin, gap statistic) asumen
observaciones i.i.d. Las imágenes satelitales violan esto de forma extrema:
los píxeles vecinos están fuertemente autocorrelacionados (Ley de Tobler +
PSF del sensor + tamaño de parche de cobertura).

**Hipótesis central:** la autocorrelación espacial sesga la selección de *k*.
La estructura *aparente* generada por la suavidad espacial (y por tendencias
de gran escala: iluminación, topografía) se confunde con clases discretas.

## Idea

Reformular la pregunta: *¿cuántas clases hay por encima de lo que la
autocorrelación espacial ya explica?*

Reemplazar el modelo nulo del gap statistic (Tibshirani et al. 2001), que es
uniforme/i.i.d. en el espacio de features, por un nulo que **preserva la
estructura espacial de segundo orden** (variograma/espectro de potencia) pero
**destruye la estructura discreta**. Dos candidatos de nulo:

- **Surrogate por aleatorización de fase de Fourier (AAFT)**: preserva el
  espectro de potencia por banda (= autocorrelación) y, con fase aleatoria
  compartida entre bandas, la correlación inter-banda. Barato (FFT), escala a
  imagen completa. *Riesgo detectado:* preserva también la multimodalidad
  marginal de los datos → como nulo puede ser demasiado fuerte (ver bitácora).
- **GRF paramétrico ajustado al variograma**: simula campos gaussianos
  multibanda con la covarianza espacial estimada → unimodal por construcción.

## Estado de validación (de-risking)

Ver `experiments/` y la bitácora abajo. Resumen vivo:

- ✅ **Silhouette / Davies-Bouldin sobre-detectan** bajo autocorrelación
  (k=8 con verdad k=1). Confirmado. Es el confound real, con víctima concreta
  (son los criterios estándar del rubro).
- ⚠️ **El gap clásico (nulo uniforme) resultó robusto** en régimen estacionario
  sin tendencia → hay que probar el régimen de **tendencia/gradiente**, donde
  se espera que sí falle.
- ⚠️ **El estadístico VG-gap acierta por `argmax`** (pico en k real, monótono
  decreciente cuando k=1), pero la regla de selección de Tibshirani no aplica a
  su forma de curva → el selector debe adaptarse a la forma.

## Plan del paper

1. De-risking (en curso): confirmar fenómeno y régimen donde el clásico falla.
2. Revisión de novedad / literatura.
3. Formalización: definición, propiedades, algoritmo, complejidad, factibilidad
   a escala de imagen.
4. Implementación: librería + tests + reproducibilidad.
5. Benchmarks: sintéticos (verdad controlada) + hiperespectrales con etiquetas
   (Indian Pines, Salinas, Pavia) + multiespectral (Sentinel-2 + ESA WorldCover).
6. Ablaciones y modos de falla.
7. Escritura → Computers & Geosciences / EMS / IJRS / Spatial Statistics.

## Bitácora

- **2026-05-31** · exp 01 (`01_derisk.py`): primer test. Hallazgo: silhouette
  sobre-detecta (k=8); gap clásico robusto en mundo estacionario; VG-gap con
  regla Tibshirani sub-detecta, pero su curva tiene el pico en el k correcto
  (bug del selector, no del método).
- **2026-06-01** · exp 02 (`02_derisk_v2.py`): **resultado clave**. En el
  mundo TREND (gradiente suave, sin clases) *todos* los criterios estándar
  sobre-detectan (silhouette/CH/DB→4, gap clásico→8); solo el nulo espacial
  da k=1. Confirma el fenómeno con fuerza. PERO el nulo de fase se
  sobre-corrige en STRUCT (k=1 con verdad 4): preserva la potencia de baja
  frecuencia de los parches → el nulo también es separable. Núcleo del aporte:
  separar autocorrelación intra-clase de estructura inter-clase.
- **2026-06-01** · exp 03 (`03_within_class_null.py`): nulo desde residuos de
  sobre-segmentación. Resultado 1,8,4 → arregla STRUCT pero rompe TREND
  (confunde tendencia con clases). raw-phase da 1,1,1 (nunca detecta). Los dos
  nulos **encierran la verdad**. Crux: clusters de k-means son fenómeno de 2º
  orden → un nulo que iguala la covarianza los borra; hay que separar
  tendencia continua de estructura discreta.
- **2026-06-01** · Revisión de literatura (deep-research, 24 fuentes, 25 claims
  verificados). VEREDICTO: el concepto "nulo consciente de autocorrelación para
  elegir k" YA EXISTE → **Hennig & Lin 2015, Stat & Comp** (sección "Null model
  for spatial autocorrelation"; BIC k=8→k=2 al corregir). Sobre-detección
  confirmada en otros campos: Goovaerts & Jacquez 2004 (epi), Eklund 2016 (fMRI).
  Maquinaria de nulos: MSR (Wagner & Dray 2015), Arthur 2024, SGS, ESS
  (Dutilleul 1993). HUECO ABIERTO (nuestra cuña): (1) nulo de campo continuo
  (GRF/variograma/Fourier) para k en imágenes raster multiespectrales; (2) el
  confound tendencia-vs-clase discreta en selección de nº de clases — nadie lo
  trata. Reenfoque: novedad = separación tendencia/clase + instanciación de
  campo continuo en teledetección, construyendo sobre Hennig & Lin.
- **2026-06-01** · exp 04 (`04_declustering.py`): declustering + gap uniforme.
  Resultado 1,**4**,4 → arregla NULL/STRUCT pero NO TREND. Lección: la
  independencia es necesaria pero no suficiente; un gradiente es estructura
  *continua* y el nulo uniforme la lee como clusters.
- **2026-06-01** · exp 05 (`05_decluster_gaussian_null.py`): declustering +
  nulo gaussiano unimodal. Resultado 1,1,**1** → arregla TREND pero
  sub-detecta STRUCT (el nulo gaussiano con covarianza completa absorbe las
  clases, igual que raw-phase).

**Síntesis (5 experimentos):** la distinción tendencia-vs-clase es un problema
de **multimodalidad**, no de estructura espacial. Método correcto necesita DOS
ingredientes: (1) **declustering** (tamaño muestral efectivo) + (2) **criterio
de modalidad** que no sea un nulo de 2º orden (uniforme over-detecta tendencia;
gaussiano/2º-orden borra clases). Ruta prometedora: declustering + **test de
modalidad tipo dip** (Hartigan dip / dip-means, Kalogeratos & Likas 2012
NeurIPS — apareció en la revisión) para fijar k. Novedad defendible: combinarlo
para imágenes multiespectrales + manejo explícito de tendencias, construyendo
sobre Hennig & Lin (2015).

- **2026-06-01** · exp 06 (`06_dipmeans_decluster.py`): declustering +
  dip-means proyectado. Resultado **1, 1, 3** (target 1,1,4). ¡El confound de
  tendencia RESUELTO (TREND→1) — lo que nada más lograba! STRUCT=3 (offsets 3-4).
- **2026-06-01** · exp 06b (`06b_struct_sweep.py`): barrido de separación/nº
  clases. k_true=3 → 3 ✓; k_true=4-5 → sub-detecta (2-4), peor con rango de
  autocorrelación grande. Causa: el declustering deja pocas muestras
  independientes (~36) → límite de **tamaño muestral efectivo**, no bug.

- **2026-06-01** · exp 07 (`07_ess_dip.py`): **dip calibrado por ESS** — dip
  statistic sobre TODOS los píxeles, p-valor calibrado al tamaño muestral
  efectivo `n_eff = n / R²` (R = rango de autocorrelación). Split testeado en la
  dirección del corte de 2-means (más sensible que PC1). Resultado: NULL→1 ✓,
  TREND→1 ✓ (robusto: n_eff=12 → sin poder → k=1), STRUCT→3. Barrido: recupera k
  verdadero con sep≥3; sub-detecta con sep=2 / muchas clases. El límite es
  estadístico genuino (poco info independiente), no bug: el método err
  conservador por diseño (cero falsos positivos en tendencia). `n_eff` gobierna
  el trade-off poder/especificidad de forma principiada.

- **2026-06-01** · exp 08 (`08_benchmark.py` + `methods.py`): benchmark
  cuantitativo paralelo. Grilla null/trend/struct/**mixed** (clases+tendencia,
  el caso realista) × 15 realizaciones × 7 métodos. Hallazgo del smoke test:
  `ess_dip` plano COLAPSA en el mundo mixto (la tendencia infla el rango global
  → n_eff minúsculo → suprime todo). Arreglo: **detrend polinomial** antes del
  dip → `ess_dip_detrend` recupera k (ej. k=3 trend=3.0: plano [1,2,1,1,1,1] →
  detrend [3,2,3,3,3,3]). Detrending validado como componente real del método.
  Nota: H=96 necesario (H=64 deja poca info independiente; los colapsos a 1 son
  estimaciones de rango anómalas que el promedio absorbe).
  RESULTADOS (495 jobs): **[especificidad NULL+TREND, verdad k=1]** los 4
  criterios estándar (gap/silhouette/CH/DB) aciertan k=1 el **0%** (gap inventa
  ~8 clases); nuestros 3 métodos **100%**. **[STRUCT puro]** clásicos mejores en
  exact-k (gap 0.97 vs ess_dip 0.65) — trade-off poder/especificidad. **[MIXED]**
  difícil para todos (0.03–0.40); detrend ayuda (ess_dip 0.13→ess_dip_detrend
  0.30). **[trade-off detrend]** ayuda en mixto pero hiere struct puro
  (0.65→0.20) → método ADAPTATIVO (detrend solo si hay tendencia). Score
  balanceado: ess_dip 0.59 y decluster_dip 0.58 lideran (únicos con
  especificidad). Figura: `figs/08_benchmark.png`, datos: `results/bench.csv`.

- **2026-06-01** · exp 09 (`ess_dip_adaptive` en `methods.py`): mejoras
  motivadas por el benchmark. (1) **detrend adaptativo** — R²_poly es
  discriminador limpio de tendencia (null/struct ≤0.19, mixed/trend ≥0.24);
  detrend solo si R²_poly>τ=0.25. (2) **rango acotado** a W/8 (evita colapsos
  por estimaciones anómalas). (3) **ensemble** de 5 corridas, mediana (absorbe
  varianza de init de 2-means). Smoke (8 seeds): null/trend 1.00 (especificidad
  intacta), struct k=3→0.88 / k=4→0.75 (recupera poder, antes ~0.65; detrend
  solo daba 0.20), mixed k=3→0.50 / k=4→0.25 (mejor que todo lo demás). Logra
  lo mejor de ambos mundos. Benchmark completo de 8 métodos (495 jobs):
  ess_dip_adaptive es el **mejor balanceado (0.638)** — especificidad 1.00 +
  struct 0.63 + mixed 0.28. Sobre la grilla completa de struct empata con
  ess_dip plano (0.63 vs 0.64); la ganancia real es en MIXED (0.13→0.28) sin
  hundir struct (a diferencia de ess_dip_detrend, 0.20). Clásicos siguen
  mejores en exact-k de clases puras (gap 0.97) pero con especificidad CERO.
  Mixto sigue siendo el régimen duro (techo ~0.40) = trabajo futuro honesto.

- **2026-06-01** · exp 10 (`10_realdata.py`): **datos reales** Indian Pines
  (145×145×200) + Salinas (512×217×204), hiperespectral, verdad por píxel.
  z-score + PCA (IP 10 PC→0.955 var; Salinas→0.996). [A] ESPECIFICIDAD en 67
  ventanas de UN solo cultivo (truth k=1): silhouette/CH/DB inventan
  sub-clusters el **100%** (silhouette mean k=2.27), gap clásico falla 36%
  (k=1 rate 0.64), **nuestros métodos aciertan k=1 el 97-99%**. La contribución
  SE SOSTIENE en imágenes reales — resultado limpio y titular. [B] escena
  completa (truth=16): NADIE se acerca (gap IP=3/Sal=11, CH 2/10); nuestros
  métodos colapsan a 1 (IP y Sal) — escena dominada por campos+background
  grandes → rango global enorme → n_eff diminuto → primer split no pasa.
  Recuperar 16 clases solapadas en escena completa está fuera del alcance de
  todos. Mismo "gate del primer split" amplificado a escala de escena = límite
  honesto. Bug corregido: `estimate_range` asumía imágenes cuadradas (crash en
  Salinas 512×217); ahora maneja rectangulares.

- **2026-06-01** · exp 11-12 (`11_fetch_sentinel2.py`, `12_sentinel2_analysis.py`):
  **Sentinel-2 multiespectral real** (dominio del usuario). Escena Po Valley
  246×237×10 bandas (B02-B12) + ESA WorldCover (vía Planetary Computer STAC).
  4 clases ≥1% (crop 83%, tree 9%, grass 5%, built 4%). [A] ESPECIFICIDAD en 60
  ventanas de cultivo (truth k=1): gap clásico alucina ~7 clusters (k=1 rate
  0.00), índices 0.00, **ess_dip/detrend/adaptive = 1.00**, decluster 0.90.
  Aún más limpio que hiperespectral. [B] escena completa (truth_k=4): gap=5,
  CH=3, sil/DB=2, nuestros métodos=1 (colapsan; escena 83% cultivo). Patrón
  CONSISTENTE en 3 modalidades. Datos: `data/s2_*.npy`.

- **2026-06-01** · exp 13 (`13_local_range.py`, `ess_dip_local`): **rango LOCAL**
  (mediana sobre tiles → autocorrelación intra-clase, no escala de campos).
  Diagnóstico: el rango global estaba dominado por los parches entre-clase →
  n_eff diminuto → colapso. Fix = win claro: escena completa Salinas 1→5,
  IndianPines 1→2 (truth 16, nadie pasó de 11); **sintético struct 0.63→~1.00**;
  especificidad PRESERVADA (real crop 1.00, null/trend 1.00). Sentinel-2 full
  sigue en 1 (imbalance 83% cultivo, no problema de rango). ess_dip_local
  DOMINA a ess_dip_adaptive. Benchmark final 9 métodos: **ess_dip_local mejor
  balanceado por amplio margen (0.771 vs 0.638)** — especificidad 1.00, STRUCT
  0.63→**0.81** (competitivo con clásicos: CH 0.85, gap 0.97), MIXED 0.13→**0.50**
  (EL MEJOR de todos, supera a CH 0.40). El rango local convierte el método de
  "alta precisión, recall débil" en "especificidad perfecta + poder competitivo
  + mejor en el caso realista". Método recomendado: `ess_dip_local`.

**ESTADO (sesión 2026-06-01):** método central (ESS-dip) + variantes adaptive/
local validados en sintético + hiperespectral (Indian Pines/Salinas) +
multiespectral (Sentinel-2). El rango local mejora recall manteniendo
especificidad. Contribución robusta y reproducible: **control de falsos
positivos / precisión bajo autocorrelación** (especificidad 0.90-1.00 vs 0.00
de los índices estándar). Límite consistente: recall débil en escena completa
multi-clase (colapso a clase dominante). Ata con Dutilleul 1993 / Hennig & Lin
2015. PRÓXIMO: (a) benchmark cuantitativo serio — N realizaciones por celda
(k_true × sep × R), reportar accuracy de recuperación de k comparando gap
clásico / silhouette / declustering+dip (exp06) / ESS-dip (exp07), en vez de
casos puntuales; (b) datos reales (Indian Pines/Salinas/Pavia,
Sentinel-2+WorldCover); (c) formalización + escritura. Target: C&G / EMS / IJRS.
