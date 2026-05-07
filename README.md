# Desarrollo de modelos de Machine Learning y Deep Learning basados en biomarcadores vocales para la clasificación binaria de ALS/HC
### Trabajo Fin de Grado — Ingeniería Biomédica · Curso 2025/2026

Este repositorio contiene el código, notebooks y artefactos del TFG cuyo objetivo es clasificar pacientes con **Esclerosis Lateral Amiotrófica (ELA/ALS)** frente a controles sanos (HC) mediante el análisis acústico de señales de voz. El proyecto explota los biomarcadores vocales de la disartria —alteraciones en la articulación, el ritmo del habla y la estabilidad de la voz— como señal de clasificación.

---

## Contexto clínico

La ELA produce en una proporción significativa de casos afectación bulbar progresiva que compromete los mecanismos neuromusculares del habla. Esta alteración se manifiesta como **disartria**: una degradación cuantificable de la articulación, la prosodia y la calidad vocal. Variables acústicas como la frecuencia fundamental (F0), el HNR (*Harmonics-to-Noise Ratio*), el jitter y el shimmer reflejan directamente la estabilidad del control motor laríngeo, y constituyen biomarcadores no invasivos del estado del sistema nervioso motor.

El dataset utilizado es **VOC-ALS**: 153 sujetos (102 ALS, 51 HC), con grabaciones de 10 tareas vocales por sujeto (vocales sostenidas A/E/I/O/U, sílabas diadococinéticas PA/TA/KA, lectura y habla espontánea).

---

## Estructura del repositorio

```
.
├── config.yaml                                   # Configuración centralizada (rutas, dataset, audio, hiperparámetros)
├── notebooks/
│   ├── 01_dataset_validation.ipynb              # Validación del dataset y definición del split CV+Test
│   ├── 02_model_random_forest_v1.0.ipynb        # RF baseline — solo variables acústicas (50)
│   ├── 02_model_random_forest_v2.0.ipynb        # RF extendido — acústicas + demográficas (edad, sexo)
│   ├── 03_data_preprocessing.ipynb              # Pipeline de audio → mel spectrograms
│   ├── 04_model_deep_learning_v1.0.ipynb        # LSTM bidireccional sobre espectrogramas
│   ├── 04_model_deep_learning_v2.0.ipynb        # Autoencoder → embedding → clasificación
│   └── 05_model_comparison.ipynb                # Comparativa final RF vs DL
│
├── artifacts/
│   ├── metadata/
│   │   └── VOC-ALS.xlsx                         # Metadata del dataset (no incluida en el repo)
│   ├── raw/                                     # Audios .wav crudos (no incluidos en el repo)
│   ├── splits/
│   │   └── subject_split.csv                    # Partición por sujeto: CV (5 folds SGKF) + Test
│   ├── preprocessed/                            # Tensores .pt del dataset preprocesado (sin sub-splits)
│   └── models/                                  # Modelos serializados (.pkl, .pt)
│
├── results/                                     # Métricas (.json/.csv), figuras y predicciones por modelo
├── mlruns/                                      # Experimentos MLflow
│
├── requirements.txt
└── README.md
```

> **Nota:** Los archivos de audio crudos y el Excel del dataset no se incluyen en el repositorio por motivos de tamaño y privacidad de los datos clínicos.

---

## `config.yaml`: configuración centralizada

Todas las rutas, parámetros del dataset, semilla, configuración del split, parámetros de audio y de los modelos están parametrizados en `config.yaml` en la raíz del repositorio. Cada notebook incluye una función `load_config()` que busca el archivo hacia arriba desde el CWD (o vía la variable de entorno `TFG_CONFIG`), resuelve las rutas relativas a la raíz y expone los valores como un diccionario `CONFIG`.

```yaml
paths:    { dataset, audio, preprocessed, splits, results, models, mlruns }
dataset:  { sheet_name, target_col, id_col, age_col, sex_col,
            expected_tasks, acoustic_prefixes }
seed:     42
split:    { test_size: 0.15, n_cv_folds: 5 }
audio:    { sr: 8000, n_fft: 1024, hop_length: 256, n_mels: 64 }
bilstm:   { hidden_size, num_layers, dropout, bidirectional, max_seq_len,
            batch_size, learning_rate, num_epochs, patience,
            weight_decay, label_smoothing }
rf:       { inner_cv_folds, param_grid: { feature_selection__k,
            classifier__n_estimators, classifier__max_depth,
            classifier__min_samples_split, classifier__min_samples_leaf,
            classifier__class_weight } }
```

Cualquier cambio de ruta, semilla o hiperparámetro se realiza en este archivo y propaga automáticamente a todos los notebooks.

---

## Notebooks

Los notebooks están diseñados para ejecutarse en orden secuencial. Cada uno depende de los artefactos generados por el anterior. El único archivo que conecta todos los notebooks es `artifacts/splits/subject_split.csv`, generado en NB01 y cargado directamente en todos los siguientes sin redefinirse.

| Notebook | Descripción | Estado | Artefactos generados |
|----------|-------------|--------|----------------------|
| `01_dataset_validation.ipynb` | Validación estructural del dataset, análisis demográfico, análisis de features acústicas (distribuciones, outliers, correlación), separabilidad (PCA, t-SNE) y definición del split **CV + Test** por sujeto: 15 % de hold-out estratificado para test y 85 % restante repartido en 5 folds mediante `StratifiedGroupKFold` | ✅ Completo (CV+Test) | `subject_split.csv` (cols: `ID`, `Category`, `Split` ∈ {CV, Test}, `Fold` ∈ {0..4, NaN}), figuras fig_01–fig_15 |
| `02_model_random_forest_v1.0.ipynb` | Random Forest baseline restringido a las **50 variables acústicas** (sin demográficas ni clínicas). Nested CV (5×3 SGKF), GridSearchCV (288 combinaciones), permutation importance, MDI, calibración de probabilidades, tracking MLflow | ✅ Completo (CV+Test, acústicas only) | métricas CV/test (`cv_metrics.json`, `test_metrics.json`), `results_summary.csv`, importancias, figuras ROC/confusión |
| `02_model_random_forest_v2.0.ipynb` | RF extendido con variables demográficas (`Age`, `Sex`). Comparativa directa con v1.0, análisis de sesgo por subgrupos, ranking AGE/SEX vs. features acústicas, evaluación automática del veredicto sobre 7 criterios cuantitativos | ✅ Completo (CV+Test, acústicas + AGE + SEX) | métricas CV/test extendido (`cv_metrics.json`, `test_metrics.json`), `results_summary.csv`, comparativa baseline vs extendido (FP/FN, importancias), figuras ROC/confusión |
| `03_data_preprocessing.ipynb` | Pipeline de preprocesamiento de audio: resampling a 8 kHz, eliminación de silencios, normalización de loudness, generación de mel spectrograms (`N_FFT=1024`, `HOP=256`, `N_MELS=64`). Estadísticos z-score calculados sobre CV (130 sujetos / 1.040 archivos / 461.538 frames). Exportación de tensores `.pt` a `preprocessed/cv/` y `preprocessed/test/` | ✅ Completo (CV+Test, parametrizado) | tensores `.pt` (1.040 CV + 184 Test), `config.json`, `preprocessing_log.csv` |
| `04_model_deep_learning_v1.0.ipynb` | BiLSTM bidireccional con attention pooling sobre mel spectrograms. **CV de 5 folds (StratifiedGroupKFold) + ensemble** de los 5 modelos en Test. Diagnóstico de colapso a clase mayoritaria por época, early stopping sobre `val_balanced_accuracy`. Saliency maps y embeddings desde el modelo del fold 0 | ✅ Completo (CV 5-fold + ensemble, v1.2) | 5 checkpoints `bilstm_fold{0..4}.pt`, 5 figuras de diagnóstico, `cv_metrics.json`, `test_metrics.json`, predicciones OOF y de test |
| `05_model_Autoencoder_v1.0.ipynb` | Autoencoder CNN 2D entrenado sobre espectrogramas mel (5-fold CV, un AE por fold) → extracción de embeddings OOF del encoder → clasificador MLP en dos fases (Fase 1: encoder congelado; Fase 2: fine-tuning conjunto con LR diferencial). Análisis del espacio latente (PCA + t-SNE), visualización de reconstrucciones y ensemble de 5 modelos en Test | ✅ Completo (CV+Test) | 5 checkpoints `autoencoder_fold{0..4}.pt`, 5 checkpoints `clf_fold{0..4}.pt`, `oof_embeddings.npz`, `test_embeddings.npz`, `cv_metrics.json`, `test_metrics.json`, figuras reconstrucciones/espacio latente/evaluación |
| `05_model_comparison.ipynb` | Evaluación final en test de todos los modelos. Tabla comparativa global RF v1/v2 vs DL v1/v2, curvas ROC superpuestas, análisis de errores clínicamente relevantes, selección del modelo definitivo | 🟡 Pendiente | tabla final, figura ROC comparativa, decisión modelo definitivo |

---

## Pipeline del proyecto

```
VOC-ALS.xlsx          audio/*.wav            config.yaml
      │                    │                       │
      ▼                    ▼                       ▼
 NB01: Validación    NB03: Preprocessing    (parametriza todo el pipeline)
  └─ subject_split.csv    └─ tensores .pt
     (CV + Test, 5 folds)
           │                        │
     ┌─────┴─────┐         ┌────────┴────────┐
     ▼           ▼         ▼                 ▼
NB02 v1.0    NB02 v2.0  NB04 v1.0        NB04 v2.0
RF acústicas  RF + demo  LSTM bidir.   Autoencoder
     │            │            │               │
     └─────┬──────┘            └──────┬────────┘
           ▼                          │
     NB05: Comparativa ◄──────────────┘
      todos los modelos (test set)
```

---

## Decisiones metodológicas clave

**Split por sujeto, no por audio.** La partición se realiza a nivel de paciente (`ID`), no de muestra acústica. Un split por audio permitiría que el mismo sujeto aparezca en CV y en test, generando data leakage e inflando artificialmente las métricas. Con N=153 sujetos, el resultado es un pool de **130 sujetos en CV** repartidos en 5 folds estratificados y un **hold-out de 23 sujetos en test**.

**`StratifiedGroupKFold` en CV.** Los 5 folds se construyen con `StratifiedGroupKFold(groups=ID)`, que garantiza simultáneamente (a) que ningún sujeto cae en más de un fold y (b) que la proporción ALS/HC se preserva dentro de cada fold (~17 ALS / ~9 HC por fold). El esquema sustituye al antiguo split 70/15/15 (Train/Validation/Test).

**Métrica primaria: recall en ALS.** En un escenario de cribado diagnóstico, un falso negativo (paciente ALS clasificado como HC) tiene un coste clínico mayor que un falso positivo: retrasa el diagnóstico y el acceso a tratamiento sintomático. Por este motivo, el recall en la clase ALS y la balanced accuracy son las métricas primarias del proyecto. La accuracy estándar es engañosa con el desbalance 2:1.

**Nested CV para estimación honesta del rendimiento.** El uso de `GridSearchCV` sin un loop externo de evaluación produce estimaciones optimistas del rendimiento porque los mismos datos se usan para optimizar hiperparámetros y para evaluar el modelo. El nested CV (5 folds externos × 3 folds internos) desacopla completamente ambos procesos: el outer loop opera sobre los folds del `subject_split.csv` y el inner loop construye un sub-`StratifiedGroupKFold` dentro de cada outer fold.

**Test set abierto una sola vez.** El conjunto de test no interviene en ninguna decisión de modelado y se evalúa únicamente al final de cada pipeline, tras haber completado el análisis de CV. En NB05, cada modelo se evalúa en test una única vez para la comparativa final.

**Espacio de features explícito en RF.**
- **v1.0 (baseline):** sólo las 50 variables acústicas (5 prefijos × 10 tareas). Las demográficas y las clínicas ALS-only (`ALSFRS-R*`, `Revised_ElEscorial_Criteria`, `OnsetRegion`, `Therapy`, `FVC%`, `DiagnosticDelay`, `DiseaseDuration`) se excluyen del modelo.
- **v2.0 (extendido):** 50 acústicas + `Age (years)` + `Sex`. Permite estudiar el impacto incremental de las demográficas sobre el modelo acústico puro y descartar dependencias artificiosas.

**Preprocessing congelado antes del modelado DL.** Los parámetros del pipeline de audio (`SR=8000 Hz`, `N_FFT=1024`, `HOP_LENGTH=256`, `N_MELS=64`) se fijan en `config.yaml`/`audio` antes de entrenar ningún modelo de DL. Modificarlos después de ver los resultados en CV equivaldría a optimizar el preprocessing sobre datos de evaluación.

---

## Entorno y dependencias

**Requisitos previos:** Python 3.13 y Git instalados.

```bash
# 1. Clonar el repositorio
git clone <URL_del_repositorio>
cd TFG

# 2. Crear el entorno virtual
python -m venv .venv
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # Linux/macOS

# 3. Instalar PyTorch (elegir una opción)
# Con GPU NVIDIA (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Sin GPU (solo CPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Instalar el resto de dependencias
pip install -r requirements.txt

# 5. Lanzar los notebooks
pip install jupyter
jupyter notebook
```

**Versiones principales (entorno de referencia):**

| Librería | Versión |
|----------|---------|
| Python | 3.13.0 |
| NumPy | 2.4.2 |
| Pandas | 2.3.3 |
| scikit-learn | ≥ 1.4 |
| Librosa | 0.11.0 |
| SoundFile | ≥ 0.12 |
| PyTorch | ≥ 2.2 |
| Torchaudio | ≥ 2.2 |
| MLflow | ≥ 2.10 |
| PyYAML | ≥ 6.0 |
| Matplotlib | ≥ 3.8 |
| Seaborn | ≥ 0.13 |
| SciPy | ≥ 1.11 |

---

## Reproducibilidad

Todos los experimentos usan `random_state=42` / `SEED=42` de forma consistente, leído desde `config.yaml`:

```python
import random, numpy as np, torch, yaml
from pathlib import Path

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

SEED = CONFIG["seed"]   # 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

Para visualizar los experimentos del Random Forest:

```bash
# Lanzar MLflow UI desde la raíz del repo
mlflow ui --backend-store-uri ./mlruns
# Acceder en: http://localhost:5000
```

Para reproducir el split exacto, cargar directamente el CSV generado en NB01:

```python
import pandas as pd
split_df = pd.read_csv("artifacts/splits/subject_split.csv")

# Hold-out de test (15 %)
test_ids = split_df.loc[split_df["Split"] == "Test", "ID"].values

# Pool de CV (85 %), con asignación a 5 folds SGKF
cv_df    = split_df.loc[split_df["Split"] == "CV", ["ID", "Category", "Fold"]]
fold0_ids = cv_df.loc[cv_df["Fold"] == 0, "ID"].values   # ejemplo: fold 0
```

---

## Resultados (actualizados conforme avanza el proyecto)

| Modelo | Conjunto | Balanced Acc | Recall ALS | AUC |
|--------|----------|:------------:|:----------:|:---:|
| **RF v1.0** (solo acústicas) | Nested CV (5×3 SGKF) | **0.5821 ± 0.0788** | **0.6392 ± 0.0457** | **0.6658 ± 0.0620** |
| **RF v1.0** (solo acústicas) | Test (n = 23)        | **0.5982**          | **0.6250**          | **0.6071**          |
| **RF v2.0** (acústicas + demográficas) | Nested CV (5×3 SGKF) | **0.5710 ± 0.0750** | **0.6392 ± 0.0457** | **0.6698 ± 0.0649** |
| **RF v2.0** (acústicas + demográficas) | Test (n = 23)        | **0.5982**          | **0.6250**          | **0.6071**          |
| **DL v1.2** — BiLSTM (CV 5-fold + ensemble) | CV (5-fold SGKF, sujeto-level) | **0.6514 ± 0.0876** | **0.7556 ± 0.1313** | **0.7034 ± 0.0743** |
| **DL v1.2** — BiLSTM (CV 5-fold + ensemble) | Test (n = 23, ensemble sujeto-level) | **0.7634** | **0.8125** | **0.8571** |
| **AE v1.0** — CNN2D Autoencoder + MLP (CV 5-fold + ensemble) | CV (5-fold SGKF, sample-level) | **0.6361 ± 0.0188** | **0.7097 ± 0.0281** | **0.6497 ± 0.0179** |
| **AE v1.0** — CNN2D Autoencoder + MLP (CV 5-fold + ensemble) | Test (n = 23, ensemble sujeto-level) | **0.5491** | **0.8125** | **0.6071** |

> Todas las métricas se reportan a **nivel de sujeto** para consistencia entre modelos. La fila CV del AE v1.0 corresponde a sample-level (múltiples tensores por sujeto) tal como los registra el bucle de entrenamiento; el test es sujeto-level por ensemble de los 5 modelos.

**Hallazgos del baseline RF v1.0:**

- El grid search selecciona `feature_selection__k=10`, `n_estimators=200`, `max_depth=5`, `min_samples_leaf=10`, `min_samples_split=10`, `class_weight='balanced'` con un best inner CV score de 0.622 en balanced accuracy.
- Permutation importance y MDI convergen en señalar `stdevF0Hz_PA` como la feature más informativa por un margen amplio (Perm. 0.084, MDI 0.196), seguida de `stdevF0Hz_TA`, `stdevF0Hz_KA`, `stdevF0Hz_U` y métricas de jitter/shimmer en vocales.
- La concentración de relevancia en tareas de diadococinesia (PA, TA, KA) es coherente con la fisiopatología de la disartria en ELA.
- La calibración de probabilidades (Platt, isotónica) no mejora la capacidad discriminativa con n=23 en test.
- Gap entre nested CV y test pequeño (< 0.06 en balanced accuracy y AUC) → el modelo no sobre-ajusta de forma severa, dentro de las limitaciones del tamaño muestral.

**Hallazgos del modelo extendido RF v2.0 (acústicas + AGE + SEX):**

- El grid search converge a **idénticos hiperparámetros que v1.0** (`k=10`, `n_estimators=200`, `max_depth=5`, `min_samples_leaf=10`, `min_samples_split=10`, `class_weight='balanced'`) con el mismo best inner CV score de 0.622.
- **AGE y SEX caen al fondo absoluto del ranking** de permutation importance (posiciones 51 y 52 de 52 features) con importancia exactamente 0.000. Ninguna de las dos supera el filtro de `SelectKBest`: en el estadístico F de ANOVA, `Sex_M` queda en rango 49/52 (F=0.122) y `Age` en 52/52 (F=0.012), por lo que **el modelo final no las usa** — las 10 features seleccionadas son todas acústicas.
- **En test, el modelo extendido produce exactamente las mismas 23 predicciones que el baseline** (3 FP coincidentes: CT014, CT053, CT067; 6 FN coincidentes: PZ016, PZ058, PZ094, PZ098, PZ099, PZ105 — todos varones, edad media 57.5 años). Δ Bal.Acc Test = 0.000, Δ AUC Test = 0.000.
- En nested CV, v2.0 obtiene Bal.Acc 0.571 ± 0.075 vs. 0.582 ± 0.079 del baseline (**Δ = −0.011**, dentro del margen de variabilidad inter-fold).
- **Veredicto cuantitativo del notebook (4 SI / 3 NO sobre 7 criterios):** las variables demográficas no mejoran el modelo y añaden complejidad innecesaria. El baseline v1.0 es preferible por parsimonia.
- **Coherencia con NB01:** las tres predicciones derivadas de la homogeneidad demográfica del Notebook 1 (importancia AGE baja, SEX no seleccionado, mejora marginal del extendido) quedan empíricamente confirmadas (3/3).

**Hallazgos del modelo profundo DL v1.2 (BiLSTM + CV 5-fold + ensemble):**

- **Test ensemble a nivel de sujeto: BalAcc=0.7634, Recall ALS=0.8125, AUC=0.8571**, claramente por encima del baseline RF tanto en discriminación como en capacidad de ranking. El BiLSTM detecta correctamente 13 de 16 sujetos ALS (recall 0.81) y 5 de 7 sujetos HC (recall 0.71) en el conjunto de Test.
- La migración de hold-out único a **CV 5-fold + ensemble** es la decisión metodológica con mayor impacto cuantitativo sobre el modelo profundo: aproximadamente +5 puntos de balanced accuracy y +12 puntos de AUC a nivel de sujeto en Test, sin cambiar la arquitectura. El ensemble actúa como regularizador implícito sobre un dataset pequeño.
- Variabilidad entre los 5 folds (a nivel de sujeto, 26 sujetos por fold): balanced_accuracy mean = 0.6514 ± 0.0876, AUC mean = 0.7034 ± 0.0743. La desviación estándar inter-fold relativamente alta (~9 puntos en BalAcc, ~7 puntos en AUC) refleja la sensibilidad a qué subconjunto concreto de 26 sujetos toca como validación, esperable en datasets pequeños. La métrica del ensemble en Test (BalAcc 0.7634, AUC 0.8571) es más estable que cualquier fold individual: el promedio de probabilidades de los 5 modelos suaviza la varianza y produce predicciones más robustas.
- **El detector de colapso a clase mayoritaria no se ha disparado en ningún fold ni época**: la fracción de predicciones ALS se ha mantenido entre 0.5 y 0.85, lejos de las zonas críticas (>0.95 o <0.05). El cambio de criterio de early stopping a `val_balanced_accuracy` (en lugar de `val_loss`) ha sido determinante para evitar checkpoints sesgados.
- La agregación a nivel de sujeto (promedio de las 8 tareas vocales) mejora todas las métricas tanto en OOF como en Test: balanced accuracy +3-5 puntos, AUC +6-8 puntos, recall ALS +7-9 puntos. Este patrón consistente confirma el valor de reportar métricas a nivel de sujeto como métricas principales del modelo.
- Análisis de saliency (modelo del fold 0): las bandas de baja frecuencia (subgraves <150 Hz, donde se sitúa F0) son las que presentan mayor activación diferencial en sujetos ALS, coherente con la fisiopatología de la disartria flácido-espástica. La banda >4 kHz tiene saliency exactamente cero, validando la decisión de fijar SR=8 kHz.
- **Limitación residual:** el recall HC en Test (0.7143 a nivel de sujeto) sigue siendo inferior al recall ALS (0.8125), reflejo del desbalance ALS:HC en Test; el modelo conserva un sesgo residual hacia la clase mayoritaria, aunque mucho menos pronunciado que en versiones single-fold previas.

**Hallazgos del Autoencoder CNN2D v1.0 (AE + MLP clasificador):**

- **Test ensemble a nivel de sujeto: BalAcc=0.549, Recall ALS=0.813, AUC=0.607.** El modelo muestra un sesgo pronunciado hacia la clase ALS en test: Recall ALS muy alto (0.813) a costa de un Recall HC muy bajo (0.286, es decir, solo 2 de 7 HC correctamente clasificados). Esto sugiere que el espacio latente del autoencoder no separa suficientemente ambas clases, y el clasificador resuelve la ambigüedad inclinándose hacia la clase mayoritaria del entrenamiento.
- **CV (5-fold, sample-level): BalAcc=0.636 ± 0.019, Recall ALS=0.710 ± 0.028, AUC=0.650 ± 0.018.** Las métricas de CV son más equilibradas que las de test, lo que indica cierta inestabilidad del modelo en el conjunto de test (23 sujetos, donde 1 sujeto equivale a ~4 puntos en BalAcc).
- **Comparativa con BiLSTM v1.2:** el BiLSTM supera al autoencoder en todas las métricas de test (BalAcc +0.167, AUC +0.207). La representación temporal secuencial del BiLSTM captura mejor las dependencias de largo alcance de la disartria que la representación espacial del CNN 2D con ratio de compresión 400:1.
- **El análisis PCA/t-SNE del espacio latente** revela separación parcial entre ALS y HC, confirmando que el autoencoder captura diferencias espectrales sin supervisión, aunque insuficientes para clasificación robusta sin fine-tuning.
- **El entrenamiento en dos fases** (encoder congelado → fine-tuning) estabiliza la convergencia: la Fase 1 establece un punto de partida razonable para el clasificador, y la Fase 2 con LR diferencial (encoder × 0.1) ajusta la representación sin destruir la estructura no supervisada aprendida.

---

## Limitaciones conocidas

- **Tamaño muestral reducido** (N=153). Con 23 sujetos en test, una diferencia de 1 sujeto mal clasificado representa ~4 puntos porcentuales en accuracy. Las métricas deben interpretarse con intervalos de confianza amplios.
- **Datos transversales**. VOC-ALS contiene una única sesión de grabación por sujeto. El análisis longitudinal de la progresión de la disartria no es posible con este dataset.
- **Grabaciones en condiciones controladas**. El rendimiento puede degradarse en entornos ruidosos o con equipos de grabación distintos a los del protocolo original.
- **Ausencia de información sobre estadio clínico** *en el espacio de features de los modelos*. Aunque el dataset contiene ALSFRS-R, El Escorial, FVC% y otras variables clínicas, éstas se excluyen deliberadamente del modelado: sólo existen para sujetos ALS y su uso introduciría leakage trivial. La severidad de la afectación bulbar tampoco está disponible como variable estratificadora.

---

## Referencias

- Dataset VOC-ALS: Dubbioso, R., Spisto, M., Verde, L., Iuzzolino, V. V., Senerchia, G., Salvatore, E., ... & Sannino, G. (2024). Voice signals database of ALS patients with different dysarthria severity and healthy controls. *Scientific Data*, 11(1), 800.
- Tsanas et al. (2012). Accurate telemonitoring of Parkinson's disease symptom severity using nonlinear speech signal processing and statistical machine learning. *Biomedical Engineering*.
- Vashkevich et al. (2021). Classification of ALS patients based on acoustic analysis of sustained vowel phonations. *Biomedical Signal Processing and Control*.

---

*TFG — Ingeniería Biomédica · Universidad Europea de Madrid · Curso 2025/2026*
