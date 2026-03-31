# Clasificación de ELA mediante análisis acústico del habla
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
├── notebooks/
│   ├── 01_dataset_validation.ipynb              # Validación del dataset y definición del split
│   ├── 02_model_random_forest_v1.0.ipynb        # RF baseline — solo variables acústicas
│   ├── 02_model_random_forest_v2.0.ipynb        # RF extendido — acústicas + demográficas
│   ├── 03_data_preprocessing.ipynb              # Pipeline de audio → mel spectrograms
│   ├── 04_model_deep_learning_v1.0.ipynb        # LSTM bidireccional sobre espectrogramas
│   ├── 04_model_deep_learning_v2.0.ipynb        # Autoencoder → embedding → clasificación
│   └── 05_model_comparison.ipynb                # Comparativa final RF vs DL
│
├── artifacts/
│   ├── metadata/
│   │   └── VOC-ALS.xlsx                         # Metadata del dataset (no incluida en el repo)
│   ├── splits/
│   │   └── subject_split.csv                    # Partición reproducible por sujeto
│   └── preprocessed/                            # Tensores .pt del dataset preprocesado
│       ├── train/
│       ├── val/
│       └── test/
│
├── results/
│   ├── figures/                                 # Figuras exportadas (ROC, confusión, importancias)
│   ├── metrics/                                 # Tablas de métricas por modelo
│   └── mlruns/                                  # Experimentos MLflow
│
├── requirements.txt
└── README.md
```

> **Nota:** Los archivos de audio crudos y el Excel del dataset no se incluyen en el repositorio por motivos de tamaño y privacidad de los datos clínicos.

---

## Notebooks

Los notebooks están diseñados para ejecutarse en orden secuencial. Cada uno depende de los artefactos generados por el anterior. El único archivo que conecta todos los notebooks es `artifacts/splits/subject_split.csv`, generado en NB01 y cargado directamente en todos los siguientes sin redefinirse.

| Notebook | Descripción | Estado | Artefactos generados |
|----------|-------------|--------|----------------------|
| `01_dataset_validation.ipynb` | Validación estructural del dataset, análisis demográfico, análisis de features acústicas (distribuciones, outliers, correlación), separabilidad (PCA, t-SNE) y definición del split 70/15/15 por sujeto | ✅ Completo (Revisar) | `subject_split.csv`, figuras fig_01–fig_15 |
| `02_model_random_forest_v1.0.ipynb` | Random Forest baseline sobre las 50 variables acústicas. Nested CV (5×3), GridSearchCV, permutation importance, SHAP, calibración de probabilidades, tracking MLflow | ✅ Completo (Revisar) | métricas val/test, importancias, figuras ROC/confusión |
| `02_model_random_forest_v2.0.ipynb` | RF extendido con variables demográficas (edad, sexo). Comparativa directa con v1.0, análisis de sesgo por subgrupos, ranking AGE/SEX vs. features acústicas | ✅ Completo (Revisar) | métricas val/test extendido, tabla comparativa v1 vs v2 |
| `03_data_preprocessing.ipynb` | Pipeline de preprocesamiento de audio: resampling a 16 kHz, eliminación de silencios, normalización de loudness, generación de mel spectrograms (N_FFT=2048, HOP=512, N_MELS=128). Exportación de tensores `.pt` por split | ✅ Completo (Revisar) | `preprocessed/train/`, `preprocessed/val/`, `preprocessed/test/`, `config.json` |
| `04_model_deep_learning_v1.0.ipynb` | LSTM bidireccional (2 capas, hidden=256) sobre mel spectrograms. VariableLengthCollator con padding y máscaras, class weighting (ALS=2, HC=1), early stopping | 🔵 En curso | checkpoint LSTM v1.0, curvas de aprendizaje, métricas val/test |
| `04_model_deep_learning_v2.0.ipynb` | Autoencoder convolucional entrenado sobre espectrogramas → extracción de embeddings del encoder → clasificador sobre el espacio latente. Comparativa con LSTM v1.0 | 🔵 En curso | checkpoint autoencoder, embeddings, métricas val/test, tabla comparativa DL v1 vs v2 |
| `05_model_comparison.ipynb` | Evaluación final en test de todos los modelos. Tabla comparativa global RF v1/v2 vs DL v1/v2, curvas ROC superpuestas, análisis de errores clínicamente relevantes, selección del modelo definitivo | 🟡 Pendiente | tabla final, figura ROC comparativa, decisión modelo definitivo |

---

## Pipeline del proyecto

```
VOC-ALS.xlsx          audio/*.wav
      │                    │
      ▼                    ▼
 NB01: Validación    NB03: Preprocessing
  └─ subject_split.csv    └─ tensores .pt por split
           │                        │
     ┌─────┴─────┐         ┌────────┴────────┐
     ▼           ▼         ▼                 ▼
NB02 v1.0    NB02 v2.0  NB04 v1.0        NB04 v2.0
RF baseline  RF extendido  LSTM bidir.   Autoencoder
     │            │            │               │
     └─────┬──────┘            └──────┬────────┘
           ▼                          │
     NB05: Comparativa ◄──────────────┘
      todos los modelos (test set)
```

---

## Decisiones metodológicas clave

**Split por sujeto, no por audio.** La partición train/validation/test se realiza a nivel de paciente (ID), no de muestra acústica. Un split por audio permitiría que el mismo sujeto aparezca en train y en test, generando data leakage e inflando artificialmente las métricas. Con N=153 sujetos, el split resultante es 107 train / 23 val / 23 test.

**Métrica primaria: recall en ALS.** En un escenario de cribado diagnóstico, un falso negativo (paciente ALS clasificado como HC) tiene un coste clínico mayor que un falso positivo: retrasa el diagnóstico y el acceso a tratamiento sintomático. Por este motivo, el recall en la clase ALS y la balanced accuracy son las métricas primarias del proyecto. La accuracy estándar es engañosa con el desbalance 2:1.

**Nested CV para estimación honesta del rendimiento.** El uso de GridSearchCV sin un loop externo de evaluación produce estimaciones optimistas del rendimiento porque los mismos datos se usan para optimizar hiperparámetros y para evaluar el modelo. El nested CV (5 folds externos × 3 folds internos) desacopla completamente ambos procesos.

**Test set abierto una sola vez.** El conjunto de test no interviene en ninguna decisión de modelado y se evalúa únicamente al final de cada pipeline, tras haber completado el análisis de validation. En NB05, cada modelo se evalúa en test una única vez para la comparativa final.

**Preprocessing congelado antes del modelado DL.** Los parámetros del pipeline de audio (SR=16000 Hz, N_FFT=2048, HOP_LENGTH=512, N_MELS=128) se fijan en NB03 antes de entrenar ningún modelo de DL. Modificarlos después de ver los resultados en validation equivaldría a optimizar el preprocessing sobre datos de evaluación.

---

## Entorno y dependencias

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt
```

**Versiones principales:**

| Librería | Versión |
|----------|---------|
| Python | 3.13.0 |
| NumPy | 2.4.2 |
| Pandas | 3.0.1 |
| scikit-learn | ≥ 1.4 |
| Librosa | 0.11.0 |
| PyTorch | ≥ 2.2 |
| Torchaudio | ≥ 2.2 |
| SHAP | ≥ 0.45 |
| MLflow | ≥ 2.10 |
| Matplotlib | ≥ 3.8 |
| Seaborn | ≥ 0.13 |

---

## Reproducibilidad

Todos los experimentos usan `random_state=42` / `SEED=42` de forma consistente:

```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

Para reproducir los resultados del Random Forest:

```bash
# Lanzar MLflow UI
mlflow ui --backend-store-uri ./results/mlruns
# Acceder en: http://localhost:5000
```

Para reproducir el split exacto, cargar directamente el CSV generado en NB01:

```python
import pandas as pd
split_df = pd.read_csv("artifacts/splits/subject_split.csv")
train_ids = split_df[split_df["Split"] == "Train"]["ID"].values
val_ids   = split_df[split_df["Split"] == "Validation"]["ID"].values
test_ids  = split_df[split_df["Split"] == "Test"]["ID"].values
```

---

## Resultados (actualizados conforme avanza el proyecto)

| Modelo | Conjunto | Balanced Acc | Recall ALS | AUC |
|--------|----------|:------------:|:----------:|:---:|
| RF v1.0 (solo acústicas) | Nested CV | 0.5452 | 0.4714 | 0.5684 |
| RF v1.0 (solo acústicas) | Validation | 0.7417 | 0.7500 | 0.8250 |
| RF v1.0 (solo acústicas) | Test | 0.6696 | 0.7143 | 0.5893 |
| RF v2.0 (acústicas + demográficas) | Nested CV | 0.5381 | 0.4429 | 0.5572 |
| RF v2.0 (acústicas + demográficas) | Validation | 0.7417 | 0.7500 | 0.8250 |
| RF v2.0 (acústicas + demográficas) | Test | 0.6696 | 0.7143 | 0.5893 |
| DL v1.0 — LSTM bidireccional | Validation | — | — | — |
| DL v1.0 — LSTM bidireccional | Test | — | — | — |
| DL v2.0 — Autoencoder + clasificador | Validation | — | — | — |
| DL v2.0 — Autoencoder + clasificador | Test | — | — | — |

> La tabla se actualiza conforme se completan los notebooks.

---

## Limitaciones conocidas

- **Tamaño muestral reducido** (N=153). Con 23 sujetos en test, una diferencia de 1 sujeto mal clasificado representa ~4 puntos porcentuales en accuracy. Las métricas deben interpretarse con intervalos de confianza amplios.
- **Datos transversales**. VOC-ALS contiene una única sesión de grabación por sujeto. El análisis longitudinal de la progresión de la disartria no es posible con este dataset.
- **Grabaciones en condiciones controladas**. El rendimiento puede degradarse en entornos ruidosos o con equipos de grabación distintos a los del protocolo original.
- **Ausencia de información sobre estadio clínico**. La severidad de la afectación bulbar no está disponible como variable, lo que impide estratificar el análisis por grado de disartria.

---

## Referencias

- Dataset VOC-ALS: Dubbioso, R., Spisto, M., Verde, L., Iuzzolino, VV, Senerchia, G., Salvatore, E., ... & Sannino, G. (2024). Base de datos de señales de voz de pacientes con ELA con diferente gravedad de disartria y controles sanos . Scientific Data, 11(1), 800.
- Tsanas et al. (2012). Accurate telemonitoring of Parkinson's disease symptom severity using nonlinear speech signal processing and statistical machine learning. *Biomedical Engineering*.
- Vashkevich et al. (2021). Classification of ALS patients based on acoustic analysis of sustained vowel phonations. *Biomedical Signal Processing and Control*.

---

*TFG — Ingeniería Biomédica · Universidad Europea de Madrid · Curso 2025/2026*
