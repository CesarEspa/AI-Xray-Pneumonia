# AI-XRay

Clasificación de radiografías de tórax pediátricas (NORMAL / PNEUMONIA) con
**CNN + Transfer Learning (ResNet50)**, entrenada con **TensorFlow/Keras**, con
seguimiento completo de experimentos en **MLflow** y servida mediante una **API REST
(FastAPI)**.

**Proyecto Final Independiente** — Clase 4, Unidad 3 · Máster en IA & Data Science (DevSeniorCode)
**Autor:** César Libardo España Salguero

**Demo en línea:** https://cesarespa.github.io/AI-Xray-Pneumonia/ · el modelo corre en el navegador
(la imagen no sale del equipo), con el umbral clínico y el mapa de activación de clase. Ver la sección 6b.

> **Aviso:** este sistema es un proyecto académico de inteligencia artificial y **no
> constituye un diagnóstico médico** ni sustituye la evaluación de un profesional de
> la salud. El aviso se devuelve en cada respuesta de la API (campo `disclaimer`) y
> aparece en el Executive Summary.

---

## 0. Resumen de resultados

Modelo registrado: **`aixray_pneumonia_classifier` v1** (stage `Staging`, alias `staging`) =
**Variante A · fine-tuning en dos fases**, elegido por el mayor F1 de validación.

| Run | F1 val | AUC val | F1 test | IC 95 % F1 test (bootstrap) | AUC test |
|---|---|---|---|---|---|
| **Var. A · fine-tuning 2 fases** ✅ | **0.962** | **0.991** | 0.937 | [0.907, 0.963] | 0.987 |
| Run 3 · fine-tuning 1 paso | 0.949 | 0.991 | 0.952 | [0.927, 0.975] | 0.989 |
| Run 1 · congelado LR 1e-3 | 0.937 | 0.982 | 0.916 | [0.880, 0.947] | 0.979 |
| Var. B · Run 1 sin flip | 0.933 | 0.981 | 0.917 | [0.881, 0.947] | 0.979 |
| Run 2 · congelado LR 1e-4, DO 0.5 | 0.909 | 0.967 | 0.889 | [0.851, 0.924] | 0.963 |

**Modelo elegido en test (302 imágenes: 156 PNEUMONIA / 146 NORMAL):**

| Umbral | Accuracy | Precision | Recall (sensibilidad) | Especificidad | F1 | FN | FP |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.934 | 0.925 | 0.949 | 0.918 | 0.937 | 8 | 12 |
| **0.585 (clínico, fijado en val)** | **0.937** | **0.936** | **0.942** | **0.932** | **0.939** | 9 | 10 |

Lectura honesta:
- En test, el Run 3 queda +0.015 de F1 por encima, pero el bootstrap pareado da un IC 95 % de
  [-0.004, +0.035]: con 302 imágenes la diferencia **no es distinguible**. Se mantiene la
  elección hecha con validación (cambiarla después de ver el test sería sobreajustar al test).
- La Variante A parte de las 10 épocas del Run 1: acumula 20 épocas en total frente a 10 del
  Run 3. Parte de su ventaja en validación puede venir de ese cómputo extra, no solo del método.
- La Variante B (sin flip) no mejoró al Run 1 (F1 val 0.933 vs 0.937): el flip no perjudica
  de forma medible en este volumen de datos. Se documenta como resultado negativo.
- Todos los runs alcanzaron su mejor `val_loss` en la última época (10/10): con más
  épocas probablemente mejorarían.
- 7 de los 9 falsos negativos con umbral clínico son neumonías **virales** (61 de 156 en test):
  los patrones intersticiales son más sutiles que las consolidaciones bacterianas.

---

## 1. Qué cambió respecto a la versión de clase (y por qué)

La base es el AI-XRay dictado en clase. Sobre esa base, esta instancia corrige errores
y mejora la metodología:

| # | Problema en la versión de clase | Qué se hizo | Dónde |
|---|---|---|---|
| 1 | Las métricas "finales" se calculaban sobre **validación**, el mismo set que usan EarlyStopping, ReduceLROnPlateau y la selección del mejor run → cifras optimistas | Cada run se evalúa en val (para **elegir**) y en **test** (solo para **reportar**) | `src/train.py` |
| 2 | El "fine-tuning" (Run 3) empezaba con una cabeza **aleatoria**: no era el entrenamiento en dos fases que describe el README | **Variante A**: fine-tuning que parte del modelo ya entrenado del Run 1 | `src/architecture.py::unfreeze_for_finetuning`, `03_training.ipynb` |
| 3 | Runs comparados con 3 épocas: el Run 2 seguía mejorando en la última → comparación injusta | Mismo presupuesto para todos: 10 épocas + EarlyStopping(patience=4) | `03_training.ipynb` |
| 4 | Umbral fijo de 0.5 en un problema donde un falso negativo pesa más | **Umbral clínico**: el más alto que, en validación, logra sensibilidad ≥ 95 %; se reporta en test | `src/metrics.py::find_threshold_for_recall` |
| 5 | Las imágenes NORMAL no se agrupaban por paciente | Se agrupan por prefijo `IM-XXXX` (75 estudios con varias imágenes; con el método original, 29 habrían quedado partidos entre splits) | `src/dataset.py` |
| 6 | El paso de PySpark **fallaba**: la UDF importaba `src` dentro de los workers de Spark (`ModuleNotFoundError`) | Reescrito con expresiones nativas (`regexp_extract`), verificado fila a fila contra pandas | `src/dataset.py::build_manifest_spark` |
| 7 | La API redimensionaba con PIL (bicúbico) y el entrenamiento con `tf.image.resize` (bilineal): *training-serving skew* | La API usa la misma función de preprocesamiento que el entrenamiento (diferencia verificada = 0.0) | `src/preprocessing.py::preprocess_image_bytes` |
| 8 | `MODEL_SOURCE = "local"` fijo en el código: `AIXRAY_MODEL_SOURCE` / `AIXRAY_MODEL_PATH` nunca se leían | Configuración real por variables de entorno + umbral desde `model_card.json` | `api/model_service.py` |
| 9 | `async def predict` llamaba a TensorFlow (bloqueante) dentro del event loop; cualquier fallo devolvía 400 | `def` (threadpool) y códigos correctos: 400 / 413 / 415 / 500 / 503 | `api/main.py` |
| 10 | `tests/test_api.py` no existía aunque el README lo citaba | 10 pruebas con pytest (contrato, errores HTTP y consistencia API = evaluación offline) | `tests/test_api.py` |
| 11 | `transition_model_version_stage` está deprecado en MLflow 3 y Unity Catalog no soporta stages | Stage `Staging` (lo pide la rúbrica) **y** alias `staging` (lo que carga la API) | `04_evaluation.ipynb` |
| 12 | Los artefactos de MLflow quedaban en `notebooks/mlruns/` (ruta relativa al notebook) | Ubicación fija en `AI-XRay/mlruns` vía `setup_mlflow()` | `src/config.py` |
| 13 | Evaluar con `model(x)` en modo eager, lote a lote, consumía ~6 GB de RAM en CPU | `model.predict` / `predict_on_batch` (1.3 GB) | `src/train.py`, `api/model_service.py` |

---

## 2. El flujo de punta a punta

```text
Dataset (Kaggle, 5,856 imágenes)
   ↓
Manifiesto + EDA (pandas + PySpark)  →  verificación de fuga de datos (PNEUMONIA y NORMAL)
   ↓
Subconjunto balanceado de 2,000 imágenes  →  split propio 70/15/15 agrupado por paciente
   ↓
Preprocesamiento + Data Augmentation (tf.data)
   ↓
CNN + Transfer Learning (ResNet50): 3 runs de referencia + 2 variantes propias
   ↓
Evaluación en validación (selección) y test (reporte) + umbral clínico
   ↓
MLflow: params, métricas por época, artefactos, modelo con signature, Model Registry
   ↓
models/aixray_model.keras + models/model_card.json
   ↓
API de inferencia (FastAPI: POST /predict, GET /health)
```

El entregable de "aplicación" de la rúbrica es la API, que se prueba en `/docs` (Swagger),
con `curl` o con `pytest`. Además hay una demo web que corre el mismo modelo en el navegador
(sección 6b).

---

## 3. Datos

Ver [`data/README.md`](data/README.md): fuente, cita obligatoria, cómo descargarlo y
dónde colocarlo.

- 5,856 imágenes (1,583 NORMAL / 4,273 PNEUMONIA), desbalance 2.7:1.
- **Fuga de datos en el split original:** 170 pacientes con neumonía tienen imágenes en
  `train/` y en `test/`. **320 de las 390** imágenes de neumonía del test original son de
  pacientes que también están en train.
- Split propio 70/15/15 agrupado por paciente (PNEUMONIA) y por estudio (NORMAL) sobre un
  subconjunto balanceado de 2,000 imágenes: train 1,389 · val 309 · test 302. El
  `assert` de `01_exploracion.ipynb` verifica que ningún paciente queda en dos splits.

---

## 4. Modelo y experimentos

```text
Imagen → resize 224×224 (bilineal, antialias) → preprocess_input (ImageNet, NO 0-1)
       → [solo train] augmentation: flip*, rotación ±5 %, zoom ±10 %, contraste ±10 %
ResNet50 (ImageNet, sin top) → GlobalAveragePooling2D → Dropout → Dense(1, sigmoid)
```

| Run | Tipo | Configuración | Justificación |
|---|---|---|---|
| Run 1 | Referencia | Base congelada · LR 1e-3 · dropout 0.3 | Línea base |
| Run 2 | Referencia | Base congelada · LR 1e-4 · dropout 0.5 | Más regularización |
| Run 3 | Referencia | Fine-tuning desde capa 143, cabeza aleatoria · LR 1e-5 | Fine-tuning "en un paso" |
| **Var. A** | **Propia** | Fine-tuning desde capa 143 **partiendo del modelo del Run 1** · LR 1e-5 | Transfer learning en dos fases real |
| **Var. B** | **Propia** | Run 1 **sin flip horizontal** (único cambio) | El flip invierte la anatomía (corazón a la derecha) y espeja el marcador "R" de la placa |

Todos: Adam, batch 32, 10 épocas máx., EarlyStopping(val_loss, patience=4,
restore_best_weights), ReduceLROnPlateau(factor 0.5, patience 2), semilla 42.

**Criterio de selección** (fijado antes de mirar el test): mayor F1 en validación; si
dos runs están a menos de 0.01, desempata el AUC de validación.

---

## 5. MLflow

`Experiment: AI-XRay`. Cada run registra:

- **Parámetros:** arquitectura, LR, dropout, épocas, batch, optimizador, capa de
  fine-tuning, run de origen (Var. A), semilla, `class_weight`, parámetros entrenables.
- **Tags:** `tipo` (referencia / variante_propia), `fase` y **`justificacion`** (texto).
- **Métricas por época:** `epoch_loss`, `epoch_accuracy`, `epoch_val_loss`,
  `epoch_val_accuracy`, `epoch_val_auc`.
- **Métricas finales:** validación (`accuracy`, `precision`, `recall`, `specificity`,
  `f1_score`, `auc`, `tp/fp/tn/fn`), test (`test_*`), umbral clínico
  (`threshold_clinical`, `clin_val_*`, `clin_test_*`), `best_epoch`, `train_minutes`.
- **Artefactos:** curvas de entrenamiento, matrices de confusión (val, test, test con
  umbral clínico), curvas ROC (val y test), classification report, `history.json`,
  `predictions.npz`.
- **Modelo** con *signature* (+ `input_example_info.json`: workaround al bug de
  `input_example` con MLflow 3 + Keras 3).

El mejor run se registra como `aixray_pneumonia_classifier`, stage **Staging** + alias
**`staging`**, con tags `threshold_clinical` y `selection_criterion`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```

> **Si mueves la carpeta del proyecto** (o la copias a otro computador), MLflow sigue
> apuntando a la ruta absoluta antigua de los artefactos. Corre una vez
> `python scripts/relocate_mlflow.py` desde `AI-XRay/` y vuelve a abrir `mlflow ui`.

---

## 6. API (FastAPI)

```bash
uvicorn api.main:app --reload --port 8000             # http://localhost:8000/docs
curl -X POST "http://localhost:8000/predict" -F "file=@ruta/a/radiografia.jpeg"
```

| Endpoint | Método | Respuesta |
|---|---|---|
| `/health` | GET | `{"status": "ok", "model_loaded": true, "model_version": "aixray_pneumonia_classifier v1 (local)", "threshold": 0.5853}` |
| `/predict` | POST | `{"prediction": "PNEUMONIA", "probability": 0.9998, "threshold": 0.5853, "model_version": "...", "disclaimer": "..."}` |
| `/docs` | GET | Swagger UI |

| Código | Cuándo |
|---|---|
| 400 | archivo vacío o que no es una imagen decodificable |
| 413 | archivo > 10 MB |
| 415 | tipo distinto de JPEG/PNG |
| 503 | modelo no disponible |
| 500 | error interno durante la inferencia |

Variables de entorno: `AIXRAY_MODEL_SOURCE` (`local` | `mlflow`), `AIXRAY_MODEL_PATH`,
`AIXRAY_THRESHOLD` (por defecto, el umbral clínico guardado en `models/model_card.json`).

```bash
pytest tests/ -v
```

---

## 6b. Demo en el navegador (`web/`)

Publicada con GitHub Pages: https://cesarespa.github.io/AI-Xray-Pneumonia/

- **Sin servidor:** el modelo se convierte a ONNX y corre con `onnxruntime-web`. La radiografía
  no sale del equipo de quien la sube, y la demo no se "duerme" como un servidor gratuito.
- **Mismo preprocesamiento:** `web/app.js` reimplementa `preprocess_image_bytes` (redimensionado
  bilineal con antialias de `tf.image.resize` y `preprocess_input` de ResNet50).
- **Mapa de activación (CAM):** con `GlobalAveragePooling2D -> Dense` el mapa es exacto: el promedio
  de las 7x7 contribuciones más el sesgo es el logit. Muestra en qué zonas se apoyó el modelo, no
  dónde está la lesión.
- **Pesos en float16** (47 MB en lugar de 94 MB).

Verificación, con `scripts/web/validate.py` sobre las 302 imágenes de test:

| Versión | Diferencia máxima de probabilidad vs. Keras | Decisiones distintas |
|---|---|---|
| ONNX float32 | 0.000002 | 0 |
| ONNX float16 (la publicada) | 0.006 | 0 |

En el navegador, sobre los 6 ejemplos de la demo, la diferencia máxima frente a TensorFlow fue 0.0007.
Con las JPEG originales del dataset (el entrenamiento usó copias reducidas a 448 px), el modelo da
F1 0.936 y 10 falsos negativos, frente a 0.939 y 9 del reporte oficial.

Regenerar (requiere además `tf2onnx onnx onnxruntime onnxconverter-common`):

```bash
python scripts/web/export_onnx.py . <salida>
python scripts/web/validate.py . "<dataset>/Pediatric Chest X-ray Pneumonia" <salida>
python scripts/web/examples.py . <salida>      # copia el modelo y los ejemplos a web/
```

---

## 7. Estructura

```text
AI-XRay/
├── data/
│   ├── README.md              fuente, cita, descarga, fuga de datos
│   ├── manifest.csv           manifiesto completo (5,856 filas)
│   └── manifest_split.csv     subconjunto con split propio (2,000 filas)
├── notebooks/
│   ├── 01_exploracion.ipynb   manifiesto, PySpark, fuga de datos, split
│   ├── 02_preprocessing.ipynb tf.data, augmentation, verificación API = entrenamiento
│   ├── 03_training.ipynb      5 runs en MLflow (3 referencia + 2 propias)
│   └── 04_evaluation.ipynb    comparación, Registry, export, test, análisis de errores
├── src/
│   ├── config.py              rutas, constantes, setup_mlflow()
│   ├── dataset.py             manifiesto (pandas y Spark), fuga, subconjunto, split
│   ├── preprocessing.py       tf.data + preprocess_image_bytes (compartido con la API)
│   ├── architecture.py        ResNet50 + transfer learning, unfreeze_for_finetuning
│   ├── train.py               train_and_log: entrena, evalúa (val+test) y registra
│   └── metrics.py             métricas, umbral clínico, figuras
├── api/                       main.py · schemas.py · model_service.py
├── tests/test_api.py
├── scripts/relocate_mlflow.py · scripts/web/ (exportación a ONNX y verificación)
├── web/                       demo en el navegador (GitHub Pages)
├── models/                    aixray_model.keras · model_card.json
├── reports/                   tablas y figuras del Executive Summary
├── mlflow.db · mlruns/        tracking de los 5 runs y Model Registry
└── requirements.txt
```

---

## 8. Fase 1 (local) vs. Fase 2 (Databricks)

El sistema está construido completo **en local**. Lo único escrito "a la manera de
Databricks" es el manifiesto con PySpark (`build_manifest_spark`). La Fase 2 no se
implementa; queda documentada:

| Etapa (Fase 1, local) | Equivalente en Databricks (Fase 2) | Qué cambia |
|---|---|---|
| `build_manifest()` (pandas) / `build_manifest_spark()` sobre carpeta local | `build_manifest_spark()` leyendo de un **Volume** de Unity Catalog o del Data Lake (ADLS/S3/GCS) | Solo la ruta de origen y el motor (cluster). Al usar expresiones nativas, sin UDF de Python, escala sin serializar filas a Python |
| `data/manifest_split.csv` | Tabla Delta `catalogo.esquema.aixray_manifest` | Consultable con SQL, versionada (time travel) |
| Notebooks en Jupyter local | Los mismos notebooks en un Workspace, en un cluster o en Serverless | El código no cambia |
| `sqlite:///mlflow.db` | Tracking gestionado (`mlflow.set_experiment("/Shared/aixray")`) | Experimentos visibles para el equipo |
| Model Registry local con stage + alias | Model Registry en **Unity Catalog** (`catalogo.esquema.aixray_pneumonia_classifier`) | UC **solo admite aliases**: por eso la API ya carga por alias (`@staging`) |
| Entrenamiento en un proceso (CPU) | Cluster de un nodo con GPU; distribuido (`TorchDistributor`/`spark-tensorflow-distributor`) solo si el volumen lo exige | Se justifica a partir de decenas de miles de imágenes |
| API con `uvicorn` | Contenedor (Docker) en servidor propio o Databricks Model Serving | El contrato `/predict` no cambia |

**Cuándo migrar:** cuando el cuello de botella sea el volumen de datos (cientos de miles de
imágenes) o la colaboración de un equipo sobre el mismo pipeline. Con 2,000-6,000
imágenes, Databricks añade complejidad sin beneficio proporcional.

---

## 9. Cómo correrlo

```bash
# 1. Entorno (Python 3.10+)
python -m venv .venv && .venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 2. Dataset en ../archive (2)/  (ver data/README.md) o AIXRAY_DATA_DIR=/ruta

# 3. Notebooks en orden (01 regenera el MISMO split con las rutas de tu máquina)
jupyter notebook notebooks/01_exploracion.ipynb
#   02_preprocessing → 03_training (≈2 h en CPU) → 04_evaluation

# 4. MLflow y API
python scripts/relocate_mlflow.py     # solo si usas el mlflow.db incluido en otra carpeta
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn api.main:app --reload --port 8000
pytest tests/ -v
```

Para usar la API con el modelo ya entrenado basta con los pasos 1 y 4. Los notebooks
incluidos ya están ejecutados.

> **Nota sobre la ejecución incluida.** Los notebooks, el `mlflow.db` y los modelos que vienen
> en esta carpeta se ejecutaron en un entorno Linux en la nube (2 CPU, TensorFlow 2.21,
> MLflow 3.15.1, ≈2 h de entrenamiento). Por eso las salidas muestran rutas `/home/claude/...` y
> `data/manifest*.csv` apunta a esas rutas. Para transferir el dataset, las imágenes se
> redujeron antes a 448 px por el lado mayor (PNG, sin pérdida); el pipeline las lleva a
> 224×224 igual que con los JPEG originales. Al correr `01_exploracion.ipynb` en otra máquina
> se regenera **exactamente el mismo split** (verificado) con las rutas locales.

---

## Referencias

- Kermany D, Goldbaum M, Cai W, et al. *Identifying Medical Diagnoses and Treatable
  Diseases by Image-Based Deep Learning*. Cell. 2018;172(5):1122-1131.
  doi:10.1016/j.cell.2018.02.010
- Dataset: [Pediatric Chest X-ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)
- [ResNet50 — Keras Applications](https://keras.io/api/applications/resnet/)
- [MLflow — Model Registry (aliases)](https://mlflow.org/docs/latest/model-registry.html)
- [FastAPI](https://fastapi.tiangolo.com/)
