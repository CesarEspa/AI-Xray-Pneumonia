"""Exporta los 5 runs de MLflow a ONNX (float16) para la demo web y los verifica contra Keras.

Convierte y verifica los 5 runs, pero la demo publica solo el elegido (la Var. A) en web/model/var_a.onnx.
En web/model/models.json guarda, para los 5, sus métricas oficiales
(tomadas de mlflow.db), su umbral clínico y el resultado de la verificación en las 302 imágenes
de test. También prepara web/test/ con las 302 imágenes reducidas para la "imagen al azar".

Uso: python scripts/web/export_all.py <carpeta dataset "Pediatric Chest X-ray Pneumonia"> <carpeta temporal>
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import tensorflow as tf
import tf2onnx
from onnxconverter_common import float16
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

PROJECT = Path(__file__).resolve().parents[2]
DATA, TMP = Path(sys.argv[1]), Path(sys.argv[2])
WEB = PROJECT / "web"
(WEB / "model").mkdir(parents=True, exist_ok=True)
(WEB / "test").mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

RUNS = [  # clave, nombre en MLflow, etiqueta, descripción corta
    ("var_a", "varA_finetune_2fases_lr1e-5", "Var. A · fine-tuning en 2 fases",
     "El modelo elegido. Parte del Run 1 ya entrenado y ajusta las últimas capas de ResNet50."),
    ("run3", "ref3_finetune_1paso_lr1e-5", "Run 3 · fine-tuning en 1 paso",
     "Ajusta las últimas capas de ResNet50 desde una cabeza aleatoria."),
    ("run1", "ref1_frozen_lr1e-3_do0.3", "Run 1 · base congelada",
     "Línea base: ResNet50 congelada, solo se entrena la capa final."),
    ("var_b", "varB_frozen_sin_flip", "Var. B · Run 1 sin flip",
     "Igual al Run 1, pero sin voltear las imágenes al aumentar datos."),
    ("run2", "ref2_frozen_lr1e-4_do0.5", "Run 2 · más regularización",
     "Base congelada con dropout 0,5 y learning rate 10 veces menor."),
]


def preprocess(raw: bytes) -> np.ndarray:  # = src/preprocessing.py::preprocess_image_bytes
    img = tf.io.decode_image(tf.constant(raw), channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224), antialias=True)
    return tf.expand_dims(preprocess_input(img), 0).numpy()


db = sqlite3.connect(PROJECT / "mlflow.db")
run_ids = dict((n, r) for r, n in db.execute("select run_uuid, name from runs"))
model_ids = dict((r, m) for m, r in db.execute("select model_id, source_run_id from logged_models"))

split = pd.read_csv(PROJECT / "data" / "manifest_split.csv")
test = split[split.split == "test"].reset_index(drop=True)


def local(r):
    for ext in (".jpeg", ".jpg", ".png"):
        p = DATA / r.origin_split / r.label_name / (Path(r.filename).stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(r.filename)


test["local"] = test.apply(local, axis=1)
test["kind"] = test.filename.str.extract(r"_(bacteria|virus)_")[0]

# Imágenes de test reducidas para la web (la web las clasifica igual que cualquier imagen subida).
items = []
for i, r in test.iterrows():
    name = f"{i:03d}.jpg"
    im = Image.open(r.local).convert("L")
    im.thumbnail((512, 512), Image.LANCZOS)
    im.save(WEB / "test" / name, "JPEG", quality=85)
    items.append({"file": name, "label": r.label_name, "kind": r.kind if isinstance(r.kind, str) else None})
(WEB / "test" / "test.json").write_text(json.dumps(items), encoding="utf-8")
inputs = [preprocess(p.read_bytes()) for p in test.local]
labels = test.label.values

out_models = []
for key, run_name, label, desc in RUNS:
    rid = run_ids[run_name]
    metrics = dict(db.execute("select key, value from latest_metrics where run_uuid=?", (rid,)))
    params = dict(db.execute("select key, value from params where run_uuid=?", (rid,)))
    src = PROJECT / "mlruns" / "models" / model_ids[rid] / "artifacts" / "data" / "model.keras"
    model = tf.keras.models.load_model(src, compile=False)
    base, dense = model.get_layer("resnet50"), model.get_layer("prediction")
    gap = next(l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
    w, b = dense.get_weights()
    w_np, b_np = w[:, 0].astype(np.float32), np.float32(b[0])

    @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name="input")])
    def serve(x):
        w_t, b_t = tf.constant(w_np), tf.constant(b_np)
        feats = base(x, training=False)
        prob = tf.sigmoid(tf.reduce_sum(gap(feats) * w_t, axis=-1, keepdims=True) + b_t)
        cam = tf.expand_dims(tf.tensordot(feats, w_t, axes=[[3], [0]]), -1)
        return {"prob": prob, "cam": cam}

    fp32 = TMP / f"{key}_fp32.onnx"
    tf2onnx.convert.from_function(serve, input_signature=serve.input_signature, opset=17, output_path=str(fp32))
    target = (WEB / "model" if key == "var_a" else TMP) / f"{key}.onnx"   # la demo publica solo la Var. A
    onnx.save(float16.convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(target))

    sess = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    k_p = np.array([float(model.predict_on_batch(x).ravel()[0]) for x in inputs])
    o_p = np.array([float(sess.run(["prob"], {"input": x})[0].ravel()[0]) for x in inputs])
    thr = float(metrics["threshold_clinical"])
    entry = {
        "key": key, "label": label, "description": desc, "file": f"model/{key}.onnx",
        "size": target.stat().st_size,
        "threshold": thr, "chosen": key == "var_a",
        "learning_rate": params.get("learning_rate"), "dropout": params.get("dropout"),
        "val": {k: metrics[f"clin_val_{k}"] for k in ("f1_score", "recall", "specificity", "auc")},
        "test": {k: metrics[f"clin_test_{k}"] for k in ("f1_score", "recall", "specificity", "auc", "tp", "fp", "tn", "fn")},
        "verification": {"max_abs_diff": float(np.abs(k_p - o_p).max()),
                         "decision_changes": int(((k_p >= thr) != (o_p >= thr)).sum()), "n": len(k_p)},
    }
    out_models.append(entry)
    print(key, entry["verification"], "val F1", round(entry["val"]["f1_score"], 4), "test F1", round(entry["test"]["f1_score"], 4), flush=True)

(WEB / "model" / "models.json").write_text(json.dumps({"models": out_models}, ensure_ascii=False, indent=2), encoding="utf-8")
print("listo")
