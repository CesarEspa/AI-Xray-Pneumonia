"""Compara el modelo Keras con las versiones ONNX en las 302 imágenes de test del proyecto.

Uso: python validate.py <carpeta AI-XRay> <carpeta dataset> <carpeta onnx>
Escribe test_predictions.csv y metrics.json en la carpeta onnx.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

project, data_root, onnx_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def preprocess(raw: bytes) -> np.ndarray:  # copia exacta de src/preprocessing.py::preprocess_image_bytes
    img = tf.io.decode_image(tf.constant(raw), channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224), antialias=True)
    return tf.expand_dims(preprocess_input(img), 0).numpy()


split = pd.read_csv(project / "data" / "manifest_split.csv")
test = split[split.split == "test"].copy()


def local(r):
    stem = Path(r.filename).stem
    for ext in (".jpeg", ".jpg", ".png"):
        p = data_root / r.origin_split / r.label_name / (stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(r.filename)


test["local"] = test.apply(local, axis=1)
model = tf.keras.models.load_model(project / "models" / "aixray_model.keras", compile=False)
sessions = {k: ort.InferenceSession(str(onnx_dir / f"aixray_{k}.onnx"), providers=["CPUExecutionProvider"]) for k in ("fp32", "fp16")}
thr = json.loads((onnx_dir / "dense_bias.json").read_text())["threshold"]

rows = []
for r in test.itertuples():
    x = preprocess(r.local.read_bytes())
    row = {"filename": r.filename, "local": str(r.local), "label": int(r.label),
           "keras": float(model.predict_on_batch(x).ravel()[0])}
    for k, s in sessions.items():
        row[k] = float(s.run(["prob"], {s.get_inputs()[0].name: x})[0].ravel()[0])
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(onnx_dir / "test_predictions.csv", index=False)


def metrics(p, y, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    order = np.argsort(-p); yy = y[order]
    pos, neg = yy.sum(), len(yy) - yy.sum()
    auc = float(np.sum(np.cumsum(1 - yy)[yy == 1] * 0) + 0)
    ranks = pd.Series(p).rank().values
    auc = float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": (tp + tn) / len(y), "precision": prec,
            "recall": rec, "specificity": tn / (tn + fp), "f1": 2 * prec * rec / (prec + rec), "auc": auc}


y = df.label.values
out = {"threshold": thr, "n": len(df)}
for k in ("keras", "fp32", "fp16"):
    out[k] = metrics(df[k].values, y, thr)
for k in ("fp32", "fp16"):
    d = np.abs(df[k] - df.keras)
    out[f"{k}_max_abs_diff"] = float(d.max())
    out[f"{k}_decision_changes"] = int(((df[k] >= thr) != (df.keras >= thr)).sum())
(onnx_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
