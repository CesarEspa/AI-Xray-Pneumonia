"""Prepara el laboratorio de entrenamiento de la demo web.

La ResNet50 de los runs "congelados" (Run 1, Run 2, Var. B) son los pesos de ImageNet sin cambios, así que
su salida para cada imagen es fija. Este script:
  1. exporta esa base congelada a web/model/base.onnx (float16), con salida (1, 7, 7, 2048);
  2. calcula las características (GlobalAveragePooling, 2048 valores) de las 2.000 imágenes del split
     (train/val/test) y las guarda en web/lab/features.bin (float16);
  3. guarda en web/lab/meta.json las etiquetas, los pesos de la cabeza del Run 1 oficial y la verificación.
En el navegador se entrena solo la cabeza (Dropout -> Dense) sobre esas características, igual que el Run 1
pero sin aumento de datos.

Uso: python scripts/web/export_lab.py <carpeta dataset "Pediatric Chest X-ray Pneumonia"> <carpeta temporal>
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
from tensorflow.keras.applications.resnet50 import preprocess_input

PROJECT = Path(__file__).resolve().parents[2]
DATA, TMP = Path(sys.argv[1]), Path(sys.argv[2])
WEB = PROJECT / "web"
(WEB / "lab").mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)


def preprocess(raw: bytes) -> np.ndarray:  # = src/preprocessing.py::preprocess_image_bytes
    img = tf.io.decode_image(tf.constant(raw), channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224), antialias=True)
    return preprocess_input(img).numpy()


db = sqlite3.connect(PROJECT / "mlflow.db")
rid = next(r for r, n in db.execute("select run_uuid, name from runs") if n.startswith("ref1"))
mid = next(m for m, r in db.execute("select model_id, source_run_id from logged_models") if r == rid)
metrics = dict(db.execute("select key, value from latest_metrics where run_uuid=?", (rid,)))
model = tf.keras.models.load_model(PROJECT / "mlruns" / "models" / mid / "artifacts" / "data" / "model.keras", compile=False)
base, dense = model.get_layer("resnet50"), model.get_layer("prediction")
w, b = dense.get_weights()

# Comprobación: la base del Run 1 es exactamente ResNet50 de ImageNet (estaba congelada).
ref = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
same = all(np.array_equal(a, c) for a, c in zip(base.get_weights(), ref.get_weights()))
print("base del Run 1 == ImageNet:", same, flush=True)


@tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name="input")])
def serve(x):
    return {"feats": base(x, training=False)}


fp32 = TMP / "base_fp32.onnx"
tf2onnx.convert.from_function(serve, input_signature=serve.input_signature, opset=17, output_path=str(fp32))
onnx.save(float16.convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(WEB / "model" / "base.onnx"))

split = pd.read_csv(PROJECT / "data" / "manifest_split.csv")


def local(r):
    for ext in (".jpeg", ".jpg", ".png"):
        p = DATA / r.origin_split / r.label_name / (Path(r.filename).stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(r.filename)


parts, labels, counts = [], {}, {}
for name in ("train", "val", "test"):
    sub = split[split.split == name].reset_index(drop=True)
    xs = np.stack([preprocess(local(r).read_bytes()) for r in sub.itertuples()])
    feats = tf.reduce_mean(base.predict(xs, batch_size=32, verbose=0), axis=[1, 2]).numpy()
    parts.append(feats)
    labels[name] = sub.label.astype(int).tolist()
    counts[name] = len(sub)
    print(name, feats.shape, flush=True)
    if name == "test":
        test_x, test_feats = xs, feats

allf = np.concatenate(parts).astype(np.float16)
allf.tofile(WEB / "lab" / "features.bin")

# Verificación 1: cabeza del Run 1 sobre las características == modelo Run 1 completo.
logit = test_feats @ w[:, 0] + b[0]
p_head = 1 / (1 + np.exp(-logit))
p_full = model.predict(test_x, batch_size=32, verbose=0).ravel()
# Verificación 2: características guardadas en float16 y base ONNX float16 frente a Keras.
f16 = allf[-counts["test"]:].astype(np.float32)
p_f16 = 1 / (1 + np.exp(-(f16 @ w[:, 0] + b[0])))
sess = ort.InferenceSession(str(WEB / "model" / "base.onnx"), providers=["CPUExecutionProvider"])
p_onnx = np.array([1 / (1 + np.exp(-(sess.run(["feats"], {"input": test_x[i:i + 1]})[0].mean(axis=(1, 2))[0] @ w[:, 0] + b[0])))
                   for i in range(40)])
thr = float(metrics["threshold_clinical"])
y = np.array(labels["test"])
pred = p_f16 >= thr
tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
f1 = 2 * tp / (2 * tp + fp + fn)

meta = {
    "dims": int(allf.shape[1]), "counts": counts, "labels": labels,
    "class_weight": {k: float(v) for k, v in zip((0, 1), np.bincount(labels["train"]).sum() / (2 * np.bincount(labels["train"])))},
    "run1": {"w": [round(float(v), 6) for v in w[:, 0]], "b": float(b[0]), "threshold": thr,
             "official_test_f1": metrics["clin_test_f1_score"], "official_val_f1": metrics["clin_val_f1_score"],
             "local_test": {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "f1": f1}},
    "verification": {
        "base_is_imagenet": bool(same),
        "head_vs_full_model_max_diff": float(np.abs(p_head - p_full).max()),
        "float16_features_max_diff": float(np.abs(p_f16 - p_full).max()),
        "onnx_base_max_diff_40_images": float(np.abs(p_onnx - p_full[:40]).max()),
    },
}
(WEB / "lab" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
print(json.dumps({k: v for k, v in meta.items() if k not in ("labels",)} | {"run1": {k: v for k, v in meta["run1"].items() if k != "w"}}, indent=2))
print("features.bin", (WEB / "lab" / "features.bin").stat().st_size, "base.onnx", (WEB / "model" / "base.onnx").stat().st_size)
