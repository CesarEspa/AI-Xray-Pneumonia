"""Convierte el modelo de AI-XRay a ONNX para correrlo en el navegador.

Salidas del modelo exportado:
  prob  (1, 1)        probabilidad de PNEUMONIA (igual que el .keras)
  cam   (1, 7, 7, 1)  mapa de activación de clase: contribución de cada zona al logit.
                      Con GAP -> Dense, el promedio del mapa + el sesgo es exactamente el logit.

Uso: python export_onnx.py <carpeta AI-XRay> <carpeta de salida>
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx

project, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(project / "models" / "aixray_model.keras", compile=False)
model.summary(print_fn=lambda s: None)
names = [l.name for l in model.layers]
print("capas:", names)

base = model.get_layer("resnet50")
gap = next(l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
dense = model.get_layer("prediction")
w, b = dense.get_weights()  # (2048, 1), (1,)
print("dense:", w.shape, b.shape, float(b[0]))

w_np = w[:, 0].astype(np.float32)
b_np = np.float32(b[0])


@tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name="input")])
def serve(x):
    w_t = tf.constant(w_np)  # dentro de la función: queda incrustada como constante del grafo
    b_t = tf.constant(b_np)
    feats = base(x, training=False)                      # (1, 7, 7, 2048)
    logit = tf.reduce_sum(gap(feats) * w_t, axis=-1, keepdims=True) + b_t
    prob = tf.sigmoid(logit)
    cam = tf.expand_dims(tf.tensordot(feats, w_t, axes=[[3], [0]]), -1)  # (1, 7, 7, 1)
    return {"prob": prob, "cam": cam}


# Comprobación: la función exportada reproduce al modelo original.
x = tf.random.uniform([1, 224, 224, 3], -120, 150, seed=1)
ref = float(model(x, training=False).numpy().ravel()[0])
got = serve(x)
cam_mean = float(tf.reduce_mean(got["cam"]).numpy()) + float(b[0])
print("original", ref, "exportada", float(got["prob"].numpy().ravel()[0]),
      "sigmoid(media cam + b)", float(tf.sigmoid(cam_mean).numpy()))

onnx_path = out / "aixray_fp32.onnx"
tf2onnx.convert.from_function(serve, input_signature=serve.input_signature, opset=17, output_path=str(onnx_path))
print("onnx fp32:", onnx_path.stat().st_size)
import onnx as _o
print("entradas:", [i.name for i in _o.load(str(onnx_path)).graph.input], "salidas:", [o.name for o in _o.load(str(onnx_path)).graph.output])

import onnx
from onnxconverter_common import float16

m = onnx.load(str(onnx_path))
m16 = float16.convert_float_to_float16(m, keep_io_types=True)
onnx.save(m16, str(out / "aixray_fp16.onnx"))
print("onnx fp16:", (out / "aixray_fp16.onnx").stat().st_size)

card = json.loads((project / "models" / "model_card.json").read_text(encoding="utf-8"))
(out / "dense_bias.json").write_text(json.dumps({"bias": float(b[0]), "threshold": card["threshold_clinical"]}), encoding="utf-8")
