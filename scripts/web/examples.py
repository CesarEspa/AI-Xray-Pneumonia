"""Elige ejemplos del conjunto de test, los guarda reducidos para la web y anota la probabilidad
que da el modelo Keras original sobre ESOS archivos (para verificar la web contra TensorFlow).

Uso: python examples.py <carpeta AI-XRay> <carpeta onnx con test_predictions.csv y metrics.json>
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

project, onnx_dir = Path(sys.argv[1]), Path(sys.argv[2])
web = project / "web"
ex_dir = web / "examples"
ex_dir.mkdir(parents=True, exist_ok=True)
(web / "model").mkdir(exist_ok=True)
shutil.copyfile(onnx_dir / "aixray_fp16.onnx", web / "model" / "aixray.onnx")

df = pd.read_csv(onnx_dir / "test_predictions.csv")
m = json.loads((onnx_dir / "metrics.json").read_text())
thr = m["threshold"]
df["kind"] = df.filename.str.extract(r"_(bacteria|virus)_")[0]
df["pred"] = (df.keras >= thr).astype(int)
ok = df[df.pred == df.label]

picks = [
    ("Normal", ok[(ok.label == 0) & (ok.keras < 0.1)].sample(2, random_state=7)),
    ("Neumonía bacteriana", ok[(ok.kind == "bacteria") & (ok.keras > 0.9)].sample(2, random_state=7)),
    ("Neumonía viral", ok[(ok.kind == "virus") & (ok.keras > 0.8)].sample(1, random_state=7)),
    ("Viral · el modelo falla", df[(df.label == 1) & (df.pred == 0) & (df.kind == "virus")].sort_values("keras").head(1)),
]

model = tf.keras.models.load_model(project / "models" / "aixray_model.keras", compile=False)


def prob(path: Path) -> float:
    img = tf.io.decode_image(tf.constant(path.read_bytes()), channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224), antialias=True)
    return float(model.predict_on_batch(tf.expand_dims(preprocess_input(img), 0)).ravel()[0])


examples = []
for title, rows in picks:
    for r in rows.itertuples():
        src = Path(r.local)
        name = src.stem + ".jpg"
        im = Image.open(src).convert("L")
        im.thumbnail((1024, 1024), Image.LANCZOS)
        im.save(ex_dir / name, "JPEG", quality=90)
        examples.append({
            "file": name, "title": title, "label": "PNEUMONIA" if r.label == 1 else "NORMAL",
            "kind": r.kind if isinstance(r.kind, str) else None,
            "expected_prob": round(prob(ex_dir / name), 5),
        })

k = m["keras"]
verification = (
    f"Comprobado: en el navegador da la misma probabilidad que el modelo de TensorFlow original "
    f"(diferencia máxima de {m['fp16_max_abs_diff']:.3f} en las {m['n']} imágenes de test y ninguna decisión distinta). "
    f"Las cifras de arriba son las del proyecto; con las JPEG originales del dataset, en lugar de las copias reducidas "
    f"que se usaron al entrenar, el resultado es casi igual: F1 de {k['f1']:.3f} y {k['fn']} neumonías no detectadas."
).replace(".", ",", 0)
(ex_dir / "examples.json").write_text(json.dumps({"threshold": thr, "verification": verification, "examples": examples},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(examples, ensure_ascii=False, indent=1))
print(verification)
