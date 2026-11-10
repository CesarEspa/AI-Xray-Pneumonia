/* AI-XRay en el navegador.
 *
 * Reproduce el preprocesamiento de src/preprocessing.py::preprocess_image_bytes:
 *   decodificar a RGB -> tf.image.resize(224x224, bilineal, antialias=True) -> preprocess_input
 *   de ResNet50 (RGB -> BGR y resta de la media de ImageNet, sin escalar a 0-1).
 * El modelo ONNX devuelve la probabilidad de neumonía y el mapa de activación de clase (7x7),
 * que con GlobalAveragePooling + Dense es exacto: promedio del mapa + sesgo = logit.
 */
(() => {
  const SIZE = 224;
  const MEAN_BGR = [103.939, 116.779, 123.68];
  const THRESHOLD = 0.5853266716003418;     // umbral clínico de models/model_card.json
  const MODEL_URL = 'model/aixray.onnx';
  const MAX_BYTES = 10 * 1024 * 1024;       // mismo límite que la API

  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 3) => v.toLocaleString('es-CO', { minimumFractionDigits: d, maximumFractionDigits: d });
  // Nunca mostrar 1,000 ni 0,000: el modelo no da certeza absoluta.
  const fmtProb = (p) => (p > 0.999 ? 'más de 0,999' : p < 0.001 ? 'menos de 0,001' : fmt(p));

  let session = null;
  let busy = false;
  let lastCam = null;

  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

  /* ---------- carga del modelo con progreso ---------- */
  async function loadModel() {
    const res = await fetch(MODEL_URL);
    if (!res.ok) throw new Error(`No se pudo descargar el modelo (${res.status})`);
    const total = Number(res.headers.get('content-length')) || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      const mb = (got / 1048576).toFixed(0);
      $('status').textContent = total ? `Descargando modelo · ${mb} de ${(total / 1048576).toFixed(0)} MB` : `Descargando modelo · ${mb} MB`;
      if (total) $('progress').firstElementChild.style.width = `${(100 * got / total).toFixed(1)}%`;
    }
    const buf = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    $('status').textContent = 'Preparando modelo…';
    session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    $('progress').classList.add('done');
    $('status').textContent = 'Modelo listo';
  }

  /* ---------- redimensionado igual a tf.image.resize(antialias=True) ---------- */
  function kernelWeights(inSize, outSize) {
    const scale = outSize / inSize;
    const ks = Math.max(1 / scale, 1);           // ancho del filtro triangular en píxeles de entrada
    const table = [];
    for (let i = 0; i < outSize; i++) {
      const center = (i + 0.5) / scale;
      const lo = Math.max(0, Math.floor(center - ks - 0.5));
      const hi = Math.min(inSize - 1, Math.ceil(center + ks - 0.5));
      const idx = [], w = [];
      let sum = 0;
      for (let j = lo; j <= hi; j++) {
        const wt = 1 - Math.abs((j + 0.5 - center) / ks);
        if (wt > 0) { idx.push(j); w.push(wt); sum += wt; }
      }
      table.push({ idx, w: w.map((v) => v / sum) });
    }
    return table;
  }

  function preprocess(rgba, w, h) {
    const wx = kernelWeights(w, SIZE);
    const wy = kernelWeights(h, SIZE);
    // pasada horizontal: (h, 224, 3)
    const tmp = new Float32Array(h * SIZE * 3);
    for (let y = 0; y < h; y++) {
      const row = y * w * 4;
      for (let x = 0; x < SIZE; x++) {
        const { idx, w: ww } = wx[x];
        let r = 0, g = 0, b = 0;
        for (let k = 0; k < idx.length; k++) {
          const p = row + idx[k] * 4;
          r += rgba[p] * ww[k]; g += rgba[p + 1] * ww[k]; b += rgba[p + 2] * ww[k];
        }
        const o = (y * SIZE + x) * 3;
        tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
      }
    }
    // pasada vertical + preprocess_input (BGR, media de ImageNet)
    const out = new Float32Array(SIZE * SIZE * 3);
    for (let y = 0; y < SIZE; y++) {
      const { idx, w: ww } = wy[y];
      for (let x = 0; x < SIZE; x++) {
        let r = 0, g = 0, b = 0;
        for (let k = 0; k < idx.length; k++) {
          const p = (idx[k] * SIZE + x) * 3;
          r += tmp[p] * ww[k]; g += tmp[p + 1] * ww[k]; b += tmp[p + 2] * ww[k];
        }
        const o = (y * SIZE + x) * 3;
        out[o] = b - MEAN_BGR[0]; out[o + 1] = g - MEAN_BGR[1]; out[o + 2] = r - MEAN_BGR[2];
      }
    }
    return out;
  }

  async function decode(blob) {
    const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none', imageOrientation: 'from-image' });
    const c = document.createElement('canvas');
    c.width = bmp.width; c.height = bmp.height;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0);
    const data = ctx.getImageData(0, 0, c.width, c.height).data;
    return { bmp, data, w: c.width, h: c.height };
  }

  /* ---------- inferencia ---------- */
  async function predict(blob) {
    const img = await decode(blob);
    const t0 = performance.now();
    const input = preprocess(img.data, img.w, img.h);
    const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', input, [1, SIZE, SIZE, 3]) };
    const out = await session.run(feeds);
    const byShape = Object.values(out);
    const probT = byShape.find((t) => t.data.length === 1);
    const camT = byShape.find((t) => t.data.length === 49);
    return { img, prob: probT.data[0], cam: Float32Array.from(camT.data), ms: performance.now() - t0 };
  }

  /* ---------- presentación ---------- */
  function drawImage(bmp) {
    const view = $('view'), heat = $('heat');
    const scale = Math.min(1, 900 / Math.max(bmp.width, bmp.height));
    view.width = heat.width = Math.round(bmp.width * scale);
    view.height = heat.height = Math.round(bmp.height * scale);
    view.getContext('2d').drawImage(bmp, 0, 0, view.width, view.height);
  }

  function color(t) {           // negro -> rojo acento -> amarillo claro
    const stops = [[0, 0, 0], [227, 43, 60], [255, 214, 120]];
    const s = t * 2, i = Math.min(1, Math.floor(s)), f = s - i;
    const a = stops[i], b = stops[i + 1];
    return a.map((v, k) => Math.round(v + (b[k] - v) * f));
  }

  function drawCam(cam) {
    const heat = $('heat');
    const ctx = heat.getContext('2d');
    ctx.clearRect(0, 0, heat.width, heat.height);
    if (!$('showCam').checked) return;
    const max = Math.max(...cam);
    if (max <= 0) return;
    const small = document.createElement('canvas');
    small.width = small.height = 7;
    const sctx = small.getContext('2d');
    const px = sctx.createImageData(7, 7);
    cam.forEach((v, i) => {
      const t = Math.max(0, v) / max;
      const [r, g, b] = color(t);
      px.data.set([r, g, b, Math.round(255 * Math.min(1, t * 1.3))], i * 4);
    });
    sctx.putImageData(px, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.globalAlpha = 0.55;
    ctx.drawImage(small, 0, 0, heat.width, heat.height);
    ctx.globalAlpha = 1;
  }

  function show(r, example) {
    $('empty').hidden = true;
    $('result').hidden = false;
    drawImage(r.img.bmp);
    lastCam = r.cam;
    drawCam(r.cam);

    const pos = r.prob >= THRESHOLD;
    const v = $('verdict');
    v.textContent = pos ? 'Compatible con neumonía' : 'Sin signos de neumonía';
    v.className = pos ? 'pos' : 'neg';
    $('fill').style.width = `${(r.prob * 100).toFixed(2)}%`;
    $('tick').style.left = `${(THRESHOLD * 100).toFixed(2)}%`;
    $('probText').textContent = `p = ${fmtProb(r.prob)}`;
    $('meter').setAttribute('aria-label', `Probabilidad de neumonía ${fmtProb(r.prob)}, umbral ${fmt(THRESHOLD)}`);

    const margin = Math.abs(r.prob - THRESHOLD);
    let text = `El modelo estima una probabilidad de neumonía de ${fmtProb(r.prob)}, ${pos ? 'por encima' : 'por debajo'} del umbral de ${fmt(THRESHOLD)}.`;
    if (margin < 0.15) text += ' Está cerca del umbral, así que es un caso dudoso para el modelo.';
    $('explain').textContent = text;
    $('camNote').textContent = Math.max(...r.cam) > 0
      ? 'El mapa resalta las zonas que más empujaron la predicción hacia neumonía. Muestra en qué se apoyó el modelo, no dónde está la lesión.'
      : 'Ninguna zona empujó la predicción hacia neumonía, por eso el mapa está vacío.';

    const truth = $('truth');
    if (example) {
      const real = example.label === 'PNEUMONIA' ? `neumonía ${example.kind === 'virus' ? 'viral' : 'bacteriana'}` : 'normal';
      const hit = pos === (example.label === 'PNEUMONIA');
      truth.textContent = `Diagnóstico real: ${real} · ${hit ? 'acierta' : 'falla'}`;
      truth.style.color = hit ? 'var(--ok)' : 'var(--accent)';
    } else {
      truth.textContent = 'Imagen subida';
      truth.style.color = '';
    }
    $('timing').textContent = `${Math.round(r.ms)} ms en tu navegador`;
  }

  async function run(blob, example, button) {
    if (busy || !session) return;
    busy = true;
    document.querySelectorAll('.ex').forEach((b) => b.classList.toggle('active', b === button));
    $('timing').textContent = 'Analizando…';
    try {
      show(await predict(blob), example);
    } catch (e) {
      $('timing').textContent = '';
      $('empty').hidden = false;
      $('result').hidden = true;
      $('empty').textContent = 'No se pudo leer esa imagen. Usa un JPEG o PNG.';
      console.error(e);
    } finally {
      busy = false;
    }
  }

  function handleFile(file) {
    if (!file) return;
    if (!/^image\/(jpeg|png)$/.test(file.type)) { $('empty').hidden = false; $('result').hidden = true; $('empty').textContent = 'Formato no soportado: usa JPEG o PNG.'; return; }
    if (file.size > MAX_BYTES) { $('empty').hidden = false; $('result').hidden = true; $('empty').textContent = 'La imagen supera 10 MB.'; return; }
    run(file, null, null);
  }

  /* ---------- ejemplos ---------- */
  async function loadExamples() {
    const data = await (await fetch('examples/examples.json')).json();
    const box = $('examples');
    for (const ex of data.examples) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'ex';
      b.disabled = true;
      b.innerHTML = `<img src="examples/${ex.file}" alt="" loading="lazy"><span>${ex.title}</span>`;
      b.addEventListener('click', async () => run(await (await fetch(`examples/${ex.file}`)).blob(), ex, b));
      box.appendChild(b);
    }
    if (data.verification) $('verify').textContent = data.verification;
  }

  /* ---------- eventos ---------- */
  $('file').addEventListener('change', (e) => handleFile(e.target.files[0]));
  const drop = $('drop');
  ['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
  $('showCam').addEventListener('change', () => lastCam && drawCam(lastCam));

  Promise.all([loadExamples(), loadModel()])
    .then(() => document.querySelectorAll('.ex').forEach((b) => { b.disabled = false; }))
    .catch((e) => { $('status').textContent = 'No se pudo cargar el modelo'; console.error(e); });

  window.__aixray = { predict, ready: () => !!session };   // usado para verificar contra TensorFlow
})();
