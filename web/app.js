/* AI-XRay en el navegador.
 *
 * Reproduce el preprocesamiento de src/preprocessing.py::preprocess_image_bytes:
 *   decodificar a RGB -> tf.image.resize(224x224, bilineal, antialias=True) -> preprocess_input
 *   de ResNet50 (RGB -> BGR y resta de la media de ImageNet, sin escalar a 0-1).
 * El modelo ONNX (la Var. A, el elegido por validación) devuelve la probabilidad de neumonía y el mapa de
 * activación de clase (7x7), que con GlobalAveragePooling + Dense es exacto: promedio del mapa + sesgo = logit.
 * Su umbral clínico y las métricas de los 5 modelos del proyecto vienen de model/models.json
 * (generado por scripts/web/export_all.py).
 */
(() => {
  const SIZE = 224;
  const MEAN_BGR = [103.939, 116.779, 123.68];
  const MAX_BYTES = 10 * 1024 * 1024;       // mismo límite que la API
  const CACHE = 'aixray-models-v1';

  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 3) => v.toLocaleString('es-CO', { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (v) => `${fmt(v * 100, 1)}%`;
  // Nunca mostrar 1,000 ni 0,000: el modelo no da certeza absoluta.
  const fmtProb = (p) => (p > 0.999 ? 'más de 0,999' : p < 0.001 ? 'menos de 0,001' : fmt(p));
  const mb = (b) => `${Math.round(b / 1048576)} MB`;

  let models = [];            // los 5 del proyecto, de models.json (solo para la tabla)
  let current = null;         // el modelo de la demo (Var. A)
  const sessions = new Map(); // archivo -> InferenceSession
  let testItems = [];
  let busy = false;
  let last = null;            // { input, img, example }
  let score = { hits: 0, total: 0 };

  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

  /* ---------- descarga de modelos (con progreso y caché del navegador) ---------- */
  async function fetchModel(m, onProgress) {
    let cache = null;
    try { cache = await caches.open(CACHE); } catch (_) { /* sin Cache Storage: se descarga cada vez */ }
    if (cache) {
      const hit = await cache.match(m.file);
      if (hit) return new Uint8Array(await hit.arrayBuffer());
    }
    const res = await fetch(m.file);
    if (!res.ok) throw new Error(`No se pudo descargar ${m.file} (${res.status})`);
    const total = Number(res.headers.get('content-length')) || m.size || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onProgress(got, total);
    }
    const buf = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    if (cache) { try { await cache.put(m.file, new Response(buf)); } catch (_) { /* cuota llena: no pasa nada */ } }
    return buf;
  }

  async function getSession(m, statusPrefix = '') {
    if (sessions.has(m.file)) return sessions.get(m.file);
    const bar = $('progress');
    bar.classList.remove('done');
    const buf = await fetchModel(m, (got, total) => {
      $('status').textContent = `${statusPrefix}Descargando el modelo · ${mb(got)}${total ? ` de ${mb(total)}` : ''}`;
      if (total) bar.firstElementChild.style.width = `${(100 * got / total).toFixed(1)}%`;
    });
    $('status').textContent = `${statusPrefix}Preparando el modelo…`;
    const s = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    sessions.set(m.file, s);
    bar.classList.add('done');
    return s;
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
    const tmp = new Float32Array(h * SIZE * 3);   // pasada horizontal: (h, 224, 3)
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
    const out = new Float32Array(SIZE * SIZE * 3); // pasada vertical + preprocess_input (BGR, media de ImageNet)
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
    return { bmp, data: ctx.getImageData(0, 0, c.width, c.height).data, w: c.width, h: c.height };
  }

  /* ---------- inferencia ---------- */
  async function infer(m, session, input) {
    const t0 = performance.now();
    const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, SIZE, SIZE, 3]) });
    const ts = Object.values(out);
    return {
      prob: ts.find((t) => t.data.length === 1).data[0],
      cam: Float32Array.from(ts.find((t) => t.data.length === 49).data),
      ms: performance.now() - t0,
    };
  }

  async function predict(blob, key) {   // usado también para verificar la web contra TensorFlow
    const m = key ? models.find((x) => x.key === key) : current;
    const img = await decode(blob);
    const input = preprocess(img.data, img.w, img.h);
    return { img, input, ...(await infer(m, await getSession(m), input)) };
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
    if (!cam || !$('showCam').checked) return;
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

  const realLabel = (ex) => (ex.label === 'PNEUMONIA' ? `neumonía ${ex.kind === 'virus' ? 'viral' : 'bacteriana'}` : 'normal');

  function show(r, example) {
    const m = current;
    $('empty').hidden = true;
    $('result').hidden = false;
    drawImage(r.img.bmp);
    last.cam = r.cam;
    drawCam(r.cam);

    const pos = r.prob >= m.threshold;
    const v = $('verdict');
    v.textContent = pos ? 'Compatible con neumonía' : 'Sin signos de neumonía';
    v.className = pos ? 'pos' : 'neg';
    $('fill').style.width = `${(r.prob * 100).toFixed(2)}%`;
    $('tick').style.left = `${(m.threshold * 100).toFixed(2)}%`;
    $('probText').textContent = `p = ${fmtProb(r.prob)}`;
    $('meter').setAttribute('aria-label', `Probabilidad de neumonía ${fmtProb(r.prob)}, umbral ${fmt(m.threshold)}`);

    let text = `El modelo estima una probabilidad de neumonía de ${fmtProb(r.prob)}, ${pos ? 'por encima' : 'por debajo'} del umbral de ${fmt(m.threshold)}.`;
    if (Math.abs(r.prob - m.threshold) < 0.15) text += ' Está cerca del umbral, así que es un caso dudoso para el modelo.';
    $('explain').textContent = text;
    $('camNote').textContent = Math.max(...r.cam) > 0
      ? 'El mapa resalta las zonas que más empujaron la predicción hacia neumonía. Muestra en qué se apoyó el modelo, no dónde está la lesión.'
      : 'Ninguna zona empujó la predicción hacia neumonía, por eso el mapa está vacío.';

    const truth = $('truth');
    if (example) {
      const hit = pos === (example.label === 'PNEUMONIA');
      truth.textContent = `Diagnóstico real: ${realLabel(example)} · ${hit ? 'acierta' : 'falla'}`;
      truth.style.color = hit ? 'var(--ok)' : 'var(--accent)';
      if (example.random) {
        score.total += 1;
        score.hits += hit ? 1 : 0;
        $('score').textContent = `${score.hits} de ${score.total} aciertos`;
      }
    } else {
      truth.textContent = 'Imagen subida';
      truth.style.color = '';
    }
    $('timing').textContent = `${Math.round(r.ms)} ms en tu navegador`;
  }

  function setBusy(on) {
    busy = on;
    document.querySelectorAll('.ex, #randomBtn').forEach((b) => { b.disabled = on; });
  }

  function showError(msg) {
    $('timing').textContent = '';
    $('empty').hidden = false;
    $('result').hidden = true;
    $('empty').textContent = msg;
  }

  async function classify(blob, example, button) {
    if (busy || !current) return;
    setBusy(true);
    document.querySelectorAll('.ex').forEach((b) => b.classList.toggle('active', b === button));
    $('timing').textContent = 'Analizando…';
    try {
      const img = await decode(blob);
      const input = preprocess(img.data, img.w, img.h);
      last = { input, img, example };
      show({ img, ...(await infer(current, await getSession(current), input)) }, example);
      $('status').textContent = 'Modelo listo';
    } catch (e) {
      console.error(e);
      showError('No se pudo leer esa imagen. Usa un JPEG o PNG.');
    } finally {
      setBusy(false);
    }
  }

  function handleFile(file) {
    if (!file) return;
    if (!/^image\/(jpeg|png)$/.test(file.type)) return showError('Formato no soportado: usa JPEG o PNG.');
    if (file.size > MAX_BYTES) return showError('La imagen supera 10 MB.');
    classify(file, null, null);
  }

  function fillModelsTable() {
    const body = $('modelsTable').querySelector('tbody');
    for (const m of models) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${m.short}${m.chosen ? ' · el de la demo' : ''}</td><td>${fmt(m.val.f1_score)}</td><td>${fmt(m.test.f1_score)}</td>`
        + `<td>${pct(m.test.recall)}</td><td>${pct(m.test.specificity)}</td><td>${fmt(m.threshold)}</td>`;
      body.appendChild(tr);
    }
    const v = current.verification;
    $('verify').textContent = `Comprobado: el modelo, convertido para el navegador, da ${v.decision_changes === 0 ? 'la misma decisión' : 'casi siempre la misma decisión'} `
      + `que su versión original de TensorFlow en las ${v.n} imágenes de test (diferencia máxima de probabilidad: ${fmt(v.max_abs_diff)}). `
      + 'Las cifras son las del proyecto, calculadas con las copias reducidas a 448 px que se usaron al entrenar.';
  }

  /* ---------- carga inicial ---------- */
  async function init() {
    const [md, ex, tj] = await Promise.all([
      fetch('model/models.json').then((r) => r.json()),
      fetch('examples/examples.json').then((r) => r.json()),
      fetch('test/test.json').then((r) => r.json()),
    ]);
    models = md.models.map((m) => ({ ...m, short: m.label.split(' · ')[0] }));
    testItems = tj;

    current = models.find((m) => m.chosen);
    fillModelsTable();
    $('modelInfo').textContent = `Modelo: ${current.label}, el elegido por validación · umbral clínico ${fmt(current.threshold)} · ${mb(current.size)} la primera vez.`;

    const box = $('examples');
    for (const e of ex.examples) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'ex';
      b.disabled = true;
      b.innerHTML = `<img src="examples/${e.file}" alt="" loading="lazy"><span>${e.title}</span>`;
      b.addEventListener('click', async () => classify(await (await fetch(`examples/${e.file}`)).blob(), e, b));
      box.appendChild(b);
    }

    await getSession(current);
    $('status').textContent = 'Modelo listo';
    setBusy(false);
  }

  $('randomBtn').addEventListener('click', async () => {
    const item = testItems[Math.floor(Math.random() * testItems.length)];
    const blob = await (await fetch(`test/${item.file}`)).blob();
    classify(blob, { ...item, random: true }, null);
  });
  $('file').addEventListener('change', (e) => handleFile(e.target.files[0]));
  const drop = $('drop');
  ['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
  $('showCam').addEventListener('change', () => last && drawCam(last.cam));

  init().catch((e) => { $('status').textContent = 'No se pudo cargar el modelo'; console.error(e); });

  window.__aixray = { predict, ready: () => !!current && sessions.has(current.file) };   // usado para verificar contra TensorFlow
})();
