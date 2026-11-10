/* Laboratorio: entrena en el navegador la cabeza del Run 1 (Dropout -> Dense(1, sigmoid)).
 *
 * Las características (GlobalAveragePooling de ResNet50 de ImageNet, 2048 valores por imagen) ya vienen
 * calculadas en lab/features.bin (float16) por scripts/web/export_lab.py. Aquí se reproduce lo que hace
 * src/train.py con la base congelada: Adam (eps 1e-7 como Keras), lotes de 32, class_weight balanceado,
 * EarlyStopping(val_loss, paciencia 4, restaura los mejores pesos) y ReduceLROnPlateau(factor 0,5, paciencia 2).
 * El umbral clínico se calcula igual que src/metrics.py::find_threshold_for_recall: el más alto que logra
 * sensibilidad >= 95% en validación.
 */
(() => {
  const $ = (id) => document.getElementById(id);
  const BATCH = 32;
  const TARGET_RECALL = 0.95;
  const BASE_FILE = 'model/base.onnx';
  const BASE_SIZE = 47000157;   // bytes de model/base.onnx

  let data = null;      // { D, train: {x, y}, val: {x, y}, test: {x, y}, meta }
  let stopFlag = false;
  let trained = null;   // último modelo entrenado

  const fmt = (v, d = 3) => v.toLocaleString('es-CO', { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (v) => `${fmt(v * 100, 1)}%`;
  const secs = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${fmt(ms / 1000, 1)} s`);
  const nextFrame = () => new Promise((r) => { let done = false; const f = () => { if (!done) { done = true; r(); } }; requestAnimationFrame(f); setTimeout(f, 30); });

  /* ---------- carga de datos ---------- */
  function halfTable() {
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      t[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
    }
    return t;
  }

  async function load() {
    const [meta, buf] = await Promise.all([
      fetch('lab/meta.json').then((r) => r.json()),
      fetch('lab/features.bin').then((r) => r.arrayBuffer()),
    ]);
    const half = new Uint16Array(buf), tbl = halfTable();
    const all = new Float32Array(half.length);
    for (let i = 0; i < half.length; i++) all[i] = tbl[half[i]];
    const D = meta.dims, c = meta.counts;
    let off = 0;
    const take = (name) => {
      const n = c[name], x = all.subarray(off * D, (off + n) * D);
      off += n;
      return { x, y: Uint8Array.from(meta.labels[name]), n };
    };
    data = { D, meta, train: take('train'), val: take('val'), test: take('test') };
    $('labStatus').textContent = `Listo: ${c.train} imágenes de entrenamiento, ${c.val} de validación y ${c.test} de test.`;
    $('trainBtn').disabled = false;
  }

  /* ---------- modelo ---------- */
  function forward(x, D, i, w, b) {
    let z = b;
    const o = i * D;
    for (let k = 0; k < D; k++) z += x[o + k] * w[k];
    return 1 / (1 + Math.exp(-z));
  }

  function predictAll(set, w, b) {
    const p = new Float32Array(set.n);
    for (let i = 0; i < set.n; i++) p[i] = forward(set.x, data.D, i, w, b);
    return p;
  }

  const bce = (p, y) => { const q = Math.min(Math.max(p, 1e-7), 1 - 1e-7); return -(y ? Math.log(q) : Math.log(1 - q)); };

  function evaluate(p, y, thr) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < p.length; i++) {
      const pos = p[i] >= thr;
      if (pos && y[i]) tp++; else if (pos) fp++; else if (y[i]) fn++; else tn++;
    }
    return { tp, fp, tn, fn, f1: 2 * tp / (2 * tp + fp + fn || 1), recall: tp / (tp + fn || 1), specificity: tn / (tn + fp || 1) };
  }

  function auc(p, y) {   // Mann-Whitney con rangos promedio en empates
    const idx = Array.from(p.keys()).sort((a, b) => p[a] - p[b]);
    const ranks = new Float64Array(p.length);
    for (let i = 0; i < idx.length;) {
      let j = i;
      while (j + 1 < idx.length && p[idx[j + 1]] === p[idx[i]]) j++;
      for (let k = i; k <= j; k++) ranks[idx[k]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    let pos = 0, sum = 0;
    for (let i = 0; i < y.length; i++) if (y[i]) { pos++; sum += ranks[i]; }
    const neg = y.length - pos;
    return (sum - pos * (pos + 1) / 2) / (pos * neg);
  }

  function clinicalThreshold(p, y) {
    const scores = [];
    for (let i = 0; i < p.length; i++) if (y[i]) scores.push(p[i]);
    scores.sort((a, b) => b - a);
    return Math.min(scores[Math.ceil(TARGET_RECALL * scores.length) - 1], 1);
  }

  /* ---------- gráficas ---------- */
  function chart(canvas, series, opts = {}) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height, pad = { l: 44, r: 10, t: 10, b: 22 };
    ctx.clearRect(0, 0, W, H);
    const all = series.flatMap((s) => s.data).filter(Number.isFinite);
    if (!all.length) return;
    let lo = opts.min ?? Math.min(...all), hi = opts.max ?? Math.max(...all);
    if (hi - lo < 1e-6) { hi += 0.05; lo -= 0.05; }
    const n = Math.max(opts.epochs || 1, ...series.map((s) => s.data.length));
    const X = (i) => pad.l + (n === 1 ? 0 : (i / (n - 1)) * (W - pad.l - pad.r));
    const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
    ctx.strokeStyle = '#262121'; ctx.fillStyle = '#6b6664'; ctx.font = '11px JetBrains Mono, monospace'; ctx.lineWidth = 1;
    for (let k = 0; k <= 3; k++) {
      const v = lo + (k / 3) * (hi - lo), y = Y(v);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      ctx.fillText(fmt(v, 2), 4, y + 4);
    }
    ctx.fillText('1', pad.l - 3, H - 6);
    ctx.fillText(String(n), W - pad.r - 8 * String(n).length, H - 6);
    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
      s.data.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
      ctx.stroke();
      if (s.mark != null && s.data[s.mark] != null) {
        ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X(s.mark), Y(s.data[s.mark]), 4, 0, 7); ctx.fill();
      }
    }
  }

  /* ---------- entrenamiento ---------- */
  async function train() {
    const epochs = Number($('epochs').value), lr0 = Number($('lr').value), drop = Number($('dropout').value);
    const early = $('early').checked;
    const { D, train: tr, val, meta } = data;
    const cw = [meta.class_weight[0], meta.class_weight[1]];

    // Glorot uniforme (el inicializador por defecto de Dense en Keras), sesgo en 0.
    const limit = Math.sqrt(6 / (D + 1));
    let w = Float32Array.from({ length: D }, () => (Math.random() * 2 - 1) * limit), b = 0;
    const mw = new Float32Array(D), vw = new Float32Array(D);
    let mb = 0, vb = 0, t = 0, lr = lr0;
    const grad = new Float32Array(D), mask = new Float32Array(D);
    const keep = 1 - drop;

    const hist = { loss: [], val: [], f1: [] };
    let best = { loss: Infinity, w: w.slice(), b, epoch: 0 }, wait = 0, lrWait = 0, lrBest = Infinity;
    stopFlag = false;
    $('trainBtn').disabled = true; $('stopBtn').disabled = false;
    $('labResult').hidden = true;
    const t0 = performance.now();
    let ran = 0, reason = '';

    for (let ep = 0; ep < epochs; ep++) {
      const order = Array.from({ length: tr.n }, (_, i) => i);
      for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      let lossSum = 0;
      for (let s = 0; s < tr.n; s += BATCH) {
        const end = Math.min(s + BATCH, tr.n), bs = end - s;
        grad.fill(0);
        let gb = 0;
        for (let q = s; q < end; q++) {
          const i = order[q], o = i * D, y = tr.y[i];
          let z = b;
          for (let k = 0; k < D; k++) {
            mask[k] = drop > 0 ? (Math.random() < keep ? 1 / keep : 0) : 1;
            z += tr.x[o + k] * mask[k] * w[k];
          }
          const p = 1 / (1 + Math.exp(-z));
          lossSum += cw[y] * bce(p, y);
          const dz = cw[y] * (p - y) / bs;     // Keras promedia la pérdida ponderada sobre el lote
          for (let k = 0; k < D; k++) grad[k] += dz * tr.x[o + k] * mask[k];
          gb += dz;
        }
        t++;
        const a = lr * Math.sqrt(1 - 0.999 ** t) / (1 - 0.9 ** t);
        for (let k = 0; k < D; k++) {
          mw[k] = 0.9 * mw[k] + 0.1 * grad[k];
          vw[k] = 0.999 * vw[k] + 0.001 * grad[k] * grad[k];
          w[k] -= a * mw[k] / (Math.sqrt(vw[k]) + 1e-7);
        }
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb * gb;
        b -= a * mb / (Math.sqrt(vb) + 1e-7);
      }

      const pv = predictAll(val, w, b);
      let vl = 0;
      for (let i = 0; i < val.n; i++) vl += bce(pv[i], val.y[i]);
      vl /= val.n;
      hist.loss.push(lossSum / tr.n); hist.val.push(vl); hist.f1.push(evaluate(pv, val.y, 0.5).f1);
      ran = ep + 1;

      if (vl < best.loss) { best = { loss: vl, w: w.slice(), b, epoch: ep }; wait = 0; } else wait++;
      if (early) {
        if (vl < lrBest - 1e-4) { lrBest = vl; lrWait = 0; } else if (++lrWait >= 2) { lr = Math.max(lr * 0.5, 1e-7); lrWait = 0; }
      }

      const el = performance.now() - t0, per = el / ran;
      $('labBar').style.width = `${(100 * ran / epochs).toFixed(1)}%`;
      $('labStatus').textContent = `Época ${ran} de ${epochs} · pérdida de validación ${fmt(vl, 4)} · `
        + `${secs(el)} transcurridos · quedan unos ${secs(per * (epochs - ran))}`;
      chart($('lossChart'), [{ data: hist.loss, color: '#f2efeb' }, { data: hist.val, color: '#e32b3c', mark: early ? best.epoch : null }], { epochs });
      chart($('f1Chart'), [{ data: hist.f1, color: '#e32b3c' }], { epochs, max: 1 });
      await nextFrame();

      if (stopFlag) { reason = 'Detenido por ti'; break; }
      if (early && wait >= 4) { reason = `Parada temprana en la época ${ran}: la validación no mejoró en 4 épocas`; break; }
    }

    if (early) { w = best.w; b = best.b; }
    const total = performance.now() - t0;
    $('labStatus').textContent = `${reason || `Terminó las ${ran} épocas`} · ${secs(total)} en total`
      + `${early ? ` · se usan los pesos de la época ${best.epoch + 1}, la de menor pérdida de validación` : ''}.`;
    $('trainBtn').disabled = false; $('stopBtn').disabled = true;
    finish({ w, b, epochs: ran, lr: lr0, drop, early });
  }

  function finish(model) {
    const { val, test, meta } = data;
    const pv = predictAll(val, model.w, model.b), pt = predictAll(test, model.w, model.b);
    const thr = clinicalThreshold(pv, val.y);
    const v = evaluate(pv, val.y, thr), r = evaluate(pt, test.y, thr), a = auc(pt, test.y);
    trained = { ...model, thr, val: v, test: { ...r, auc: a } };

    $('labStats').innerHTML = [['F1', fmt(r.f1)], ['Sensibilidad', pct(r.recall)], ['Especificidad', pct(r.specificity)], ['AUC', fmt(a)]]
      .map(([k, x]) => `<div><dt>${k}</dt><dd>${x}</dd></div>`).join('');
    const ref = meta.run1.local_test.f1;
    const diff = r.f1 - ref;
    $('labCompare').textContent = `Con su umbral clínico (${fmt(thr)}, fijado en validación) detecta ${r.tp} de ${r.tp + r.fn} neumonías `
      + `y da ${r.fp} falsas alarmas. El Run 1 oficial, sobre estas mismas imágenes, logra un F1 de ${fmt(ref)}: tu modelo queda `
      + `${Math.abs(diff) < 0.01 ? 'prácticamente igual' : diff > 0 ? `${fmt(diff)} por encima` : `${fmt(-diff)} por debajo`}. `
      + 'Una diferencia de un par de centésimas puede ser solo azar: vuelve a entrenar con lo mismo y verás que cambia un poco.';
    $('labResult').hidden = false;
  }

  /* ---------- usar el modelo entrenado ---------- */
  $('useBtn').addEventListener('click', async () => {
    if (!trained || !window.__aixray) return;
    const t = trained;
    $('useBtn').disabled = true;
    try {
      await window.__aixray.registerCustom({
        key: 'custom', label: 'Tu modelo · entrenado aquí', short: 'Tu modelo', chosen: false,
        description: `Entrenado en tu navegador: ${t.epochs} épocas, learning rate ${String(t.lr).replace('.', ',')}, dropout ${String(t.drop).replace('.', ',')}.`,
        file: BASE_FILE, size: BASE_SIZE, threshold: t.thr,
        val: { f1_score: t.val.f1 }, test: { f1_score: t.test.f1, recall: t.test.recall, specificity: t.test.specificity, auc: t.test.auc },
        head: { w: t.w, b: t.b },
      });
    } finally { $('useBtn').disabled = false; }
  });

  $('epochs').addEventListener('input', (e) => { $('epochsVal').textContent = e.target.value; });
  $('trainBtn').addEventListener('click', () => train().catch((e) => { console.error(e); $('labStatus').textContent = 'Algo falló durante el entrenamiento.'; $('trainBtn').disabled = false; $('stopBtn').disabled = true; }));
  $('stopBtn').addEventListener('click', () => { stopFlag = true; });
  load().catch((e) => { console.error(e); $('labStatus').textContent = 'No se pudieron cargar los datos del laboratorio.'; });

  window.__aixrayLab = { evaluateRun1: () => { const r = data.meta.run1; return evaluate(predictAll(data.test, Float32Array.from(r.w), r.b), data.test.y, r.threshold); } };
})();
