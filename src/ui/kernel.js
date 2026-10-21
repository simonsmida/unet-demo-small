/* Optional, full-precision Conv2d inspection. Display scaling never enters the calculation. */
const kernelInspector = (() => {
  const style = document.createElement('style');
  style.textContent = `
    .kernel-open { align-self: center; font-size: 11px; padding: 2px 8px; margin-top: 5px; min-height: 22px; color: #475569; }
    .selected-panel:has(.kernel-open:not([hidden])) .selected-content { padding-top: 8px; }
    #kernel-dialog { color-scheme: light; color: #17233d; background: #fff; border: 1px solid #dbe3ed; border-radius: 16px; padding: 20px; width: min(940px, calc(100vw - 32px)); max-height: calc(100dvh - 32px); overflow: auto; box-sizing: border-box; box-shadow: 0 24px 80px #0f172a33; }
    #kernel-dialog::backdrop { background: #0f172a80; backdrop-filter: blur(3px); }
    #kernel-dialog h2 { font: 600 20px system-ui; margin: 0; }
    #kernel-dialog p { font: 12px/1.4 system-ui; margin: 6px 0; color: #64748b; }
    #kernel-dialog button, #kernel-dialog select { font: 13px system-ui; border: 1px solid #dbe3ed; border-radius: 7px; background: #fff; color: #17233d; padding: 7px 10px; cursor: pointer; }
    #kernel-dialog button:focus-visible, #kernel-dialog select:focus-visible, #kernel-dialog canvas:focus-visible { outline: 2px solid #2563eb; outline-offset: 3px; }
    .kernel-header, .kernel-controls { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
    .kernel-controls { justify-content: flex-start; flex-wrap: wrap; margin-top: 12px; }
    #kernel-dialog label { font: 12px system-ui; display: flex; align-items: center; gap: 8px; }
    .kernel-columns { display: grid; grid-template-columns: 1fr 160px 1fr; gap: 20px; align-items: start; margin-top: 16px; }
    #kernel-dialog figure { margin: 0; text-align: center; min-width: 0; }
    #kernel-dialog figcaption { font: 600 13px system-ui; margin-bottom: 8px; min-height: 30px; }
    #kernel-dialog figcaption small { display: block; font: 11px system-ui; color: #64748b; margin-top: 4px; }
    #kernel-dialog canvas { width: 100%; max-width: 200px; aspect-ratio: 1; image-rendering: pixelated; border-radius: 4px; cursor: crosshair; }
    #kernel-weights { display: grid; gap: 3px; margin: auto; width: 100%; max-width: 156px; }
    #kernel-weights span { aspect-ratio: 1; display: grid; place-items: center; border-radius: 4px; font: 10px ui-monospace, monospace; min-width: 0; }
    .kernel-scale { font: 11px system-ui; color: #64748b; margin-top: 8px; }
    .kernel-calculation { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 14px; margin-top: 12px; }
    #kernel-dialog .kernel-calculation p { margin: 4px 0; color: #334155; }
    #kernel-equation { font: 14px ui-monospace, monospace !important; color: #17233d !important; }
    #kernel-patch { font: 10px/1.45 ui-monospace, monospace; white-space: pre; overflow-x: auto; max-width: 100%; margin-top: 8px; color: #475569; }
    #kernel-position { flex: 1; min-width: 150px; accent-color: #f97316; }
    #kernel-play { background: #ecfdf5 !important; }
    #kernel-dialog [hidden] { display: none !important; }
    @media(max-width: 650px) { #kernel-dialog { padding: 16px; } .kernel-columns { grid-template-columns: 1fr 110px; gap: 18px; } .kernel-columns figure:last-child { grid-column: 1 / -1; } #kernel-patch { font-size: 9px; } }
    @media(prefers-reduced-motion: reduce) { #kernel-dialog::backdrop { backdrop-filter: none; } }
  `;
  document.head.append(style);
  const dialog = document.createElement('dialog');
  dialog.id = 'kernel-dialog';
  dialog.setAttribute('aria-labelledby', 'kernel-title');
  dialog.innerHTML = `
    <div class="kernel-header"><h2 id="kernel-title">How a learned filter works</h2><button id="kernel-close" aria-label="Close kernel inspector">✕</button></div>
    <p id="kernel-subtitle"></p>
    <p id="kernel-loading" role="status">Loading the learned weights and full-precision inputs…</p>
    <div id="kernel-body" hidden>
      <div class="kernel-controls">
        <label>Input channel <select id="kernel-input-channel" aria-label="Input channel"></select></label>
        <label>Output <select id="kernel-activation" aria-label="Output activation"><option value="pre">Before ReLU</option><option value="post">After ReLU</option></select></label>
      </div>
      <div class="kernel-columns">
        <figure><figcaption>Input to this convolution<small id="kernel-input-size"></small></figcaption><canvas id="kernel-input" aria-label="Choose an input location"></canvas><div class="kernel-scale" id="kernel-input-scale"></div></figure>
        <figure><figcaption>Learned weights<small id="kernel-weight-label"></small></figcaption><div id="kernel-weights"></div><div class="kernel-scale" id="kernel-weight-scale"></div><pre id="kernel-patch"></pre></figure>
        <figure><figcaption>Output feature map<small id="kernel-output-label"></small></figcaption><canvas id="kernel-output" aria-label="Choose an output pixel"></canvas><div class="kernel-scale" id="kernel-output-scale"></div></figure>
      </div>
      <div class="kernel-controls"><button id="kernel-play">▶ Build output</button><input type="range" id="kernel-position" min="0" value="0" aria-label="Output pixel"><span id="kernel-coordinate" class="kernel-scale"></span></div>
      <div class="kernel-calculation" aria-live="off"><p id="kernel-slice"></p><p id="kernel-equation"></p><p id="kernel-result"></p></div>
      <p>Output = weighted patches from <strong>all input channels</strong> + bias, then ReLU or sigmoid. The 3 × 3 layers use zero padding.</p>
      <p>Blue = negative. Red = positive. Weight scale is shared across input channels. Image scales are independent. Playback timing is illustrative.</p>
    </div>`;
  document.body.append(dialog);
  const el = id => document.getElementById(`kernel-${id}`);
  let data, feature, requestId, loadingId, channel = 0, pixel = 0, frame = 0, revealing = false;
  const format = value => Math.abs(value) < 1e-9 ? '0' : Number(value.toPrecision(5)).toString();
  const stop = () => { cancelAnimationFrame(frame); frame = 0; revealing = false; el('play').textContent = '▶ Build output'; };
  const close = () => { stop(); clearTimeout(loadingId); requestId = null; dialog.close(); };
  el('close').onclick = close;
  dialog.addEventListener('cancel', close);
  dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) close(); } });
  async function unpack(encoded) {
    const raw = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
    return new Float32Array(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  }
  function outputValues() {
    if (el('activation').value === 'pre') return data.output;
    return data.output.map(value => feature.has_sigmoid ? 1 / (1 + Math.exp(-value)) : Math.max(0, value));
  }
  function draw(canvas, values, signed, limit = values.length) {
    const [, h, w] = data.shape;
    canvas.width = w; canvas.height = h;
    const context = canvas.getContext('2d');
    let low = signed ? -Math.max(...values.map(Math.abs)) : 0;
    let high = signed ? -low : Math.max(...values);
    if (canvas === el('output') && feature.has_sigmoid && el('activation').value === 'post') { low = 0; high = 1; }
    const image = context.createImageData(w, h);
    for (let i = 0; i < values.length; i++) {
      const v = high === low ? 0 : Math.max(0, Math.min(1, (values[i] - low) / (high - low)));
      const offset = i * 4;
      if (i >= limit) image.data.set([238, 242, 247, 255], offset);
      else if (signed) {
        const t = Math.abs(v * 2 - 1), color = v < .5 ? [33,102,172] : [178,24,43];
        image.data.set([...color.map(c => Math.round(247 + t * (c - 247))),255], offset);
      } else image.data.set([v*255,v*255,v*255,255], offset);
    }
    context.putImageData(image, 0, 0);
    return `${format(low)} → ${format(high)}`;
  }
  function renderWeights() {
    const k = data.kernel_size, offset = channel*k*k;
    const weights = data.weights.subarray(offset, offset+k*k);
    const peak = Math.max(...data.weights.map(Math.abs)); // Shared scale across slices of this filter.
    el('weights').style.gridTemplateColumns = `repeat(${k}, 1fr)`;
    el('weights').replaceChildren(...Array.from(weights, value => {
      const cell = document.createElement('span'); cell.textContent = value && Math.abs(value) < .001 ? value.toExponential(1) : Number(value.toPrecision(3)).toString(); cell.title = value.toString();
      const t = peak ? Math.abs(value)/peak : 0;
      const color = value < 0 ? [33,102,172] : [178,24,43];
      cell.style.background = `rgb(${color.map(c => Math.round(247+t*(c-247))).join(',')})`;
      cell.style.color = t > .55 ? '#fff' : '#17233d';
      return cell;
    }));
    el('weight-label').textContent = `${k} × ${k} slice · input ${channel+1}`;
    el('weight-scale').textContent = `${format(-peak)} → ${format(peak)}`;
  }
  function renderPixel() {
    const [channels,h,w] = data.shape, k = data.kernel_size, p = data.padding;
    const x = pixel % w, y = Math.floor(pixel / w);
    const contributions = new Float64Array(channels);
    const patch = [];
    for (let c = 0; c < channels; c++) {
      for (let ky = 0; ky < k; ky++) {
        const row = [];
        for (let kx = 0; kx < k; kx++) {
          const ix = x+kx-p, iy = y+ky-p;
          const input = ix<0 || ix>=w || iy<0 || iy>=h ? 0 : data.inputs[c*h*w+iy*w+ix];
          contributions[c] += input * data.weights[c*k*k+ky*k+kx];
          row.push(Number(input.toPrecision(3)).toString());
        }
        if (c === channel) patch.push(row.join('  '));
      }
    }
    const sum = contributions.reduce((a,b) => a+b, 0), before = data.output[pixel], after = feature.has_sigmoid ? 1/(1+Math.exp(-before)) : Math.max(0,before);
    el('patch').textContent = `Input patch\n${patch.join('\n')}`;
    el('slice').textContent = `Input channel ${channel+1}: sum(patch × weights) = ${format(contributions[channel])}`;
    el('equation').textContent = `${channels === 1 ? 'Weighted patch' : `Sum of ${channels} channel contributions`} ${format(sum)} + bias ${format(data.bias)} = ${format(sum+data.bias)}`;
    el('result').textContent = `Stored pre-activation: ${format(before)} → ${feature.has_sigmoid ? 'sigmoid' : 'ReLU'}: ${format(after)}`;
    el('coordinate').textContent = `x ${x}, y ${y}`;
    el('position').value = pixel;
    const input = data.inputs.subarray(channel*h*w,(channel+1)*h*w);
    el('input-scale').textContent = draw(el('input'),input,input.some(v => v<0));
    const ctx = el('input').getContext('2d');
    ctx.strokeStyle = '#f97316'; ctx.lineWidth = .7;
    ctx.strokeRect(Math.max(0,x-p),Math.max(0,y-p), Math.min(w,x+k-p)-Math.max(0,x-p),Math.min(h,y+k-p)-Math.max(0,y-p));
    const values = outputValues();
    el('output-scale').textContent = draw(el('output'),values,el('activation').value==='pre',revealing ? pixel+1 : values.length);
    const out = el('output').getContext('2d'); out.strokeStyle='#f97316'; out.lineWidth=.7; out.strokeRect(x,y,1,1);
    el('output-label').textContent = `Channel ${data.channel+1} · ${el('activation').selectedOptions[0].textContent}`;
  }
  el('input-channel').onchange = () => { channel = +el('input-channel').value; renderWeights(); renderPixel(); };
  el('activation').onchange = renderPixel;
  el('position').oninput = () => { stop(); pixel = +el('position').value; renderPixel(); };
  for (const name of ['input','output']) el(name).onclick = event => {
    stop(); const r = event.target.getBoundingClientRect(), [,h,w] = data.shape;
    pixel = Math.min(h-1,Math.floor((event.clientY-r.top)/r.height*h))*w + Math.min(w-1,Math.floor((event.clientX-r.left)/r.width*w)); renderPixel();
  };
  el('play').onclick = () => {
    if (frame) { cancelAnimationFrame(frame); frame = 0; el('play').textContent = '▶ Continue'; return; }
    const length = data.output.length, offset = revealing ? pixel : 0;
    revealing = true; el('play').textContent = 'Pause';
    const start = performance.now() - offset / length * 8000;
    function tick(now) {
      pixel = Math.min(length-1,Math.floor((now-start)/8000*length)); renderPixel();
      if (pixel < length-1) frame = requestAnimationFrame(tick);
      else { stop(); renderPixel(); }
    }
    frame = requestAnimationFrame(tick);
  };
  return {
    open(selected, outputChannel, request) {
      feature = selected; data = null; channel = 0; pixel = 0;
      el('subtitle').textContent = `${feature.label} · output channel ${outputChannel+1}. Convolution slides the learned weights over its input without flipping them (PyTorch cross-correlation).`;
      el('body').hidden = true; el('loading').hidden = false;
      el('loading').textContent = 'Loading the learned weights and full-precision inputs…';
      dialog.showModal(); requestId = Date.now(); request(requestId);
      loadingId = setTimeout(() => { if (!data && dialog.open) el('loading').textContent = 'Still loading. Close and reopen to retry if the local server is unavailable.'; },15000);
    },
    async receive(payload) {
      if (!payload || payload.event !== requestId || data) return;
      const id = requestId;
      const [inputs, weights, output] = await Promise.all(['inputs','weights','output'].map(key => unpack(payload[key])));
      if (id !== requestId || !dialog.open) return;
      clearTimeout(loadingId); data = {...payload, inputs, weights, output};
      const [channels,h,w] = data.shape;
      el('input-channel').replaceChildren(...Array.from({length: channels},(_,c) => new Option(`${c+1} of ${channels}`,c)));
      el('input-channel').disabled = channels === 1;
      el('activation').options[0].textContent = feature.has_sigmoid ? 'Before sigmoid' : 'Before ReLU';
      el('activation').options[1].textContent = feature.has_sigmoid ? 'After sigmoid' : 'After ReLU';
      el('activation').value = 'pre';
      el('position').max = h*w-1; pixel = Math.floor(h/2)*w+Math.floor(w/2);
      if (data.kernel_size === 1) el('subtitle').textContent = `${feature.label} · output channel ${data.channel+1}. A 1 × 1 convolution mixes input channels at each pixel without using neighboring pixels.`;
      el('input-size').textContent = `${channels} ${channels===1 ? 'channel' : 'channels'} · ${w} × ${h} px`;
      el('loading').hidden = true; el('body').hidden = false;
      renderWeights(); renderPixel();
    }
  };
})();
