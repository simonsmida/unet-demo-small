// The teaching steps follow model.forward(). Saved skip tensors are references,
// not additional computation. Convolution and ReLU are separate tensor views.
export function mobileSteps(features) {
  const stages = features.filter(feature => !feature.id.endsWith('_skip'));
  const steps = stages.flatMap((feature, index) => {
    const previous = stages[index - 1];
    const input = previous ? {id: previous.id, view: previous.has_relu ? 'post' : 'output', label: previous.label.replace('Convolution', 'Conv')}
      : {id: 'input', label: 'Input image'};
    const step = {id: feature.id, feature, inputs: [input], view: 'output',
      label: feature.label.replace('Convolution', 'Conv'), operation: '3 × 3 Conv', kind: 'conv',
      note: 'Each learned filter combines all input channels. These are the values before ReLU.',
      saved: /^enc_l[12]_conv2$/.test(feature.id)};
    if (feature.has_relu) return [
      {...step, view: 'pre', saved: false},
      {...step, id: `${feature.id}_relu`, view: 'post', kind: 'relu', operation: 'ReLU',
        label: `${step.label} · ReLU`,
        inputs: [{id: feature.id, view: 'pre', label: 'Convolution output'}],
        note: 'ReLU replaces negative values with zero. Positive values and tensor dimensions stay unchanged.'}
    ];
    else if (feature.id.startsWith('pool')) {
      step.operation = 'Max pooling'; step.kind = 'pool';
      step.note = 'Each 2 × 2 window keeps its largest value. Channel count stays the same.';
    } else if (feature.id.endsWith('_deconv')) {
      step.operation = 'Upsample'; step.kind = 'up';
      step.note = 'A learned transposed convolution doubles the spatial resolution.';
    } else if (feature.id.endsWith('_concat')) {
      const encoder = feature.id.replace('dec_', 'enc_').replace('_concat', '_conv2');
      step.inputs = [{...input, label: 'Upsampled'}, {id: encoder, view: 'post', label: 'Encoder skip'}];
      step.operation = 'Concatenate'; step.kind = 'concat';
      step.note = 'Saved encoder channels are copied alongside upsampled channels. Values are not added.';
    } else if (feature.id === 'final_conv') {
      step.operation = '1 × 1 Conv'; step.kind = 'output'; step.view = 'pre';
      step.label = 'Output · Logits';
      step.note = 'One foreground logit per pixel. These values are not probabilities.';
    }
    return [step];
  });
  const output = stages[stages.length - 1];
  steps.push({id: 'sigmoid', feature: output, inputs: [{id: output.id, view: 'pre', label: 'Logits'}],
    label: 'Output · Probability', operation: 'Sigmoid', kind: 'output', view: 'post',
    note: 'Sigmoid maps logits to foreground probabilities between 0 and 1.'});
  steps.push({id: 'threshold', feature: output, inputs: [{id: output.id, view: 'post', label: 'Probability'}],
    label: 'Output · Prediction', operation: 'Threshold 0.5', kind: 'output', view: 'post', binary: true,
    note: 'Probability ≥ 0.5 becomes foreground. All other pixels become background.'});
  return steps;
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

// Same orthographic plane projection as the desktop SVG. A single scale is
// shared by every tensor so pooling, upsampling and channel changes stay visible.
export function tensorGeometry(feature) {
  const channels = feature?.channels || 1;
  const width = (feature?.width || 128) * 112 / 128;
  const height = (feature?.height || 128) * 112 / 128;
  const depth = (channels - 1) * 2.5;
  return {width, height, depth, footprintWidth: .42 * width + depth,
    footprintHeight: .7 * height + .12 * width};
}

// Every channel has its own real image. Overlapping sheets keep the full tensor
// compact, with the same fixed projection and channel spacing throughout.
function tensor(label, feature, image, {result = false, select = null, role = '', view = 'output', split = null} = {}) {
  const card = node('div', `pass-tensor ${result ? 'pass-result-tensor' : ''} ${role}`);
  card.dataset.tensor = feature?.id || 'input';
  card.dataset.view = view;
  const geometry = tensorGeometry(feature);
  for (const [name, value] of Object.entries(geometry)) card.style.setProperty(`--${name}`, `${value}px`);
  card.append(node('div', 'pass-tensor-label', label));
  const count = feature?.channels || 1;
  const height = feature?.height || 128, width = feature?.width || 128;
  card.append(node('div', 'pass-shape', `${count} ${count === 1 ? 'channel' : 'channels'} · ${width} × ${height}`));
  const maps = node('div', `pass-maps ${count === 1 ? 'single' : 'stacked'}`);
  maps.setAttribute('aria-label', `${count} channels in original order`);
  const section = node('div', 'pass-stack-section');
  const stack = node('div', 'pass-stack');
  stack.style.setProperty('--sheets', count);
  for (let channel = 0; channel < count; channel++) {
    const front = channel === count - 1;
    const figure = node(select ? 'button' : 'figure', 'pass-map');
    figure.style.setProperty('--sheet', channel);
    const img = node('img');
    img.src = image(channel);
    img.alt = feature ? `${label}, channel ${channel + 1}` : label;
    img.dataset.tensor = feature?.id || 'input';
    img.dataset.view = view;
    img.dataset.mapChannel = String(channel);
    img.width = width; img.height = height;
    figure.append(img);
    if (select) {
      figure.type = 'button';
      figure.dataset.channel = String(channel);
      figure.setAttribute('aria-label', `Inspect channel ${channel + 1}`);
      figure.setAttribute('aria-pressed', 'false');
      figure.onclick = () => select(channel);
      // The full-size grid below provides touch and keyboard access to every
      // channel. Only unobscured front sheets are tab stops in the stack.
      figure.tabIndex = front ? 0 : -1;
    }
    stack.append(figure);
  }
  section.append(stack);
  if (split) section.append(node('div', 'pass-channel-range',
    `${split[0]} · ${count / 2}  +  ${split[1]} · ${count / 2}`));
  maps.append(section);
  card.append(maps);
  return card;
}

export function renderMobileStep(container, step, features, image, inputUrl, predictionUrl, select) {
  const inputs = node('div', `pass-inputs ${step.inputs.length === 2 ? 'pair' : ''}`);
  for (const source of step.inputs) {
    const feature = features.find(item => item.id === source.id);
    inputs.append(tensor(source.label, feature,
      channel => feature ? image(feature, channel, source.view) : inputUrl,
      {role: source.label === 'Encoder skip' ? 'skip-source' : '', view: source.view || 'output'}));
  }
  const operation = node('div', `pass-operation ${step.kind}`);
  const pill = node('span', 'pass-operation-pill', step.operation);
  operation.append(pill);
  if (step.kind === 'concat') {
    const [first, second] = step.inputs.map(source => features.find(item => item.id === source.id).channels);
    operation.append(node('span', 'pass-join-count', `${first} + ${second} → ${step.feature.channels} channels`));
  }
  const result = node('div', 'pass-result');
  const label = step.binary ? 'Prediction' : step.id === 'sigmoid' ? 'Probability'
    : step.feature.id === 'final_conv' ? 'Logits' : 'Output features';
  result.append(tensor(label, step.feature,
    channel => step.binary ? predictionUrl : image(step.feature, channel, step.view),
    {result: true, select: step.binary ? null : select, view: step.binary ? 'binary' : step.view,
      split: step.kind === 'concat' ? ['Upsampled', 'Skip'] : null}));
  const caption = node('div', 'pass-result-caption');
  if (step.saved) caption.append(node('span', 'pass-saved', 'Saved for decoder'));
  caption.append(node('span', '', step.binary ? 'Binary mask' : step.id === 'sigmoid' ? 'Fixed 0–1 scale'
    : step.feature.channels === 1 ? 'Per-channel scale' : 'All channels · per-channel scale'));
  result.append(caption);
  container.classList.toggle('pass-upward', step.kind === 'up');
  container.replaceChildren(...(step.kind === 'up' ? [result, operation, inputs] : [inputs, operation, result]));
}
