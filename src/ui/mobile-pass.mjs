// The teaching steps follow model.forward(). Saved skip tensors are references,
// not additional computation. Convolutions include their real ReLU output.
export function mobileSteps(features) {
  const stages = features.filter(feature => !feature.id.endsWith('_skip'));
  const steps = stages.map((feature, index) => {
    const previous = stages[index - 1];
    const input = previous ? {id: previous.id, view: previous.has_relu ? 'post' : 'output', label: 'Input features'}
      : {id: 'input', label: 'Input image'};
    const step = {id: feature.id, feature, inputs: [input], view: 'output',
      label: feature.label.replace('Convolution', 'Conv'), operation: 'Conv + ReLU', kind: 'conv',
      note: 'Each output channel combines the input channels. Shown after ReLU.',
      saved: /^enc_l[12]_conv2$/.test(feature.id)};
    if (feature.has_relu) step.view = 'post';
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
    return step;
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

function tensor(label, feature, channels, image, {result = false, select = null, role = '', view = 'output'} = {}) {
  const card = node('div', `pass-tensor ${result ? 'pass-result-tensor' : ''} ${role}`);
  card.append(node('div', 'pass-tensor-label', label));
  const count = feature?.channels || 1;
  const height = feature?.height || 128, width = feature?.width || 128;
  card.append(node('div', 'pass-shape', `${count} ${count === 1 ? 'channel' : 'channels'} · ${width} × ${height}`));
  const maps = node('div', `pass-maps ${channels.length === 1 ? 'single' : ''}`);
  for (const channel of channels) {
    const figure = node(select ? 'button' : 'figure', 'pass-map');
    const img = node('img');
    img.src = image(channel);
    img.alt = feature ? `${label}, channel ${channel + 1}` : label;
    img.dataset.tensor = feature?.id || 'input';
    img.dataset.view = view;
    img.dataset.mapChannel = String(channel);
    img.width = width; img.height = height;
    figure.append(img);
    if (count > 1) figure.append(node('span', 'pass-channel-number', String(channel + 1)));
    if (select) {
      figure.type = 'button';
      figure.dataset.channel = String(channel);
      figure.setAttribute('aria-label', `Inspect channel ${channel + 1}`);
      figure.setAttribute('aria-pressed', 'false');
      figure.onclick = () => select(channel);
    }
    maps.append(figure);
  }
  card.append(maps);
  return card;
}

export function renderMobileStep(container, step, features, image, inputUrl, predictionUrl, select) {
  const inputs = node('div', `pass-inputs ${step.inputs.length === 2 ? 'pair' : ''}`);
  for (const source of step.inputs) {
    const feature = features.find(item => item.id === source.id);
    const channels = Array.from({length: Math.min(2, feature?.channels || 1)}, (_, index) => index);
    inputs.append(tensor(source.label, feature, channels,
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
  let channels = Array.from({length: Math.min(4, step.feature.channels)}, (_, index) => index);
  if (step.kind === 'concat') {
    const half = step.feature.channels / 2;
    channels = [0, 1, half, half + 1];
  }
  const label = step.binary ? 'Prediction' : step.id === 'sigmoid' ? 'Probability'
    : step.feature.id === 'final_conv' ? 'Logits' : 'Output features';
  result.append(tensor(label, step.feature, channels,
    channel => step.binary ? predictionUrl : image(step.feature, channel, step.view),
    {result: true, select: step.binary ? null : select, view: step.binary ? 'binary' : step.view}));
  const caption = node('div', 'pass-result-caption');
  if (step.saved) caption.append(node('span', 'pass-saved', 'Saved for decoder'));
  caption.append(node('span', '', step.binary ? 'Binary mask' : step.id === 'sigmoid' ? 'Fixed 0–1 scale'
    : `${channels.length} of ${step.feature.channels} ${step.feature.channels === 1 ? 'channel' : 'channels'} · per-channel scale`));
  result.append(caption);
  container.replaceChildren(inputs, operation, result);
}
