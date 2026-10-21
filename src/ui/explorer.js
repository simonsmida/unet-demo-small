const $ = id => document.getElementById(id);
const blocks = ['Encoder', 'Bottleneck', 'Decoder', 'Output'];
const mobileMode = matchMedia('(max-width: 650px)');
let args, features = [], activeLayer = 'enc_l1_conv1', activeChannel = 0;
let activationView = 'post';
let sigmoidView = 'pre';
let displaySize = 'uniform', sampleId, referenceSource, currentDiagram, lastHeight = 0;
let inputPixels, prediction, reference, inputUrl, predictionUrl, referenceUrl;
let imageCache = new Map(), lastLayers = new Map();
let diagramFacesKey;
let selectionVisible = true, selectedDiagramFace = null;
let collapseDiagram = () => {};
let selectionVersion = 0, sampleVersion = 0, updatingSample = false;
let diagramPromise, inputTimeout;
let forwardRun = 0, forwardPlaying = false;
const forwardAnimations = new Set();
function loadDiagram() {
  if (!diagramPromise) {
    diagramPromise = fetch('./architecture.svg').then(response => {
      if (!response.ok) throw new Error('Could not load the architecture diagram.');
      return response.text();
    }).catch(error => { diagramPromise = null; throw error; });
  }
  return diagramPromise;
}
function prepareDiagramViews(items, encoding) {
  // Every displayed stack needs its real output; unused activation alternatives stay compressed.
  return Promise.all(items.flatMap(feature => {
    if (feature.id.endsWith('_concat') || feature.id.endsWith('_skip')) return [];
    const names = new Set([viewName(feature)]);
    // Encoder copies always carry post-ReLU values, even in the before-ReLU view.
    if (/^enc_l[12]_conv2$/.test(feature.id)) names.add('post');
    return [...names].map(name => decodeView(feature, name, encoding));
  }));
}
async function decodeView(feature, name, encoding) {
  const view = feature.views[name];
  if (!view.ready) {
    const stream = new Blob([bytes(view.pixels)]).stream()
      .pipeThrough(new DecompressionStream(encoding));
    view.ready = new Response(stream).arrayBuffer().then(buffer => {
      view.decodedPixels = new Uint8Array(buffer);
    }).catch(error => { delete view.ready; throw error; });
  }
  await view.ready;
}
const namespace = 'http://www.w3.org/2000/svg';

function send(type, data) {
  window.parent.postMessage({isStreamlitMessage: true, type, ...data}, '*');
}
function updateHeight() {
  let height = Math.ceil($('explorer').getBoundingClientRect().height) + 4;
  if (mobileMode.matches) {
    height = window.innerHeight;
    document.documentElement.style.setProperty('--mobile-height', `${height}px`);
  }
  if (height !== lastHeight) {
    lastHeight = height;
    send('streamlit:setFrameHeight', {height});
  }
}
function bytes(encoded) {
  return Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
}
function pixelImage(pixels, width, height, mask = null, opacity = 0) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  const image = context.createImageData(width, height);
  const color = [249, 115, 22];
  for (let index = 0; index < pixels.length; index++) {
    const alpha = mask?.[index] ? opacity : 0;
    for (let channel = 0; channel < 3; channel++) {
      image.data[index * 4 + channel] = Math.round(pixels[index] * (1 - alpha) + color[channel] * alpha);
    }
    image.data[index * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  return canvas.toDataURL('image/png');
}
function currentFeature() {
  return features.find(feature => feature.id === activeLayer);
}
function viewName(feature) {
  return feature.has_relu ? activationView : feature.has_sigmoid ? sigmoidView : 'output';
}
function channelSource(feature, channel) {
  if (feature.id.endsWith('_concat')) {
    const half = feature.channels / 2;
    const id = feature.id.replace('_concat', channel < half ? '_deconv' : '_skip');
    return channelSource(features.find(item => item.id === id), channel % half);
  }
  if (feature.id.endsWith('_skip')) {
    const id = feature.id.replace('dec_', 'enc_').replace('_skip', '_conv2');
    return {feature: features.find(item => item.id === id), channel, name: 'post'};
  }
  return {feature, channel, name: viewName(feature)};
}
function channelImage(feature, channel) {
  // Concatenation copies channels. Keep their source colors and per-channel scale.
  const source = channelSource(feature, channel);
  ({feature, channel} = source);
  const name = source.name;
  const view = feature.views[name];
  const key = `${feature.id}:${name}`;
  if (!imageCache.has(key)) imageCache.set(key, []);
  const images = imageCache.get(key);
  if (!images[channel]) {
    const size = feature.width * feature.height;
    const data = view.decodedPixels.subarray(channel * size, (channel + 1) * size);
    const canvas = document.createElement('canvas');
    canvas.width = feature.width;
    canvas.height = feature.height;
    const context = canvas.getContext('2d');
    const image = context.createImageData(feature.width, feature.height);
    for (let index = 0; index < size; index++) {
      const value = view.signed ? (data[index] - 128) / 127 : data[index] / 255;
      const edge = value < 0 ? [33, 102, 172] : [178, 24, 43];
      for (let rgb = 0; rgb < 3; rgb++) {
        image.data[4 * index + rgb] = view.signed
          ? Math.round(247 + Math.abs(value) * (edge[rgb] - 247))
          : Math.round(255 * value);
      }
      image.data[4 * index + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    images[channel] = canvas.toDataURL('image/png');
  }
  return images[channel];
}
function renderActivationControl(feature) {
  const available = feature.has_relu || feature.has_sigmoid;
  const operation = feature.has_sigmoid ? 'sigmoid' : 'ReLU';
  const selected = feature.has_sigmoid ? sigmoidView : activationView;
  $('activation').setAttribute('aria-label', available ? `${operation} activation` : 'Operation output');
  for (const mode of ['pre', 'post']) {
    const button = $(`activation-${mode}`);
    button.textContent = `${mode === 'pre' ? 'Before' : 'After'} ${operation}`;
    button.hidden = !available;
    button.disabled = !available;
    button.setAttribute('aria-pressed', String(available && selected === mode));
  }
  $('activation-output').hidden = available;
}
function shortLabel(feature) {
  return feature.label.replace(/Encoder (\d+)/, 'Level $1').replace(/Decoder (\d+)/, 'Level $1')
    .replace(/convolution/i, 'Conv');
}
function renderSelectors(feature) {
  $('block').value = feature.block;
  $('mobile-blocks').querySelectorAll('button').forEach(button =>
    button.setAttribute('aria-pressed', String(button.dataset.block === feature.block)));
  const select = $('layer');
  select.replaceChildren();
  let lastGroup;
  for (const item of features.filter(item => item.block === feature.block)) {
    const groupName = item.label.split(' · ')[0];
    if (!lastGroup || lastGroup.label !== groupName) {
      lastGroup = document.createElement('optgroup');
      lastGroup.label = groupName;
      select.appendChild(lastGroup);
    }
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = shortLabel(item);
    lastGroup.appendChild(option);
  }
  select.value = feature.id;
  const index = features.findIndex(item => item.id === feature.id);
  $('previous-layer').disabled = index === 0;
  $('next-layer').disabled = index === features.length - 1;
}
function makeSelectable(element, choose, label) {
  element.setAttribute('role', 'button');
  element.setAttribute('tabindex', '0');
  element.setAttribute('aria-label', `Select ${label}`);
  element.addEventListener('click', event => { event.stopPropagation(); choose(); });
  element.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(); }
  });
  const title = document.createElementNS(namespace, 'title');
  title.textContent = `Select ${label}`;
  element.appendChild(title);
}
function fillDiagramFace(feature, face) {
  const channel = Number(face.dataset.channel);
  face.querySelector('image').setAttribute('href', channelImage(feature, channel));
  face.setAttribute('aria-label', `Select ${feature.label}, channel ${channel + 1}`);
  const view = feature.has_relu ? `${activationView === 'post' ? 'after' : 'before'} ReLU`
    : feature.has_sigmoid ? `${sigmoidView === 'post' ? 'after' : 'before'} sigmoid` : 'operation output';
  face.querySelector('title').textContent = `${feature.label} · Channel ${channel + 1} of ${feature.channels} · ${view}`;
}
function updateDiagramSelection() {
  $('explorer').classList.toggle('selection-cleared', !selectionVisible);
  if (selectedDiagramFace) {
    selectedDiagramFace.setAttribute('aria-pressed', 'false');
    selectedDiagramFace.setAttribute('tabindex', '-1');
  }
  selectedDiagramFace = null;
  $('diagram').querySelectorAll('.stack').forEach(stack => {
    const selected = selectionVisible && stack.id === activeLayer;
    if (stack.dataset.selected !== String(selected)) stack.dataset.selected = String(selected);
    if (selected) selectedDiagramFace = stack.querySelector(`.feature-face[data-channel="${activeChannel}"]`);
  });
  if (selectedDiagramFace) {
    selectedDiagramFace.setAttribute('aria-pressed', 'true');
    selectedDiagramFace.setAttribute('tabindex', $('simplified').checked ? '-1' : '0');
  }
}

function updateDiagramMode() {
  const simplified = $('simplified').checked;
  $('explorer').classList.toggle('simplified', simplified);
  $('diagram').querySelectorAll('.stack').forEach(stack => {
    if (simplified) {
      stack.setAttribute('role', 'button');
      stack.setAttribute('tabindex', '0');
      stack.setAttribute('aria-label', `Select ${features.find(item => item.svg_id === stack.id).label}`);
    } else {
      for (const name of ['role', 'tabindex', 'aria-label']) stack.removeAttribute(name);
    }
    // One focus stop instead of hundreds; all channels remain available in the grid.
    stack.querySelectorAll('.feature-face').forEach(face => {
      const selected = selectionVisible && stack.id === activeLayer && Number(face.dataset.channel) === activeChannel;
      face.setAttribute('tabindex', !simplified && selected ? '0' : '-1');
    });
  });
}
function renderDiagram(feature, svgText) {
  if (mobileMode.matches) return;
  if (currentDiagram !== svgText) {
    currentDiagram = svgText;
    $('diagram').innerHTML = svgText;
    const svg = $('diagram').querySelector('svg');
    const stackHomes = new Map();
    selectedDiagramFace = null;
    let expandedStack = null;
    const restoreTimers = new Map();
    const closingTimers = new Map();
    let concatReady = null, concatOpenTimer = null, pendingConcatFace = null;
    const diagramNodes = [...svg.querySelectorAll('.stack, .operation')];
    let overlapTimers = [];
    const intersects = (a, b) => a.left <= b.right + 1 && a.right >= b.left - 1
      && a.top <= b.bottom + 1 && a.bottom >= b.top - 1;
    function updateOcclusion() {
      const faces = expandedStack ? [...expandedStack.querySelectorAll('.feature-face, .hover-face')]
        .map(node => node.getBoundingClientRect()) : [];
      // Read geometry in one batch, then update opacity. Only touching content is dimmed.
      const states = diagramNodes.map(node => {
        if (node === expandedStack || (node.classList.contains('operation') && node.dataset.operation === 'skip')) return [node, false];
        const parts = node.classList.contains('operation')
          ? [...node.querySelectorAll('.edge, .edge-caption')] : [node];
        const bounds = parts.map(part => part.getBoundingClientRect());
        return [node, faces.some(face => bounds.some(bound => intersects(face, bound)))];
      });
      states.forEach(([node, dim]) => {
        if (node.dataset.occluded !== String(dim)) node.dataset.occluded = String(dim);
      });
    }
    function scheduleOcclusion() {
      overlapTimers.forEach(clearTimeout);
      updateOcclusion();
      // A few bounded checks follow the CSS spread, without a continuous render loop.
      overlapTimers = expandedStack ? [80, 180, 360].map(delay => setTimeout(updateOcclusion, delay)) : [];
    }
    const expandStack = next => {
      if (expandedStack === next) return;
      clearTimeout(concatOpenTimer);
      concatReady = null;
      pendingConcatFace = null;
      if (expandedStack) {
        const previous = expandedStack;
        const fold = () => {
          previous.dataset.expanded = 'false';
          previous.querySelectorAll('.concat-component').forEach(group => group.dataset.expanded = 'false');
          previous.style.setProperty('--front-fan-extra', '0px');
          closingTimers.delete(previous);
          restoreTimers.set(previous, setTimeout(() => {
            if (previous !== expandedStack) stackHomes.get(previous).after(previous);
            restoreTimers.delete(previous);
          }, 280));
        };
        // Concat groups fold only after the face-on map has turned back.
        if (previous.dataset.operation === 'concat') closingTimers.set(previous, setTimeout(fold, 240));
        else fold();
      }
      expandedStack = next;
      scheduleOcclusion();
      if (next) {
        clearTimeout(closingTimers.get(next));
        closingTimers.delete(next);
        clearTimeout(restoreTimers.get(next));
        restoreTimers.delete(next);
        const moving = [...next.querySelectorAll(next.dataset.operation === 'concat' ? '.concat-component' : '.feature-face')];
        const starts = moving.map(node => getComputedStyle(node).transform);
        svg.appendChild(next);
        // Preserve motion across SVG reparenting with explicit starting transforms.
        requestAnimationFrame(() => {
          if (expandedStack !== next) return;
          next.dataset.expanded = 'true';
          if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
            moving.forEach((node, index) => {
              let offset = parseFloat(node.style.getPropertyValue('--spread'));
              if (node.classList.contains('feature-face')) offset += parseFloat(node.style.getPropertyValue('--hover-gap')) || 0;
              else if (node.dataset.component === 'skip') offset -= parseFloat(next.style.getPropertyValue('--front-fan-extra')) || 0;
              node.getAnimations().forEach(animation => animation.cancel());
              node.animate([{transform: starts[index]}, {transform: `translateX(${offset}px)`}],
                {duration: 220, easing: 'cubic-bezier(.4,0,.2,1)'});
            });
          }
          scheduleOcclusion();
          const readyDelay = next.dataset.operation === 'concat' ? 820 : 420;
          // Finish spreading before a face-on preview can change the stack's spacing.
          concatOpenTimer = setTimeout(() => {
            if (expandedStack !== next) return;
            concatReady = next;
            const face = pendingConcatFace;
            if (face) {
              expandComponent(face.closest('.concat-component'));
              facePreview(face);
            }
          }, readyDelay);
        });
      }
    };
    let expandedComponent = null;
    const expandComponent = next => {
      if (expandedComponent === next) return;
      if (expandedComponent) {
        const previousStack = expandedComponent.closest('.stack');
        if (previousStack === expandedStack || previousStack.dataset.operation !== 'concat') {
          expandedComponent.dataset.expanded = 'false';
          previousStack.style.setProperty('--front-fan-extra', '0px');
        }
      }
      expandedComponent = next;
      if (next) {
        next.dataset.expanded = 'true';
        // Keep the original gap between groups while the front group's maps fan out.
        const extra = next.dataset.component === 'up' ? next.dataset.fanExtra : 0;
        next.closest('.stack').style.setProperty('--front-fan-extra', `${extra}px`);
      }
      scheduleOcclusion();
    };
    let hoveredFace = null, hoverPreview = null;
    let shiftedFaces = [];
    let hoverTimer = null;
    let pointerScrub = null;
    const facePreview = face => {
      if (face === hoveredFace) return;
      shiftedFaces.forEach(item => item.style.removeProperty('--hover-gap'));
      shiftedFaces = [];
      if (hoverPreview) {
        const previous = hoverPreview;
        const plane = previous.querySelector('.feature-plane');
        const currentTransform = getComputedStyle(plane).transform;
        const currentOpacity = getComputedStyle(plane).opacity;
        const currentPosition = getComputedStyle(previous).transform;
        plane.getAnimations().forEach(animation => animation.cancel());
        previous.getAnimations().forEach(animation => animation.cancel());
        previous.style.transform = currentPosition;
        previous.dataset.facing = 'false';
        previous.style.pointerEvents = 'none';
        plane.style.transform = previous.dataset.originalTransform;
        plane.style.opacity = '0';
        if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
          plane.animate([
            {transform: currentTransform, opacity: currentOpacity},
            {transform: previous.dataset.originalTransform, opacity: 0}
          ], {duration: 220, easing: 'cubic-bezier(.4,0,.2,1)'});
        }
        setTimeout(() => previous.remove(), 240);
      }
      hoveredFace = face;
      hoverPreview = null;
      if (!face) { pointerScrub = null; scheduleOcclusion(); return; }
      const stack = face.closest('.stack');
      hoverPreview = document.createElementNS(namespace, 'g');
      hoverPreview.setAttribute('class', 'hover-face');
      hoverPreview.setAttribute('aria-hidden', 'true');
      hoverPreview.sourceFace = face;
      const size = Number(stack.dataset.height);
      const displaySize = Number(stack.dataset.displaySize);
      // Hovered maps share one display size, centered on the stack height.
      const previewSize = 160;
      hoverPreview.style.setProperty('--face-scale', String(previewSize / size));
      hoverPreview.style.setProperty('--face-top', String(.29 * displaySize - previewSize / 2));
      const expandedOffset = item => {
        const group = item.closest('.concat-component');
        let offset = group ? parseFloat(group.style.getPropertyValue('--spread')) : 0;
        if (group?.dataset.component === 'skip') {
          offset -= parseFloat(stack.style.getPropertyValue('--front-fan-extra')) || 0;
        }
        const property = !group || group.dataset.expanded === 'true' ? '--spread' : '--rest';
        return offset + parseFloat(item.style.getPropertyValue(property));
      };
      const faceX = expandedOffset(face);
      stack.querySelectorAll('.feature-face').forEach(other => {
        // Use stable channel positions, not positions midway through an earlier animation.
        if (other !== face && expandedOffset(other) > faceX + .1) {
          other.style.setProperty('--hover-gap', `${previewSize / 2 - .21 * displaySize + 12}px`);
          shiftedFaces.push(other);
        }
      });
      hoverPreview.addEventListener('click', event => {
        event.stopPropagation();
        face.dispatchEvent(new MouseEvent('click', {bubbles: true}));
      });
      hoverPreview.appendChild(face.querySelector('.feature-plane').cloneNode(true));
      stack.appendChild(hoverPreview);
      // Read placement once; the browser animates rotation and movement without a render loop.
      const matrix = stack.getCTM().inverse().multiply(face.getCTM());
      hoverPreview.style.transform = `translateX(${matrix.e}px)`;
      const targetX = faceX;
      const plane = hoverPreview.querySelector('.feature-plane');
      const scale = previewSize / size;
      const projection = displaySize / size;
      const half = size / 2;
      const centerX = .21 * displaySize - half;
      const centerY = .29 * displaySize - half;
      // With a central origin and identical translations, rotation and enlargement
      // preserve the projected image center throughout the animation.
      plane.style.transformBox = 'fill-box';
      plane.style.transformOrigin = `${half}px ${half}px`;
      const originalTransform = `matrix(${.42 * projection}, ${-.12 * projection}, 0, ${.7 * projection}, ${centerX}, ${centerY})`;
      const finalTransform = `matrix(${scale}, 0, 0, ${scale}, ${centerX}, ${centerY})`;
      hoverPreview.dataset.originalTransform = originalTransform;
      hoverPreview.dataset.facing = 'true';
      hoverPreview.style.transition = 'none';
      plane.style.transition = 'none';
      hoverPreview.style.transform = `translateX(${targetX}px)`;
      plane.style.transform = finalTransform;
      plane.style.opacity = '1';
      // Explicit keyframes also animate newly inserted SVG elements reliably.
      if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
        const timing = {duration: 320, easing: 'cubic-bezier(.4,0,.2,1)'};
        hoverPreview.animate([
          {transform: `translateX(${matrix.e}px)`},
          {transform: `translateX(${targetX}px)`}
        ], timing);
        plane.animate([
          {transform: originalTransform, opacity: 0},
          {transform: finalTransform, opacity: 1}
        ], timing);
      }
      scheduleOcclusion();
    };
    const exploreFace = (stack, face) => {
      expandStack(stack);
      if (concatReady !== stack) {
        pendingConcatFace = face;
        return;
      }
      expandComponent(face?.closest('.concat-component') || null);
      if (face) facePreview(face);
    };
    collapseDiagram = () => {
      clearTimeout(hoverTimer);
      facePreview(null);
      expandStack(null);
      expandComponent(null);
    };
    svg.addEventListener('pointermove', event => {
      if (forwardPlaying || $('simplified').checked) return;
      const preview = event.target.closest('.hover-face');
      let target = preview?.sourceFace || event.target;
      const directFace = target.closest('.feature-face');
      if (directFace) {
        const parent = directFace.parentElement;
        if (!preview || !pointerScrub || pointerScrub.parent !== parent) {
          pointerScrub = {
            parent, startX: event.clientX,
            startOffset: parseFloat(directFace.style.getPropertyValue('--spread')),
            scale: directFace.closest('.stack').getScreenCTM().a,
            faces: [...parent.querySelectorAll('.feature-face')].map(face => ({
              face, offset: parseFloat(face.style.getPropertyValue('--spread'))
            }))
          };
        } else {
          // Browse the original channel positions even where the enlarged preview covers them.
          const offset = pointerScrub.startOffset + (event.clientX - pointerScrub.startX) / pointerScrub.scale;
          target = pointerScrub.faces.reduce((closest, item) =>
            Math.abs(item.offset - offset) < Math.abs(closest.offset - offset) ? item : closest
          ).face;
        }
      }
      const stack = target.closest('.stack');
      clearTimeout(hoverTimer);
      if (!stack) {
        // Allow crossing a small gap without repeatedly closing and reopening the stack.
        hoverTimer = setTimeout(collapseDiagram, 140);
        return;
      }
      if (hoveredFace && stack !== hoveredFace.closest('.stack')) facePreview(null);
      exploreFace(stack, target.closest('.feature-face'));
    });
    svg.addEventListener('pointerleave', () => {
      clearTimeout(hoverTimer);
      hoverTimer = setTimeout(collapseDiagram, 140);
    });
    for (const item of features) {
      const stack = svg.querySelector(`[id="${item.svg_id}"]`);
      stack.addEventListener('click', () => selectLayer(item.id));
      stack.addEventListener('keydown', event => {
        if ($('simplified').checked && event.target === stack && (event.key === 'Enter' || event.key === ' ')) {
          event.preventDefault();
          selectLayer(item.id);
        }
      });
      // Raise the hovered tensor as a whole above neighboring stacks.
      const home = document.createComment('stack position');
      stack.before(home);
      stackHomes.set(stack, home);
      stack.querySelectorAll('.feature-face').forEach(face => {
        const channel = Number(face.dataset.channel);
        makeSelectable(face, () => {
          clearTimeout(hoverTimer);
          exploreFace(stack, face);
          selectLayer(item.id, true, Number(face.dataset.channel));
        }, `${item.label}, channel ${channel + 1}`);
      });
    }
    updateDiagramMode();
  }
  const facesKey = `${sampleId}:${activationView}:${sigmoidView}`;
  if (diagramFacesKey !== facesKey) {
    collapseDiagram();
    $('diagram').querySelectorAll('.hover-face').forEach(face => face.remove());
    for (const item of features) {
      const stack = $(item.svg_id);
      stack.querySelectorAll('.feature-face').forEach(face => {
        // Reuse exactly the same channel renderer as the grid and enlarged preview.
        fillDiagramFace(item, face);
      });
    }
    diagramFacesKey = facesKey;
  }
  alignArchitectureImages();
  $('diagram-activation-note').textContent = `Convolution faces: ${activationView === 'post' ? 'after' : 'before'} ReLU`;
  updateDiagramSelection();
  const references = {
    'convolution': 'https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer',
    '1 × 1 convolution': 'https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer',
    'relu': 'https://en.wikipedia.org/wiki/Rectifier_(neural_networks)',
    'pooling': 'https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer',
    'upsampling': 'https://en.wikipedia.org/wiki/Convolutional_neural_network#Deconvolutional',
    'sigmoid': 'https://en.wikipedia.org/wiki/Sigmoid_function',
  };
  $('stage-note').replaceChildren(...feature.note.replace('The view selects before or after ReLU.', feature.has_relu ? `Shown ${activationView === 'post' ? 'after' : 'before'} ReLU.` : '').split(/(1 × 1 convolution|convolution|ReLU|pooling|upsampling|sigmoid)/gi).map(text => {
    const url = references[text.toLowerCase()];
    if (!url) return document.createTextNode(text);
    const link = document.createElement('a');
    link.textContent = text;
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    return link;
  }));
}
function applyDisplaySize() {
  $('channels').classList.toggle('uniform', mobileMode.matches || displaySize === 'uniform');
  const size = !mobileMode.matches && displaySize === 'native' ? currentFeature().width : 64;
  $('channels').style.setProperty('--map-size', `${size}px`);
  $('channels').style.setProperty('--tile-size', `${size + 4}px`);
  $('uniform').setAttribute('aria-pressed', String(displaySize === 'uniform'));
  $('native').setAttribute('aria-pressed', String(displaySize === 'native'));
}
function renderGrid(feature) {
  const images = Array.from({length: feature.channels}, (_, channel) => channelImage(feature, channel));
  const fragment = document.createDocumentFragment();
  for (let index = 0; index < feature.channels; index++) {
    const button = document.createElement('button');
    button.className = 'channel';
    button.setAttribute('aria-label', `Channel ${index + 1}`);
    button.setAttribute('aria-pressed', String(selectionVisible && index === activeChannel));
    const image = document.createElement('img');
    image.src = images[index];
    image.alt = `Channel ${index + 1}`;
    const number = document.createElement('span');
    number.className = 'number';
    number.textContent = index + 1;
    button.append(image, number);
    button.onclick = () => selectChannel(index);
    fragment.appendChild(button);
  }
  $('channels').replaceChildren(fragment);
  $('channels').parentElement.scrollTop = 0;
  $('maps-info').textContent = `${feature.channels} ${feature.channels === 1 ? 'channel' : 'channels'} · ${feature.width} × ${feature.height} px`;
  applyDisplaySize();
}
function renderPreview() {
  updateDiagramSelection();
  const feature = currentFeature();
  $('preview').src = channelImage(feature, activeChannel);
  $('preview').alt = `${feature.label}, channel ${activeChannel + 1}`;
  $('selected-size').textContent = `${feature.width} × ${feature.height}`;
  $('selected-info').setAttribute('aria-label', `${feature.width} by ${feature.height} pixels`);
  const probabilityView = feature.has_sigmoid && sigmoidView === 'post';
  const source = channelSource(feature, activeChannel);
  const view = source.feature.views[source.name];
  const signed = view.signed;
  const peak = view.peaks[source.channel];
  const format = value => Number(value.toPrecision(3)).toString();
  $('color-strip').style.background = signed
    ? 'linear-gradient(to right, #2166ac, #f7f7f7, #b2182b)'
    : 'linear-gradient(to right, #000, #fff)';
  if (!peak) $('color-strip').style.background = signed ? '#f7f7f7' : '#000';
  $('color-min').textContent = signed ? format(-peak) : '0';
  $('color-zero').textContent = '0';
  $('color-zero').hidden = !signed || !peak;
  document.querySelector('.colorbar').classList.toggle('signed', signed && peak > 0);
  $('color-max').textContent = format(peak);
  $('color-caption').textContent = probabilityView ? 'Probability' : peak
    ? `${feature.id === 'final_conv' ? 'Logits' : 'Activation'}` : 'All zero';
  $('color-caption').title = probabilityView ? 'Fixed 0–1 scale' : 'Per-channel scale';
  $('channel-position').textContent = `${activeChannel + 1} / ${feature.channels}`;
  $('previous-channel').disabled = activeChannel === 0;
  $('next-channel').disabled = activeChannel === feature.channels - 1;
}
function selectChannel(index, reveal = false, showSelection = true) {
  if (forwardPlaying) finishForwardPass();
  if (updatingSample || $('channels').parentElement.getAttribute('aria-busy') === 'true') return;
  selectionVisible = showSelection;
  activeChannel = index;
  renderPreview();
  $('channels').querySelectorAll('button').forEach((button, channel) =>
    button.setAttribute('aria-pressed', String(selectionVisible && channel === index)));
  if (reveal) {
    const grid = $('channels').parentElement;
    const button = $('channels').children[index];
    const top = button.getBoundingClientRect().top - grid.getBoundingClientRect().top;
    if (top < 0) grid.scrollTop += top;
    else if (top + button.offsetHeight > grid.clientHeight) grid.scrollTop += top + button.offsetHeight - grid.clientHeight;
  }
}
function dismissSelection() {
  if (updatingSample || !features.length) return;
  selectionVisible = false;
  collapseDiagram();
  activeChannel = 0;
  renderPreview();
  $('channels').querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', 'false'));
}
document.addEventListener('click', event => {
  if (mobileMode.matches) return;
  if (event.target.closest('button, select, input, label, a, .stack, .operation, #preview, .output-column')) return;
  dismissSelection();
});
async function selectLayer(id, resetChannel = true, channel = null) {
  if (forwardPlaying) finishForwardPass();
  if (updatingSample || !features.length) return;
  selectionVisible = true;
  const version = ++selectionVersion;
  const currentFeatures = features;
  activeLayer = id;
  const feature = currentFeature();
  renderSelectors(feature);
  renderActivationControl(feature);
  $('channels').parentElement.setAttribute('aria-busy', 'true');
  try {
    const [svgText] = await Promise.all([
      mobileMode.matches ? Promise.resolve('') : loadDiagram(), prepareDiagramViews(features, args.data.encoding)
    ]);
    // A newer layer choice or sample must never be overwritten by an older request.
    if (version !== selectionVersion || currentFeatures !== features) return;
    if (resetChannel) activeChannel = channel ?? 0;
    lastLayers.set(feature.block, id);
    renderDiagram(feature, svgText);
    renderGrid(feature);
    $('channels').parentElement.setAttribute('aria-busy', 'false');
    selectChannel(activeChannel, channel !== null, selectionVisible);
  } catch (error) {
    if (version !== selectionVersion || currentFeatures !== features) return;
    $('status').textContent = error.message;
    $('maps-info').textContent = 'Could not load feature maps.';
  }
  if (version === selectionVersion) $('channels').parentElement.setAttribute('aria-busy', 'false');
}
function selectBlock(block) {
  selectLayer(lastLayers.get(block) || features.find(feature => feature.block === block).id);
}
function renderMasks() {
  const opacity = Number($('opacity').value);
  const overlay = opacity > 0 && !($('simplified').checked && !mobileMode.matches);
  const level = opacity ? `${Math.round(opacity * 100)}%` : 'Off';
  $('overlay-value').textContent = level;
  $('opacity').setAttribute('aria-valuetext', opacity ? `${level} mask opacity on input` : 'Off: binary masks');
  $('hero-prediction').src = overlay ? pixelImage(inputPixels, 128, 128, prediction, opacity) : predictionUrl;
  if (reference) $('hero-reference').src = overlay ? pixelImage(inputPixels, 128, 128, reference, opacity) : referenceUrl;
  $('reference-figure').hidden = !reference;
  document.querySelector('.hero').classList.toggle('no-reference', !reference);
}
function showForwardFeature(feature, probability = false) {
  activeLayer = feature.id;
  activeChannel = 0;
  selectionVisible = true;
  if (feature.has_sigmoid) sigmoidView = probability ? 'post' : 'pre';
  lastLayers.set(feature.block, feature.id);
  renderSelectors(feature);
  renderActivationControl(feature);
  renderDiagram(feature, currentDiagram);
  renderGrid(feature);
  $('channels').parentElement.setAttribute('aria-busy', 'false');
  renderPreview();
  $('explorer').classList.remove('forward-awaiting-feature');
}
function finishForwardPass(showOutput = false) {
  ++forwardRun;
  forwardPlaying = false;
  for (const animation of forwardAnimations) animation.cancel();
  forwardAnimations.clear();
  const svg = $('diagram').querySelector('svg');
  if (svg) {
    svg.querySelectorAll('.stack, .operation').forEach(node => { node.style.opacity = '1'; });
    svg.querySelectorAll('.edge[mask]').forEach(edge => edge.removeAttribute('mask'));
    svg.querySelectorAll('[data-forward-mask]').forEach(mask => mask.remove());
    svg.querySelectorAll('[data-forward-current]').forEach(stack => delete stack.dataset.forwardCurrent);
  }
  $('hero-prediction').closest('figure').style.opacity = '1';
  $('explorer').classList.remove('forward-playing', 'forward-preparing', 'forward-awaiting-feature');
  $('forward-progress').textContent = `${features.length} / ${features.length}`;
  $('forward-progress').title = 'Illustrative timing of the model’s forward pass';
  $('forward-toggle').setAttribute('aria-label', 'Replay forward pass');
  $('forward-toggle').title = 'Replay forward pass';
  $('forward-toggle').querySelector('path').setAttribute('d', 'M8 5v14l11-7z');
  if (showOutput && features.length) showForwardFeature(features[features.length - 1], true);
}
async function animateForward(node, frames, duration) {
  const animation = node.animate(frames, {duration, easing: 'ease-in-out', fill: 'forwards'});
  forwardAnimations.add(animation);
  let completed = false;
  try { await animation.finished; completed = true; }
  catch (error) { if (error.name !== 'AbortError') throw error; }
  finally {
    if (completed) {
      for (const [property, value] of Object.entries(frames[frames.length - 1])) node.style[property] = value;
      animation.cancel();
    }
    forwardAnimations.delete(animation);
  }
}
async function revealOperation(operation, run) {
  const edge = operation.querySelector('.edge');
  const caption = operation.querySelector('.edge-caption');
  const length = edge.getTotalLength();
  const mask = document.createElementNS(namespace, 'mask');
  mask.id = `forward-mask-${run}-${$('diagram').querySelectorAll('[data-forward-mask]').length}`;
  mask.dataset.forwardMask = 'true';
  mask.setAttribute('maskUnits', 'userSpaceOnUse');
  mask.setAttribute('x', '-100'); mask.setAttribute('y', '-100');
  const bounds = $('diagram').querySelector('svg').viewBox.baseVal;
  mask.setAttribute('width', String(bounds.width + 200)); mask.setAttribute('height', String(bounds.height + 200));
  const trace = document.createElementNS(namespace, 'path');
  trace.setAttribute('d', edge.getAttribute('d'));
  trace.setAttribute('fill', 'none'); trace.setAttribute('stroke', 'white');
  trace.setAttribute('stroke-width', '28'); trace.setAttribute('stroke-linecap', 'round');
  trace.style.strokeDasharray = `${length} ${length}`;
  trace.style.strokeDashoffset = String(length);
  mask.appendChild(trace);
  $('diagram').querySelector('defs').appendChild(mask);
  edge.setAttribute('mask', `url(#${mask.id})`);
  operation.style.opacity = '1';
  await Promise.all([
    animateForward(trace, [{strokeDashoffset: String(length)}, {strokeDashoffset: '0'}], 220),
    animateForward(caption, [{opacity: 0}, {opacity: 1}], 220)
  ]);
  if (edge.getAttribute('mask') === `url(#${mask.id})`) edge.removeAttribute('mask');
  mask.remove();
}
async function playForwardPass() {
  if (!features.length || updatingSample || mobileMode.matches) return;
  finishForwardPass();
  ++selectionVersion;
  const run = forwardRun;
  const passFeatures = features;
  const outputFeature = passFeatures[passFeatures.length - 1];
  forwardPlaying = true;
  try {
    await Promise.all(['pre', 'post'].map(view => decodeView(outputFeature, view, args.data.encoding)));
    if (run !== forwardRun || passFeatures !== features) return;
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      showForwardFeature(outputFeature, true);
      return;
    }
    dismissSelection();
    const svg = $('diagram').querySelector('svg');
    const operations = [...svg.querySelectorAll('.operation')];
    $('explorer').classList.add('forward-playing', 'forward-awaiting-feature');
    $('forward-toggle').setAttribute('aria-label', 'Skip forward-pass animation');
    $('forward-toggle').title = 'Skip animation and show all outputs';
    $('forward-toggle').querySelector('path').setAttribute('d', 'M6 6h12v12H6z');
    svg.querySelectorAll('.stack, .operation').forEach(node => { node.style.opacity = '0'; });
    $('hero-prediction').closest('figure').style.opacity = '0';
    // The backend's stage list follows model.forward(). Skip stages copy saved encoder outputs.
    for (const [index, feature] of features.entries()) {
      if (run !== forwardRun) return;
      $('forward-progress').textContent = `${index + 1} / ${features.length}`;
      $('forward-progress').title = feature.label;
      const incoming = operations.filter(operation => {
        const edge = operation.querySelector('.edge');
        return edge.dataset.to === feature.id || (index === 0 && !edge.dataset.from && edge.getAttribute('d').startsWith('M0 '));
      });
      // Both concat inputs must arrive before the combined tensor is revealed.
      await Promise.all(incoming.map(operation => revealOperation(operation, run)));
      if (run !== forwardRun) return;
      svg.querySelectorAll('[data-forward-current]').forEach(stack => delete stack.dataset.forwardCurrent);
      const stack = $(feature.svg_id);
      stack.dataset.forwardCurrent = 'true';
      showForwardFeature(feature);
      await animateForward(stack, [{opacity: 0}, {opacity: 1}], 180);
      if (run !== forwardRun) return;
      stack.style.opacity = '1';
    }
    const output = operations.find(operation => {
      const edge = operation.querySelector('.edge');
      return !edge.dataset.to && !edge.getAttribute('d').startsWith('M0 ');
    });
    $('forward-progress').textContent = `${features.length} / ${features.length}`;
    $('forward-progress').title = 'Sigmoid → threshold 0.5';
    if (output) await revealOperation(output, run);
    if (run !== forwardRun) return;
    showForwardFeature(outputFeature, true);
    await animateForward($('hero-prediction').closest('figure'), [{opacity: 0}, {opacity: 1}], 220);
  } finally {
    if (run === forwardRun) finishForwardPass();
  }
}
function requestInput(action, extra = {}) {
  if (forwardPlaying) finishForwardPass();
  $('status').textContent = 'Loading…';
  $('examples').querySelectorAll('button').forEach(button => button.disabled = true);
  $('upload').disabled = true;
  clearTimeout(inputTimeout);
  inputTimeout = setTimeout(() => {
    $('status').textContent = 'The server is taking longer than expected. Please try again.';
    $('examples').querySelectorAll('button').forEach(button => button.disabled = false);
    $('upload').disabled = false;
  }, 30000);
  send('streamlit:setComponentValue', {value: {action, event: Date.now(), ...extra}, dataType: 'json'});
}
blocks.forEach(block => $('block').add(new Option(block, block)));
$('block').onchange = () => selectBlock($('block').value);
$('layer').onchange = () => selectLayer($('layer').value);
$('previous-layer').onclick = () => selectLayer(features[features.findIndex(f => f.id === activeLayer) - 1].id);
$('next-layer').onclick = () => selectLayer(features[features.findIndex(f => f.id === activeLayer) + 1].id);
$('previous-channel').onclick = () => selectChannel(activeChannel - 1, true);
$('next-channel').onclick = () => selectChannel(activeChannel + 1, true);
for (const mode of ['uniform', 'native']) $(mode).onclick = () => { displaySize = mode; applyDisplaySize(); };
for (const mode of ['pre', 'post']) $(`activation-${mode}`).onclick = async () => {
  const feature = currentFeature();
  if ((feature.has_sigmoid ? sigmoidView : activationView) === mode) return;
  const scroll = $('channels').parentElement;
  const position = scroll.scrollTop;
  if (feature.has_sigmoid) sigmoidView = mode;
  else activationView = mode;
  renderActivationControl(feature);
  await selectLayer(activeLayer, false);
  scroll.scrollTop = position;
};
$('simplified').onchange = () => {
  if (forwardPlaying) finishForwardPass();
  collapseDiagram();
  $('diagram').querySelectorAll('.hover-face').forEach(face => face.remove());
  updateDiagramMode();
  renderMasks();
  alignArchitectureImages();
};
let maskFrame;
$('opacity').oninput = () => {
  cancelAnimationFrame(maskFrame);
  maskFrame = requestAnimationFrame(renderMasks);
};
$('forward-toggle').onclick = () => {
  if (forwardPlaying) finishForwardPass(true);
  else playForwardPass().catch(error => { $('status').textContent = error.message; });
};
function renderExamples(items, active) {
  if (!$('examples').children.length) {
    items.forEach(item => {
      const button = document.createElement('button');
      button.dataset.name = item.name;
      button.title = `Validation example: ${item.name}`;
      button.setAttribute('aria-label', button.title);
      const image = document.createElement('img');
      image.src = `data:image/png;base64,${item.image}`;
      image.alt = '';
      button.appendChild(image);
      button.onclick = () => { if (button.dataset.name !== args.active_example) requestInput('example', {name: item.name}); };
      $('examples').appendChild(button);
    });
  }
  $('examples').querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.name === active)));
}
$('upload').onclick = () => $('upload-file').click();
$('upload-file').onchange = () => {
  const file = $('upload-file').files[0];
  if (!file) return;
  if (!['image/png', 'image/jpeg'].includes(file.type) || file.size > 5 * 1024 * 1024) {
    $('status').textContent = 'Please upload a PNG or JPEG up to 5 MB.';
    $('upload-file').value = '';
    return;
  }
  const reader = new FileReader();
  reader.onload = () => requestInput('upload', {bytes: reader.result.split(',')[1]});
  reader.onerror = () => { $('status').textContent = 'Could not read this image. Please try again.'; };
  reader.readAsDataURL(file);
  $('upload-file').value = '';
};
window.addEventListener('message', async event => {
  if (event.source !== window.parent || event.data?.type !== 'streamlit:render') return;
  const nextArgs = event.data.args;
  if (!nextArgs?.data) return;
  clearTimeout(inputTimeout);
  const changed = sampleId !== nextArgs.sample_id || referenceSource !== nextArgs.reference;
  const version = changed ? ++sampleVersion : sampleVersion;
  if (changed) {
    if (forwardPlaying) finishForwardPass();
    updatingSample = true;
    ++selectionVersion;
    $('explorer').setAttribute('aria-busy', 'true');
    setControlsDisabled(true);
    try {
      await Promise.all([
        mobileMode.matches ? Promise.resolve('') : loadDiagram(),
        prepareDiagramViews(nextArgs.data.features, nextArgs.data.encoding)
      ]);
    } catch (error) {
      if (version !== sampleVersion) return;
      updatingSample = false;
      $('explorer').setAttribute('aria-busy', 'false');
      if (features.length) {
        setControlsDisabled(false);
        renderSelectors(currentFeature());
        renderActivationControl(currentFeature());
      }
      $('examples').querySelectorAll('button').forEach(button => button.disabled = false);
      $('upload').disabled = false;
      $('status').textContent = error.message;
      return;
    }
    if (version !== sampleVersion) return;
    // Commit the new input and features together only after its selected view is ready.
    args = nextArgs;
    renderExamples(args.examples, args.active_example);
    sampleId = args.sample_id;
    referenceSource = args.reference;
    features = args.data.features;
    imageCache.clear();
    inputPixels = bytes(args.data.input);
    prediction = bytes(args.data.prediction);
    reference = args.reference ? bytes(args.reference) : null;
    inputUrl = pixelImage(inputPixels, 128, 128);
    predictionUrl = pixelImage(Uint8Array.from(prediction, value => value * 255), 128, 128);
    referenceUrl = reference ? pixelImage(Uint8Array.from(reference, value => value * 255), 128, 128) : null;
    $('explorer').classList.add('forward-preparing');
    activeLayer = 'enc_l1_conv1';
    activeChannel = 0;
    $('hero-input').src = inputUrl;
    updatingSample = false;
    setControlsDisabled(false);
    await selectLayer(activeLayer);
    if (version !== sampleVersion) return;
    renderMasks();
    $('explorer').classList.remove('loading');
    $('explorer').setAttribute('aria-busy', 'false');
    if (mobileMode.matches) $('explorer').classList.remove('forward-preparing');
    else playForwardPass().catch(error => { $('status').textContent = error.message; });
  }
  if (version !== sampleVersion) return;
  $('status').textContent = nextArgs.error || '';
  $('examples').querySelectorAll('button').forEach(button => button.disabled = false);
  $('upload').disabled = false;
  updateHeight();
});
function setControlsDisabled(disabled) {
  document.querySelectorAll('button, select, input').forEach(control => control.disabled = disabled);
}
setControlsDisabled(true);
if (!mobileMode.matches) loadDiagram().catch(error => { $('status').textContent = error.message; });
function alignArchitectureImages() {
  if (mobileMode.matches) return;
  const row = document.querySelector('.architecture-row');
  const hero = document.querySelector('.hero');
  const svg = $('diagram').querySelector('svg');
  if (!svg) return;
  const style = getComputedStyle(hero);
  const toolbar = document.querySelector('.diagram-toolbar');
  const footer = document.querySelector('.diagram-footer');
  const verticalGaps = parseFloat(getComputedStyle(toolbar).marginTop)
    + (footer.offsetHeight ? parseFloat(getComputedStyle(footer).marginTop) : 0);
  const availableWidth = hero.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
  const availableHeight = hero.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom)
    - footer.offsetHeight - toolbar.offsetHeight - verticalGaps;
  const imageSize = Math.min(96, Math.max(72, availableHeight * .28));
  const sideWidth = innerWidth >= 1101 ? 2 * imageSize + 24 : imageSize;
  const wideLayout = innerWidth >= 1101;
  const rowLimit = wideLayout ? availableWidth - 2 * Math.max(0, 180 - imageSize - 12) : availableWidth;
  const bounds = svg.viewBox.baseVal;
  const bottom = bounds.y + bounds.height;
  const inputY = Number(svg.dataset.inputY);
  const scale = Math.min((rowLimit - 2 * sideWidth) / bounds.width,
    Math.max(1, availableHeight - imageSize / 2) / (bottom - inputY));
  row.style.setProperty('--hero-image-size', `${imageSize}px`);
  row.style.setProperty('--hero-side-width', `${sideWidth}px`);
  const rowWidth = scale * bounds.width + 2 * sideWidth;
  row.style.width = `${rowWidth}px`;
  const dividerX = wideLayout
    ? (availableWidth - rowWidth) / 2 + imageSize + 12 : 180;
  hero.style.setProperty('--diagram-divider-x', `${dividerX}px`);
  // Input and prediction centers stay aligned with the first and last arrows.
  const top = Math.max(0, inputY - imageSize / (2 * scale));
  svg.setAttribute('viewBox', `0 ${top} ${bounds.width} ${bottom - top}`);
  row.style.setProperty('--hero-image-top', '0px');
}
function fitFeaturePanels() {
  const scroll = $('channels').parentElement;
  const columns = innerWidth <= 650 ? 4 : innerWidth <= 900 ? 8 : 16;
  const mapSize = Math.min(64, Math.max(0, (scroll.clientWidth - 8 - 4 * (columns - 1)) / columns - 4));
  const gridHeight = 8 + 2 * (mapSize + 21) + 4 + 12;
  const height = Math.ceil(document.querySelector('.maps-heading').offsetHeight + 8 + 24 + gridHeight);
  if (!mobileMode.matches) document.querySelector('.workspace').style.setProperty('--workspace-height', `${height}px`);
  document.querySelector('.grid-viewport').dataset.overflow = String(scroll.scrollHeight > scroll.clientHeight + 1);
}
const wideDiagram = matchMedia('(min-width: 1101px)');
const overlayControls = document.querySelector('.output-controls');
function placeDiagramControls() {
  const footer = document.querySelector('.diagram-footer');
  if (wideDiagram.matches) {
    document.querySelector('.output-column').appendChild(overlayControls);
  } else {
    footer.appendChild(overlayControls);
  }
  alignArchitectureImages();
}
wideDiagram.addEventListener('change', placeDiagramControls);
placeDiagramControls();
const layerControls = document.querySelector('.layer-controls');
const desktopLayerParent = layerControls.parentElement;
const activationControl = $('activation');
const desktopActivationParent = activationControl.parentElement;
const summary = document.querySelector('.header > .summary');
blocks.forEach(block => {
  const button = document.createElement('button');
  button.textContent = block;
  button.dataset.block = block;
  button.disabled = !features.length;
  button.setAttribute('aria-pressed', String(block === 'Encoder'));
  button.onclick = () => selectBlock(block);
  $('mobile-blocks').appendChild(button);
});
function applyResponsiveLayout() {
  if (forwardPlaying) finishForwardPass();
  if (mobileMode.matches) {
    $('mobile-layer-slot').appendChild(layerControls);
    $('mobile-activation-slot').appendChild(activationControl);
    $('mobile-model-details').appendChild(summary);
  } else {
    desktopLayerParent.prepend(layerControls);
    desktopActivationParent.prepend(activationControl);
    document.querySelector('.header').appendChild(summary);
  }
  if (features.length) {
    selectLayer(activeLayer, false);
    renderMasks();
  }
  updateHeight();
}
mobileMode.addEventListener('change', applyResponsiveLayout);
applyResponsiveLayout();
const featureLayout = new ResizeObserver(fitFeaturePanels);
featureLayout.observe(document.querySelector('.maps-heading'));
featureLayout.observe($('channels'));
const diagramLayout = new ResizeObserver(alignArchitectureImages);
for (const selector of ['.hero', '.diagram-toolbar', '.diagram-footer']) diagramLayout.observe(document.querySelector(selector));
new ResizeObserver(updateHeight).observe($('explorer'));
window.addEventListener('resize', updateHeight);
send('streamlit:componentReady', {apiVersion: 1});
updateHeight();
