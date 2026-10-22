import { mobileSteps, renderMobileStep } from './mobile-pass.mjs';

const $ = id => document.getElementById(id);
const blocks = ['Encoder', 'Bottleneck', 'Decoder', 'Output'];
const mobileMode = matchMedia('(max-width: 650px)');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
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
let forwardRun = 0, forwardPlaying = false, forwardStep = null, forwardIndex = 0;
const forwardAnimations = new Set();
let mobilePlan = [], mobileIndex = 0, mobilePlaying = false;
let mobileTimer, mobileWake;
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
    if (mobileMode.matches && feature.has_relu) names.add('post');
    if (mobileMode.matches && feature.has_sigmoid) { names.add('pre'); names.add('post'); }
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
function channelImage(feature, channel, forcedView = null) {
  // Concatenation copies channels. Keep their source colors and per-channel scale.
  const source = channelSource(feature, channel);
  if (forcedView && !feature.id.endsWith('_concat') && !feature.id.endsWith('_skip')) source.name = forcedView;
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
  const choices = navigationFeatures().filter(item => item.block === feature.block);
  const key = choices.map(item => item.id).join(',');
  if (select.dataset.choices !== key) {
    select.dataset.choices = key;
    select.replaceChildren();
    let lastGroup;
    for (const item of choices) {
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
  }
  select.value = feature.id;
  const items = navigationFeatures();
  const index = items.findIndex(item => item.id === feature.id);
  $('previous-layer').disabled = index === 0;
  $('next-layer').disabled = index === items.length - 1;
}
function navigationFeatures() {
  return mobileMode.matches ? features.filter(feature => !feature.id.endsWith('_skip')) : features;
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
      if (forwardPlaying || forwardStep !== null || $('simplified').checked) return;
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
  $('final_conv').querySelector('.output-label').textContent = sigmoidView === 'post' ? 'Probability' : 'Logits';
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
  if (mobileMode.matches) $('pass-stage').querySelectorAll('[data-channel]').forEach(button =>
    button.setAttribute('aria-pressed', String(Number(button.dataset.channel) === activeChannel)));
}
function selectChannel(index, reveal = false, showSelection = true) {
  if (mobilePlaying) {
    const id = activeLayer;
    pauseMobilePass();
    const step = mobilePlan.findIndex(item => item.feature.id === id);
    showMobileStep(step + 1, false, true, index).catch(error => { $('status').textContent = error.message; });
    return;
  }
  else if (forwardPlaying) finishForwardPass();
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
  if (mobileMode.matches && $('explorer').dataset.mobilePhase === 'walk') {
    const index = mobilePlan[mobileIndex - 1]?.feature.id === id
      ? mobileIndex - 1 : mobilePlan.findIndex(step => step.feature.id === id);
    if (index >= 0) {
      pauseMobilePass();
      await showMobileStep(index + 1, false, !resetChannel, channel);
      return;
    }
  }
  if (forwardPlaying || forwardStep !== null) finishForwardPass();
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
  const items = navigationFeatures();
  const saved = items.find(feature => feature.id === lastLayers.get(block));
  selectLayer(saved?.id || items.find(feature => feature.block === block).id);
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
function showForwardFeature(feature, probability = false, channel = 0) {
  activeLayer = feature.id;
  activeChannel = channel;
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

function updateMobileControls() {
  const phase = $('explorer').dataset.mobilePhase;
  const complete = phase === 'walk' && mobileIndex === mobilePlan.length;
  $('pass-progress').textContent = `${mobileIndex} / ${mobilePlan.length || 19}`;
  $('pass-progress').setAttribute('aria-live', mobilePlaying ? 'off' : 'polite');
  $('pass-previous').disabled = updatingSample || mobileIndex === 0;
  $('pass-next').disabled = updatingSample || mobileIndex === mobilePlan.length;
  $('pass-play').disabled = updatingSample || !mobilePlan.length;
  $('pass-finish').disabled = updatingSample || !mobilePlan.length;
  $('pass-play').setAttribute('aria-pressed', String(mobilePlaying));
  $('pass-play').setAttribute('aria-label', mobilePlaying ? 'Pause forward pass'
    : phase === 'result' ? 'Replay forward pass' : complete ? 'Compare results'
    : reducedMotion.matches ? 'Next forward-pass step' : 'Play forward pass');
  $('pass-play').querySelector('span').textContent = mobilePlaying ? 'Pause' : phase === 'result' ? 'Replay'
    : complete ? 'Compare results' : reducedMotion.matches ? phase === 'ready' ? 'Step through' : 'Next step' : 'Forward pass';
  $('pass-play').querySelector('path').setAttribute('d', mobilePlaying ? 'M7 5h3v14H7zM14 5h3v14h-3z' : 'M8 5v14l11-7z');
}
function pauseMobilePass() {
  const wasPlaying = mobilePlaying;
  ++forwardRun;
  ++selectionVersion;
  mobilePlaying = false;
  forwardPlaying = false;
  clearTimeout(mobileTimer);
  if (mobileWake) { mobileWake(); mobileWake = null; }
  for (const animation of forwardAnimations) animation.cancel();
  forwardAnimations.clear();
  const result = $('pass-stage').querySelector('.pass-result');
  if (result) { result.style.opacity = '1'; result.style.transform = 'none'; result.inert = false; }
  const step = mobilePlan[mobileIndex - 1];
  if (wasPlaying && mobileMode.matches && step && !updatingSample &&
      (activeLayer !== step.feature.id || (step.feature.has_sigmoid && sigmoidView !== step.view))) {
    showForwardFeature(step.feature, step.view === 'post');
  }
  updateMobileControls();
}
function resetMobilePass() {
  mobilePlan = mobileSteps(features);
  mobileIndex = 0;
  $('explorer').dataset.mobilePhase = 'ready';
  $('pass-heading').textContent = 'From image to segmentation';
  $('pass-input').src = inputUrl;
  $('pass-reference').src = referenceUrl || inputUrl;
  $('pass-reference-figure').hidden = !reference;
  document.querySelector('.pass-start-note').innerText = reference
    ? 'Follow the image through the network.\nThe reference mask is only for comparison.'
    : 'Follow the image through the network.\nNo reference mask was provided.';
  $('pass-ready').hidden = false;
  $('pass-stage').hidden = true;
  $('pass-note').textContent = '';
  forwardStep = null;
  updateMobileControls();
}
async function showMobileStep(index, animate = false, keepView = false, channel = null) {
  if (!features.length || updatingSample) return;
  if (index <= 0) { resetMobilePass(); return; }
  const run = forwardRun, version = ++selectionVersion;
  const step = mobilePlan[Math.min(index, mobilePlan.length) - 1];
  const oldChannel = activeLayer === step.feature.id ? activeChannel : 0;
  if (!keepView) activationView = 'post';
  await prepareDiagramViews(features, args.data.encoding);
  if (run !== forwardRun || version !== selectionVersion || !mobileMode.matches) return;
  const previousImages = new Map();
  const imageKey = image => `${image.dataset.tensor}:${image.dataset.view}:${image.dataset.mapChannel}`;
  if (animate) {
    const phase = $('explorer').dataset.mobilePhase;
    if (phase === 'walk') $('pass-stage').querySelectorAll('.pass-result img').forEach(image =>
      previousImages.set(imageKey(image), image.getBoundingClientRect()));
    else previousImages.set('input:output:0', $(phase === 'ready' ? 'pass-input' : 'hero-input').getBoundingClientRect());
  }
  mobileIndex = index;
  forwardStep = index;
  $('explorer').dataset.mobilePhase = 'walk';
  $('pass-heading').textContent = step.label;
  $('pass-ready').hidden = true;
  $('pass-stage').hidden = false;
  $('pass-stage').dataset.step = step.id;
  $('pass-stage').dataset.feature = step.feature.id;
  renderMobileStep($('pass-stage'), step, features, channelImage, inputUrl, predictionUrl, selected => {
    pauseMobilePass();
    selectChannel(selected, true);
  });
  $('pass-note').textContent = step.note;
  const output = $('pass-stage').querySelector('.pass-result');
  output.inert = animate;
  if (animate) output.style.opacity = '0';
  updateMobileControls();
  if (animate) {
    const moving = [...$('pass-stage').querySelectorAll('.pass-inputs .face img')]
      .filter(image => previousImages.has(imageKey(image)))
      .map(image => ({image, from: previousImages.get(imageKey(image)), to: image.getBoundingClientRect()}));
    await Promise.all(moving.map(({image, from, to}) => moveMobileImage(image, from, to)));
  }
  if (run !== forwardRun || version !== selectionVersion) return;
  if (animate) await animateForward($('pass-stage').querySelector('.pass-operation-pill'), [{opacity: .35}, {opacity: 1}], 160);
  if (run !== forwardRun || version !== selectionVersion) return;
  const inspection = keepView && step.feature.has_sigmoid ? sigmoidView === 'post' : step.view === 'post';
  showForwardFeature(step.feature, inspection, channel ?? (keepView ? oldChannel : 0));
  if (channel !== null) selectChannel(channel, true);
  output.inert = false;
  if (animate && step.kind === 'concat') {
    output.style.opacity = '1';
    const sources = [...$('pass-stage').querySelectorAll('.pass-inputs .pass-stack')].map(stack => stack.getBoundingClientRect());
    const targets = [...output.querySelectorAll('.pass-stack')].map(image => ({image, to: image.getBoundingClientRect()}));
    await Promise.all(targets.map(({image, to}, index) => moveMobileImage(image, sources[index], to)));
  } else if (animate) await animateForward(output, [{opacity: 0, transform: 'translateY(3px)'}, {opacity: 1, transform: 'translateY(0)'}], 200);
}
function moveMobileImage(image, from, to) {
  image.style.transformOrigin = 'top left';
  return animateForward(image, [
    {opacity: .55, transform: `translate(${from.x - to.x}px, ${from.y - to.y}px) scale(${from.width / to.width}, ${from.height / to.height})`},
    {opacity: 1, transform: 'translate(0, 0) scale(1)'}
  ], 200);
}
function showMobileResults(keepSelection = false) {
  pauseMobilePass();
  mobileIndex = mobilePlan.length;
  forwardStep = null;
  $('explorer').dataset.mobilePhase = 'result';
  $('pass-heading').textContent = 'Input and segmentation';
  if (!keepSelection) showForwardFeature(features[features.length - 1], true);
  renderMasks();
  updateMobileControls();
}
async function stepMobilePass(direction) {
  if (updatingSample || !mobilePlan.length) return;
  pauseMobilePass();
  const index = Math.max(0, Math.min(mobilePlan.length, mobileIndex + direction));
  mobileIndex = index;
  await showMobileStep(index);
}
async function playMobilePass() {
  if (updatingSample || !mobilePlan.length) return;
  pauseMobilePass();
  if ($('explorer').dataset.mobilePhase === 'walk' && mobileIndex === mobilePlan.length) {
    showMobileResults();
    return;
  }
  const start = $('explorer').dataset.mobilePhase === 'result' ? 0 : mobileIndex;
  if (reducedMotion.matches) {
    await showMobileStep(start + 1);
    return;
  }
  const run = forwardRun;
  mobilePlaying = true;
  forwardPlaying = true;
  try {
    for (let index = start + 1; index <= mobilePlan.length; index++) {
      await showMobileStep(index, true);
      if (run !== forwardRun || !mobileMode.matches) return;
      await new Promise(resolve => {
        mobileWake = resolve;
        mobileTimer = setTimeout(() => { mobileWake = null; resolve(); }, mobilePlan[index - 1].kind === 'concat' ? 1100 : 650);
      });
      if (run !== forwardRun) return;
    }
    showMobileResults();
  } catch (error) {
    if (run === forwardRun) { pauseMobilePass(); $('status').textContent = error.message; }
  }
}
function updateForwardControls() {
  const total = features.length + 1; // Tensor outputs, then sigmoid / binary prediction.
  $('forward-progress').textContent = `${forwardIndex} / ${total}`;
  $('forward-previous').disabled = updatingSample || (!forwardPlaying && forwardStep === null) || forwardIndex === 0;
  $('forward-next').disabled = updatingSample || forwardStep === total;
  $('forward-next').title = forwardStep === null && !forwardPlaying ? 'Start step-by-step' : 'Next step';
  if (!forwardPlaying) {
    $('forward-toggle').setAttribute('aria-label', forwardStep === null ? 'Replay forward pass' : 'Continue forward pass');
    $('forward-toggle').title = forwardStep === null ? 'Replay forward pass' : 'Continue from this step';
  }
}
function renderForwardStep(step) {
  const svg = $('diagram').querySelector('svg');
  const visible = new Set(features.slice(0, step).map(feature => feature.id));
  svg.querySelectorAll('.stack').forEach(stack => {
    const shown = visible.has(stack.id);
    stack.style.opacity = shown ? '1' : '0';
    stack.setAttribute('aria-hidden', String(!shown));
    delete stack.dataset.forwardCurrent;
  });
  svg.querySelectorAll('.operation').forEach(operation => {
    const edge = operation.querySelector('.edge');
    const shown = edge.dataset.to ? visible.has(edge.dataset.to) : !edge.dataset.from ? step > 0 : step > features.length;
    operation.style.opacity = shown ? '1' : '0';
    operation.setAttribute('aria-hidden', String(!shown));
    operation.setAttribute('tabindex', shown ? '0' : '-1');
  });
  $('hero-prediction').closest('figure').style.opacity = step > features.length ? '1' : '0';
  forwardIndex = step;
  $('forward-progress').title = step === 0 ? 'Input' : step > features.length
    ? 'Sigmoid → threshold 0.5' : features[step - 1].label;
  if (step) {
    const feature = features[Math.min(step, features.length) - 1];
    showForwardFeature(feature, step > features.length);
    $(feature.svg_id).dataset.forwardCurrent = 'true';
  } else {
    selectionVisible = false;
    updateDiagramSelection();
    $('explorer').classList.add('forward-awaiting-feature');
    $('stage-note').textContent = 'Input ready. The first convolution produces the first feature maps.';
  }
  updateForwardControls();
}
async function stepForward(direction) {
  if (!features.length || updatingSample || mobileMode.matches) return;
  const start = forwardStep ?? (forwardPlaying ? forwardIndex : 0);
  const step = Math.max(0, Math.min(features.length + 1, start + direction));
  finishForwardPass();
  const run = forwardRun;
  const version = ++selectionVersion;
  const output = features[features.length - 1];
  try {
    await Promise.all(['pre', 'post'].map(view => decodeView(output, view, args.data.encoding)));
    if (run !== forwardRun || version !== selectionVersion) return;
    collapseDiagram();
    forwardStep = step;
    $('explorer').classList.add('forward-stepping');
    renderForwardStep(step);
  } catch (error) { $('status').textContent = error.message; }
}
function finishForwardPass(showOutput = false) {
  if (mobilePlaying) pauseMobilePass();
  ++forwardRun;
  forwardPlaying = false;
  forwardStep = null;
  for (const animation of forwardAnimations) animation.cancel();
  forwardAnimations.clear();
  const svg = $('diagram').querySelector('svg');
  if (svg) {
    svg.querySelectorAll('.stack, .operation').forEach(node => {
      node.style.opacity = '1';
      node.removeAttribute('aria-hidden');
      if (node.classList.contains('operation')) node.setAttribute('tabindex', '0');
    });
    svg.querySelectorAll('.edge[mask]').forEach(edge => edge.removeAttribute('mask'));
    svg.querySelectorAll('[data-forward-mask]').forEach(mask => mask.remove());
    svg.querySelectorAll('[data-forward-current]').forEach(stack => delete stack.dataset.forwardCurrent);
  }
  $('hero-prediction').closest('figure').style.opacity = '1';
  $('explorer').classList.remove('forward-playing', 'forward-stepping', 'forward-preparing', 'forward-awaiting-feature');
  forwardIndex = features.length + 1;
  updateForwardControls();
  $('forward-progress').title = 'Illustrative timing of the model’s forward pass';
  $('forward-toggle').setAttribute('aria-label', 'Replay forward pass');
  $('forward-toggle').title = 'Replay forward pass';
  $('forward-toggle').querySelector('path').setAttribute('d', 'M8 5v14l11-7z');
  $('forward-toggle').querySelector('span').textContent = 'Forward pass';
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
  operation.removeAttribute('aria-hidden');
  operation.setAttribute('tabindex', '0');
  await Promise.all([
    animateForward(trace, [{strokeDashoffset: String(length)}, {strokeDashoffset: '0'}], 220),
    animateForward(caption, [{opacity: 0}, {opacity: 1}], 220)
  ]);
  if (edge.getAttribute('mask') === `url(#${mask.id})`) edge.removeAttribute('mask');
  mask.remove();
}
async function playForwardPass() {
  if (!features.length || updatingSample || mobileMode.matches) return;
  const start = forwardStep !== null && forwardStep < features.length + 1 ? forwardStep : 0;
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
    $('forward-toggle').querySelector('span').textContent = 'Skip';
    renderForwardStep(start);
    // The backend's stage list follows model.forward(). Skip stages copy saved encoder outputs.
    for (const [index, feature] of passFeatures.entries()) {
      if (index < start) continue;
      if (run !== forwardRun) return;
      $('forward-progress').textContent = `${index + 1} / ${features.length + 1}`;
      $('forward-progress').title = feature.label;
      const incoming = operations.filter(operation => {
        const edge = operation.querySelector('.edge');
        return edge.dataset.to === feature.id;
      });
      // Both concat inputs must arrive before the combined tensor is revealed.
      await Promise.all(incoming.map(operation => revealOperation(operation, run)));
      if (run !== forwardRun) return;
      svg.querySelectorAll('[data-forward-current]').forEach(stack => delete stack.dataset.forwardCurrent);
      const stack = $(feature.svg_id);
      stack.dataset.forwardCurrent = 'true';
      stack.removeAttribute('aria-hidden');
      showForwardFeature(feature);
      forwardIndex = index + 1;
      updateForwardControls();
      await animateForward(stack, [{opacity: 0}, {opacity: 1}], 180);
      if (run !== forwardRun) return;
      stack.style.opacity = '1';
    }
    const output = operations.find(operation => {
      const edge = operation.querySelector('.edge');
      return edge.dataset.from === outputFeature.id && !edge.dataset.to;
    });
    $('forward-progress').textContent = `${features.length + 1} / ${features.length + 1}`;
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
  if (forwardPlaying || forwardStep !== null) finishForwardPass();
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
$('layer').addEventListener('focus', () => {
  if (!mobilePlaying) return;
  const index = mobilePlan.findIndex(step => step.feature.id === activeLayer);
  pauseMobilePass();
  showMobileStep(index + 1, false, true).catch(error => { $('status').textContent = error.message; });
});
$('previous-layer').onclick = () => selectLayer(navigationFeatures()[navigationFeatures().findIndex(f => f.id === activeLayer) - 1].id);
$('next-layer').onclick = () => selectLayer(navigationFeatures()[navigationFeatures().findIndex(f => f.id === activeLayer) + 1].id);
$('previous-channel').onclick = () => selectChannel(activeChannel - 1, true);
$('next-channel').onclick = () => selectChannel(activeChannel + 1, true);
for (const mode of ['uniform', 'native']) $(mode).onclick = () => { displaySize = mode; applyDisplaySize(); };
for (const mode of ['pre', 'post']) $(`activation-${mode}`).onclick = async () => {
  const feature = currentFeature();
  if (mobilePlaying) pauseMobilePass();
  if ((feature.has_sigmoid ? sigmoidView : activationView) === mode) return;
  const scroll = $('channels').parentElement;
  const position = scroll.scrollTop;
  if (feature.has_sigmoid) sigmoidView = mode;
  else activationView = mode;
  renderActivationControl(feature);
  await selectLayer(feature.id, false);
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
$('forward-previous').onclick = () => stepForward(-1);
$('forward-next').onclick = () => stepForward(1);
$('forward-toggle').onclick = () => {
  if (forwardPlaying) finishForwardPass(true);
  else playForwardPass().catch(error => { $('status').textContent = error.message; });
};
$('pass-play').onclick = () => {
  if (mobilePlaying) {
    pauseMobilePass();
    showMobileStep(mobileIndex).catch(error => { $('status').textContent = error.message; });
  } else playMobilePass().catch(error => { pauseMobilePass(); $('status').textContent = error.message; });
};
$('pass-previous').onclick = () => stepMobilePass(-1).catch(error => { $('status').textContent = error.message; });
$('pass-next').onclick = () => stepMobilePass(1).catch(error => { $('status').textContent = error.message; });
$('pass-finish').onclick = () => showMobileResults();
reducedMotion.addEventListener('change', () => {
  if (mobilePlaying && reducedMotion.matches) {
    pauseMobilePass();
    showMobileStep(mobileIndex).catch(error => { $('status').textContent = error.message; });
  }
  updateMobileControls();
});
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
    if (forwardPlaying || forwardStep !== null) finishForwardPass();
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
    mobilePlan = mobileSteps(features);
    lastLayers.clear();
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
    if (mobileMode.matches) resetMobilePass();
    updatingSample = false;
    setControlsDisabled(false);
    await selectLayer(activeLayer);
    if (version !== sampleVersion) return;
    renderMasks();
    $('explorer').classList.remove('loading');
    $('explorer').setAttribute('aria-busy', 'false');
    if (mobileMode.matches) {
      $('explorer').classList.remove('forward-preparing');
      updateMobileControls();
    }
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
function setDiagramDepth(svg, extra) {
  if (!svg.layoutGeometry) {
    const stacks = [...svg.querySelectorAll('.stack')].map(node => {
      const matrix = node.transform.baseVal.getItem(0).matrix;
      return {node, x: matrix.e, y: matrix.f, center: matrix.f + .29 * Number(node.dataset.displaySize)};
    });
    const centers = new Map(stacks.map(item => [item.node.id, item.center]));
    const operations = [...svg.querySelectorAll('.operation')].map(node => {
      const edge = node.querySelector('.edge');
      const points = edge.getAttribute('d').match(/-?\d+(?:\.\d+)?/g).map(Number);
      const caption = node.querySelector('.edge-caption');
      return {node, edge, points, caption, y: Number(caption.getAttribute('y'))};
    });
    const inputY = Number(svg.dataset.inputY);
    const lastY = Math.max(...centers.values());
    const bottom = svg.viewBox.baseVal.y + svg.viewBox.baseVal.height;
    svg.layoutGeometry = {stacks, centers, operations, inputY, lastY, bottom};
  }
  if (svg.layoutGeometry.extra === extra) return;
  svg.layoutGeometry.extra = extra;
  const {stacks, centers, operations, inputY, lastY} = svg.layoutGeometry;
  const shift = id => ((centers.get(id) ?? inputY) - inputY) / (lastY - inputY) * extra;
  stacks.forEach(({node, x, y}) => node.setAttribute('transform', `translate(${x} ${y + shift(node.id)})`));
  operations.forEach(({node, edge, points, caption, y}) => {
    const from = shift(edge.dataset.from), to = shift(edge.dataset.to);
    const [x1, y1, x2, y2] = points;
    node.querySelectorAll('path').forEach(path => path.setAttribute('d', `M${x1} ${y1 + from} L${x2} ${y2 + to}`));
    caption.setAttribute('y', String(y + (from + to) / 2));
  });
}
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
  const rowLimit = wideLayout ? availableWidth - 2 * Math.max(0, 196 - imageSize - 12) : availableWidth;
  const bounds = svg.viewBox.baseVal;
  setDiagramDepth(svg, svg.layoutGeometry?.extra ?? 0);
  const baseBottom = svg.layoutGeometry.bottom;
  const minimumBottom = baseBottom + 160;
  const inputY = Number(svg.dataset.inputY);
  const scale = Math.min((rowLimit - 2 * sideWidth) / bounds.width,
    Math.max(1, availableHeight - imageSize / 2) / (minimumBottom - inputY));
  const bottom = Math.max(minimumBottom, inputY + (availableHeight - imageSize / 2) / scale);
  setDiagramDepth(svg, bottom - baseBottom);
  row.style.setProperty('--hero-image-size', `${imageSize}px`);
  row.style.setProperty('--hero-side-width', `${sideWidth}px`);
  const rowWidth = scale * bounds.width + 2 * sideWidth;
  row.style.width = `${rowWidth}px`;
  const dividerX = wideLayout
    ? (availableWidth - rowWidth) / 2 + imageSize + 12 : 196;
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
const inputTools = document.querySelector('.input-tools');
const desktopInputParent = inputTools.parentElement;
const architectureRow = document.querySelector('.architecture-row');
const diagramToolbar = document.querySelector('.diagram-toolbar');
const diagramFooter = document.querySelector('.diagram-footer');
blocks.forEach(block => {
  const button = document.createElement('button');
  button.textContent = block;
  button.dataset.block = block;
  button.disabled = !features.length;
  button.setAttribute('aria-pressed', String(block === 'Encoder'));
  button.onclick = () => selectBlock(block);
  $('mobile-blocks').appendChild(button);
});
async function applyResponsiveLayout() {
  if (forwardPlaying || forwardStep !== null) finishForwardPass();
  document.querySelector('.hero').setAttribute('aria-label', mobileMode.matches ? 'Forward-pass walkthrough' : 'U-Net architecture');
  if (mobileMode.matches) {
    if (features.length) $('explorer').dataset.mobilePhase = 'result';
    $('mobile-result').append(architectureRow, diagramFooter);
    $('mobile-sample-slot').appendChild(inputTools);
    $('mobile-layer-slot').appendChild(layerControls);
    $('mobile-activation-slot').appendChild(activationControl);
    $('mobile-model-details').appendChild(summary);
  } else {
    diagramToolbar.before(architectureRow);
    diagramToolbar.after(diagramFooter);
    desktopInputParent.appendChild(inputTools);
    desktopLayerParent.prepend(layerControls);
    desktopActivationParent.prepend(activationControl);
    document.querySelector('.header').appendChild(summary);
  }
  if (features.length) {
    const id = mobileMode.matches && activeLayer.endsWith('_skip')
      ? activeLayer.replace('dec_', 'enc_').replace('_skip', '_conv2') : activeLayer;
    await selectLayer(id, false);
    if (mobileMode.matches) showMobileResults(true);
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
