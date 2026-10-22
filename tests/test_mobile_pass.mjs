// Run with the project's Python environment: PYTHON=.../python node tests/test_mobile_pass.mjs
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {mobileSteps, tensorGeometry} from '../src/ui/mobile-pass.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const features = JSON.parse(execFileSync(process.env.PYTHON || 'python', ['-c', `
import json
from inference import ROOT, explorer_data
data = explorer_data((ROOT / 'files/examples/Tv33_003.png').read_bytes())
keys = ['id', 'label', 'block', 'channels', 'height', 'width', 'has_relu', 'has_sigmoid']
print(json.dumps([{key: feature[key] for key in keys} for feature in data['features']]))
`], {cwd:root, env:{...process.env, PYTHONPATH:'src'}, encoding:'utf8'}));

const plan = mobileSteps(features);
assert.equal(plan.length, 29);
assert.deepEqual(plan.filter(step => !step.id.endsWith('_relu')).slice(0,17).map(step => step.id), features.filter(feature => !feature.id.endsWith('_skip')).map(feature => feature.id));
assert.deepEqual(plan.slice(-3).map(step => step.id), ['final_conv', 'sigmoid', 'threshold']);
assert.equal(plan[0].inputs[0].id, 'input');
assert.equal(plan[0].feature.channels, 16);
assert.equal(plan[0].view, 'pre');
assert.equal(plan[4].operation, 'Max pooling');
assert.equal(plan[4].feature.width, 64);
assert.equal(plan[12].feature.width, 32);
assert.equal(plan[12].feature.channels, 64);
assert.deepEqual(plan.filter(step => step.saved).map(step => step.id), ['enc_l1_conv2_relu', 'enc_l2_conv2_relu']);
for (const level of [2,1]) {
  const concat = plan.find(step => step.id === `dec_l${level}_concat`);
  assert.deepEqual(concat.inputs.map(source => source.id), [`dec_l${level}_deconv`, `enc_l${level}_conv2`]);
  assert.equal(concat.inputs[1].view, 'post');
  const counts = concat.inputs.map(source => features.find(feature => feature.id === source.id).channels);
  assert.equal(counts[0] + counts[1], concat.feature.channels);
  assert.equal(concat.kind, 'concat');
}
assert.equal(plan.at(-3).view, 'pre');
assert.equal(plan.at(-2).inputs[0].view, 'pre');
assert.equal(plan.at(-2).view, 'post');
assert.equal(plan.at(-1).inputs[0].view, 'post');
assert.equal(plan.at(-1).binary, true);
assert(plan.every(step => step.inputs.every(source => source.id !== 'reference')));
console.log('Mobile forward pass: real stage order, shapes, skip sources, activation views and output steps pass.');

for (const feature of features.filter(feature => feature.has_relu)) {
  const index = plan.findIndex(step => step.id === feature.id);
  const conv = plan[index], relu = plan[index + 1];
  assert.equal(conv.view, 'pre');
  assert.equal(relu.kind, 'relu');
  assert.equal(relu.view, 'post');
  assert.equal(relu.inputs[0].id, conv.feature.id);
  assert.equal(relu.inputs[0].view, 'pre');
}
for (let index = 1; index < plan.length; index++) {
  const previous = plan[index - 1], input = plan[index].inputs[0];
  assert.equal(input.id, previous.feature.id);
  assert.equal(input.view, previous.view);
}

const byId = id => features.find(feature => feature.id === id);
for (const level of [1, 2]) {
  const input = tensorGeometry(byId(`enc_l${level}_conv2`));
  const pooled = tensorGeometry(byId(`pool${level}`));
  assert.equal(pooled.width, input.width / 2);
  assert.equal(pooled.height, input.height / 2);
  assert.equal(pooled.depth, input.depth);
  const up = tensorGeometry(byId(`dec_l${level}_deconv`));
  const coarse = tensorGeometry(byId(level === 2 ? 'bottleneck_conv2' : 'dec_l2_conv2'));
  assert.equal(up.width, coarse.width * 2);
  assert.equal(up.height, coarse.height * 2);
  const concat = tensorGeometry(byId(`dec_l${level}_concat`));
  assert.equal(concat.width, up.width);
  assert.equal(concat.depth + 2.5, (up.depth + 2.5) * 2);
}
