// Run with the project's Python environment: PYTHON=.../python node tests/test_mobile_pass.mjs
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {mobileSteps} from '../src/ui/mobile-pass.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const features = JSON.parse(execFileSync(process.env.PYTHON || 'python', ['-c', `
import json
from inference import ROOT, explorer_data
data = explorer_data((ROOT / 'files/examples/Tv33_003.png').read_bytes())
keys = ['id', 'label', 'block', 'channels', 'height', 'width', 'has_relu', 'has_sigmoid']
print(json.dumps([{key: feature[key] for key in keys} for feature in data['features']]))
`], {cwd:root, env:{...process.env, PYTHONPATH:'src'}, encoding:'utf8'}));

const plan = mobileSteps(features);
assert.equal(plan.length, 19);
assert.deepEqual(plan.slice(0,17).map(step => step.id), features.filter(feature => !feature.id.endsWith('_skip')).map(feature => feature.id));
assert.deepEqual(plan.slice(-3).map(step => step.id), ['final_conv', 'sigmoid', 'threshold']);
assert.equal(plan[0].inputs[0].id, 'input');
assert.equal(plan[0].feature.channels, 16);
assert.equal(plan[0].view, 'post');
assert.equal(plan[2].operation, 'Max pooling');
assert.equal(plan[2].feature.width, 64);
assert.equal(plan[7].feature.width, 32);
assert.equal(plan[7].feature.channels, 64);
assert.deepEqual(plan.filter(step => step.saved).map(step => step.id), ['enc_l1_conv2', 'enc_l2_conv2']);
for (const level of [2,1]) {
  const concat = plan.find(step => step.id === `dec_l${level}_concat`);
  assert.deepEqual(concat.inputs.map(source => source.id), [`dec_l${level}_deconv`, `enc_l${level}_conv2`]);
  assert.equal(concat.inputs[1].view, 'post');
  const counts = concat.inputs.map(source => features.find(feature => feature.id === source.id).channels);
  assert.equal(counts[0] + counts[1], concat.feature.channels);
  assert.equal(concat.kind, 'concat');
}
assert.equal(plan[16].view, 'pre');
assert.equal(plan[17].inputs[0].view, 'pre');
assert.equal(plan[17].view, 'post');
assert.equal(plan[18].inputs[0].view, 'post');
assert.equal(plan[18].binary, true);
assert(plan.every(step => step.inputs.every(source => source.id !== 'reference')));
console.log('Mobile forward pass: real stage order, shapes, skip sources, activation views and output steps pass.');
