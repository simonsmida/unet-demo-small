"""Run with: PYTHONPATH=src python -m unittest discover -s tests."""
import base64
import gzip
import io
import unittest
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch

from inference import ROOT, STAGES, explorer_data, feature_view, load_model, predict, preprocess, validate_upload


def image_bytes(value=128, format='PNG'):
    buffer = io.BytesIO()
    Image.new('L', (128, 128), value).save(buffer, format=format)
    return buffer.getvalue()


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.image = (ROOT / 'files/examples/Tv33_003.png').read_bytes()
        cls.input, cls.probability, cls.acts = predict(cls.image)

    def test_outputs_match_model_and_true_channel_order(self):
        self.assertEqual(set(self.acts), set(STAGES))
        with torch.inference_mode():
            expected = load_model()(preprocess(Image.open(io.BytesIO(self.image)))).sigmoid()[0, 0].numpy()
        np.testing.assert_array_equal(self.probability, expected)
        for level in (1, 2):
            np.testing.assert_array_equal(self.acts[f'dec_l{level}_skip'], np.maximum(self.acts[f'enc_l{level}_conv2'], 0))
            joined = np.concatenate([self.acts[f'dec_l{level}_deconv'], self.acts[f'dec_l{level}_skip']])
            np.testing.assert_array_equal(self.acts[f'dec_l{level}_concat'], joined)

    def test_display_aliases_and_probability_scale(self):
        data = explorer_data(self.image)
        for feature in data['features']:
            if feature['id'].endswith(('_skip', '_concat')):
                self.assertEqual(feature['views'], {})
        output = data['features'][-1]['views']['post']
        self.assertEqual(output['peaks'], [1.0])
        pixels = np.frombuffer(gzip.decompress(base64.b64decode(output['pixels'])), dtype=np.uint8)
        np.testing.assert_array_equal(pixels, np.rint(self.probability.ravel() * 255))
        zero = feature_view(np.zeros((1, 2, 2), np.float32), True)
        self.assertEqual(zero['peaks'], [0.0])
        self.assertEqual(gzip.decompress(base64.b64decode(zero['pixels'])), bytes([128]) * 4)

    def test_concurrent_inputs_do_not_share_hooks(self):
        images = [image_bytes(0), image_bytes(255)]
        expected = [predict(image)[1] for image in images]
        with ThreadPoolExecutor(max_workers=2) as pool:
            actual = list(pool.map(predict, images))
        for reference, (_, probability, _) in zip(expected, actual):
            np.testing.assert_array_equal(reference, probability)

    def test_diagram_has_explicit_input_and_output_connections(self):
        svg = ET.parse(ROOT / 'src/ui/architecture.svg')
        edges = svg.findall('.//{http://www.w3.org/2000/svg}path[@class="edge"]')
        incoming = [edge for edge in edges if not edge.get('data-from')]
        outgoing = [edge for edge in edges if not edge.get('data-to')]
        self.assertEqual(len(incoming), 1)
        self.assertEqual(incoming[0].get('data-to'), STAGES[0])
        self.assertEqual(len(outgoing), 1)
        self.assertEqual(outgoing[0].get('data-from'), STAGES[-1])

    def test_upload_validation(self):
        for format in ('PNG', 'JPEG'):
            raw = image_bytes(format=format)
            self.assertEqual(validate_upload(base64.b64encode(raw).decode()), raw)
        for candidate in (None, 'not base64!', base64.b64encode(b'not an image').decode(),
                          base64.b64encode(image_bytes(format='GIF')).decode(), 'A' * 7_000_000):
            with self.assertRaises(ValueError):
                validate_upload(candidate)
        buffer = io.BytesIO()
        Image.new('L', (4001, 4000)).save(buffer, format='PNG')
        with self.assertRaisesRegex(ValueError, '16 megapixels'):
            validate_upload(base64.b64encode(buffer.getvalue()).decode())
        with self.assertRaises(ValueError):
            validate_upload(base64.b64encode(image_bytes()[:40]).decode())


if __name__ == '__main__':
    unittest.main()
