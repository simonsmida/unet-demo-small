"""CPU inference, display encoding, and bounded PNG/JPEG uploads."""
import base64
import binascii
import gzip
import io
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from model import UNet

ROOT = Path(__file__).resolve().parents[1]

LAYERS = {
    "enc_l1_conv1": "Encoder 1 · Convolution 1",
    "enc_l1_conv2": "Encoder 1 · Convolution 2",
    "pool1": "Encoder 1 · Max pooling",
    "enc_l2_conv1": "Encoder 2 · Convolution 1",
    "enc_l2_conv2": "Encoder 2 · Convolution 2",
    "pool2": "Encoder 2 · Max pooling",
    "bottleneck_conv1": "Bottleneck · Convolution 1",
    "bottleneck_conv2": "Bottleneck · Convolution 2",
    "dec_l2_deconv": "Decoder 2 · Upsampling",
    "dec_l2_skip": "Decoder 2 · Encoder skip",
    "dec_l2_concat": "Decoder 2 · Concatenation",
    "dec_l2_conv1": "Decoder 2 · Convolution 1",
    "dec_l2_conv2": "Decoder 2 · Convolution 2",
    "dec_l1_deconv": "Decoder 1 · Upsampling",
    "dec_l1_skip": "Decoder 1 · Encoder skip",
    "dec_l1_concat": "Decoder 1 · Concatenation",
    "dec_l1_conv1": "Decoder 1 · Convolution 1",
    "dec_l1_conv2": "Decoder 1 · Convolution 2",
    "final_conv": "Output · 1 × 1 convolution",
}

STAGES = tuple(LAYERS)


def load_model():
    model = UNet()
    weights = torch.load(ROOT / "files/model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    return model.eval()


def preprocess(image):
    image = image.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None, None]


def predict(image_bytes):
    # Each inference owns its model and hooks: no cross-session mutable state.
    model = load_model()
    with Image.open(io.BytesIO(image_bytes)) as image:
        x = preprocess(image)
    acts = {}

    def capture(name):
        def hook(_module, _inputs, output):
            acts[name] = output.detach()
        return hook

    handles = [module.register_forward_hook(capture(name))
               for name, module in model.named_modules() if name in STAGES]
    for level in [2, 1]:
        def capture_concat(_module, inputs, level=level):
            acts[f"dec_l{level}_concat"] = inputs[0].detach()
        handles.append(getattr(model, f"dec_l{level}_conv1").register_forward_pre_hook(capture_concat))
    try:
        with torch.inference_mode():
            probability = model(x).sigmoid()[0, 0].numpy()
    finally:
        for handle in handles:
            handle.remove()
    for level in [2, 1]:
        acts[f"dec_l{level}_skip"] = acts[f"enc_l{level}_conv2"].relu()
    return x[0, 0].numpy(), probability, {name: value[0].numpy() for name, value in acts.items()}


def layer_block(layer):
    if layer.startswith("enc") or layer.startswith("pool"):
        return "Encoder"
    if layer.startswith("bottleneck"):
        return "Bottleneck"
    return "Output" if layer == "final_conv" else "Decoder"


def layer_note(layer):
    if layer == "final_conv":
        return "A 1 × 1 convolution produces one foreground logit per pixel. Sigmoid converts logits to probabilities."
    if layer.endswith("_skip"):
        return "Skip connections carry the encoder’s post-ReLU features unchanged."
    if layer.endswith("_concat"):
        return "Concatenation joins channels: upsampling first, encoder skip second. Values are not added."
    if "deconv" in layer:
        return "Upsampling restores resolution. Concatenation joins these maps with encoder detail."
    if layer.startswith("pool"):
        return "Pooling reduces detail and expands the area represented by each response."
    if layer.startswith("dec"):
        return "Decoder convolution combines coarse features with encoder detail. The view selects before or after ReLU."
    if layer.startswith("bottleneck"):
        return "Bottleneck convolution learns patterns at the coarsest resolution. The view selects before or after ReLU."
    return "Convolution learns local patterns. The view selects before or after ReLU."


def encode_pixels(array, compress=False):
    pixels = np.ascontiguousarray(array, dtype=np.uint8).tobytes()
    if compress:
        pixels = gzip.compress(pixels, compresslevel=1, mtime=0)
    return base64.b64encode(pixels).decode("ascii")


def feature_view(values, signed):
    """Scale for display without moving zero; keep real range values for the legend."""
    values = values if signed else np.maximum(values, 0)
    peaks = np.max(np.abs(values), axis=(1, 2))
    scaled = np.divide(values, peaks[:, None, None], out=np.zeros_like(values),
                       where=peaks[:, None, None] > 0)
    pixels = np.rint(scaled * 127 + 128) if signed else np.rint(scaled * 255)
    return {"pixels": encode_pixels(pixels, compress=True), "peaks": peaks.tolist(), "signed": signed}


def explorer_data(image_bytes):
    image, probability, acts = predict(image_bytes)
    features = []
    for stage in STAGES:
        values = acts[stage]
        channels, height, width = values.shape
        has_relu = "conv" in stage and "deconv" not in stage and stage != "final_conv"
        nonnegative = stage.startswith("pool") or stage.endswith("_skip")
        # Copies and concatenations reuse their source views in the browser.
        if stage.endswith(("_skip", "_concat")):
            views = {}
        elif has_relu:
            views = {"pre": feature_view(values, True), "post": feature_view(values, False)}
        else:
            views = {"output": feature_view(values, not nonnegative)}
        if stage == "final_conv":
            views = {
                "pre": feature_view(values, True),
                "post": {"pixels": encode_pixels(np.rint(probability[None] * 255), compress=True),
                         "peaks": [1.0], "signed": False},
            }
        features.append({
            "id": stage, "label": LAYERS[stage], "block": layer_block(stage),
            "svg_id": stage, "note": layer_note(stage),
            "channels": channels, "height": height, "width": width,
            "views": views, "has_relu": has_relu, "has_sigmoid": stage == "final_conv",
            "nonnegative": nonnegative,
        })
    return {
        "encoding": "gzip",
        "input": encode_pixels(image * 255),
        "prediction": encode_pixels(probability >= 0.5),
        "features": features,
    }


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_UPLOAD_PIXELS = 16_000_000


def validate_upload(encoded):
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_UPLOAD_BYTES + 2) // 3):
        raise ValueError("Please upload a PNG or JPEG up to 5 MB.")
    try:
        candidate = base64.b64decode(encoded.encode("ascii"), validate=True)
        if len(candidate) > MAX_UPLOAD_BYTES:
            raise ValueError("Please upload a PNG or JPEG up to 5 MB.")
        with Image.open(io.BytesIO(candidate), formats=("PNG", "JPEG")) as image:
            if image.width * image.height > MAX_UPLOAD_PIXELS:
                raise ValueError("Please upload an image up to 16 megapixels.")
            image.verify()
        # Decode once now so truncated images cannot replace the current input.
        with Image.open(io.BytesIO(candidate), formats=("PNG", "JPEG")) as image:
            image.load()
    except (binascii.Error, UnicodeError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Please upload a valid PNG or JPEG image.") from error
    return candidate
