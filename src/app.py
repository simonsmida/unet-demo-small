"""Run inference in Python; browse the feature explorer locally in the browser."""
from pathlib import Path
import base64
import gzip
import hashlib
import io
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import streamlit as st
from streamlit.components.v1 import declare_component
import torch

from model import UNet

torch.set_num_threads(2)
st.set_page_config(page_title="U-Net Feature Explorer", layout="wide")
st.markdown("""<style>
.block-container { padding: 1.25rem 2rem 1rem; }
.block-container [data-testid="stVerticalBlock"] { gap: 0; }
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
.st-key-explorer { opacity: 1 !important; }
@media (max-width: 750px) {
    .block-container { padding: 12px; }
}
@media (max-width: 674px) {
    [data-testid="stMain"] { overflow: hidden; }
    iframe { height: calc(100dvh - 24px) !important; }
}
html, body, iframe { color-scheme: light !important; }
[data-testid="stAppViewContainer"] { background: #f8fafc !important; }
@media (prefers-color-scheme: dark) {
    html, body, iframe { color-scheme: dark !important; }
    [data-testid="stAppViewContainer"] { background: #111827 !important; }
}
</style>""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parents[1]
explorer_ui = declare_component("explorer_ui", path=str(ROOT / "src/ui"))

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


def preprocess(img: Image.Image, size: Tuple[int, int] = (128, 128)) -> torch.Tensor:
    img = img.convert("L").resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None]


def register_stage_hooks(
    model: torch.nn.Module, stages: List[str]
) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    acts: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for name, mod in model.named_modules():
        if name in stages:
            def _make_hook(layer_name: str):
                def _hook(_m, _inp, out):
                    acts[layer_name] = out.detach().cpu()
                return _hook
            handles.append(mod.register_forward_hook(_make_hook(name)))
    return acts, handles


def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    for h in handles:
        h.remove()


@st.cache_data(max_entries=8, show_spinner=False)
def predict(image_bytes):
    # Each inference owns its model and hooks; cached results are safe across sessions.
    model = load_model()
    with Image.open(io.BytesIO(image_bytes)) as image:
        x = preprocess(image)
    acts, handles = register_stage_hooks(model, list(STAGES))
    # Capture the actual concatenation consumed by each decoder convolution.
    for level in [2, 1]:
        module = getattr(model, f"dec_l{level}_conv1")
        def capture_concat(_module, inputs, level=level):
            acts[f"dec_l{level}_concat"] = inputs[0].detach().cpu()
        handles.append(module.register_forward_pre_hook(capture_concat))
    try:
        with torch.inference_mode():
            probability = model(x).sigmoid()[0, 0].numpy()
    finally:
        remove_hooks(handles)
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


@st.cache_data(max_entries=8, show_spinner=False)
def explorer_data(image_bytes):
    image, probability, acts = predict(image_bytes)
    features = []
    for stage in STAGES:
        values = acts[stage]
        channels, height, width = values.shape
        has_relu = "conv" in stage and "deconv" not in stage and stage != "final_conv"
        nonnegative = stage.startswith("pool") or stage.endswith("_skip")
        views = {"pre": feature_view(values, True), "post": feature_view(values, False)} if has_relu else {
            "output": feature_view(values, not nonnegative)
        }
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


examples = sorted(path.stem for path in (ROOT / "files/examples").glob("*.png")
                  if not path.stem.endswith("_mask"))
@st.cache_data(show_spinner=False)
def example_thumbnails():
    names = [examples[i] for i in np.linspace(0, len(examples) - 1, 18, dtype=int)]
    if "Tv33_003" not in names:
        names[0] = "Tv33_003"
    thumbnails = []
    for name in names:
        with Image.open(ROOT / f"files/examples/{name}.png") as image:
            image = image.convert("L").resize((48, 48), Image.Resampling.BILINEAR)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        thumbnails.append({"name": name, "image": base64.b64encode(buffer.getvalue()).decode("ascii")})
    return thumbnails


st.session_state.setdefault("example", "Tv33_003")
st.session_state.setdefault("upload_bytes", None)


def handle_input_request():
    request = st.session_state["explorer"]
    if not request or request["event"] == st.session_state.get("input_event"):
        return
    st.session_state["input_error"] = None
    st.session_state["input_event"] = request["event"]
    if request["action"] == "example" and request.get("name") in examples:
        st.session_state["example"] = request["name"]
        st.session_state["upload_bytes"] = None
    elif request["action"] == "upload":
        candidate = base64.b64decode(request["bytes"])
        try:
            with Image.open(io.BytesIO(candidate)) as image:
                image.verify()
            st.session_state["upload_bytes"] = candidate
        except Exception:
            st.session_state["input_error"] = "Please upload a valid PNG or JPEG image."


if st.session_state["upload_bytes"] is not None:
    image_bytes = st.session_state["upload_bytes"]
    reference = None
else:
    example = st.session_state["example"]
    image_bytes = (ROOT / f"files/examples/{example}.png").read_bytes()
    with Image.open(ROOT / f"files/examples/{example}_mask.png") as mask:
        reference = np.asarray(mask.convert("L").resize((128, 128), Image.Resampling.NEAREST)) > 0

try:
    data = explorer_data(image_bytes)
except Exception as error:
    st.error(f"Could not process the image: {error}")
    st.stop()

explorer_ui(
    sample_id=hashlib.sha256(image_bytes).hexdigest(), data=data,
    reference=encode_pixels(reference) if reference is not None else None,
    examples=example_thumbnails(),
    active_example=st.session_state["example"] if st.session_state["upload_bytes"] is None else None,
    error=st.session_state.get("input_error"),
    default=None, key="explorer", on_change=handle_input_request,
)
