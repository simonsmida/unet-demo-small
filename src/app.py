"""Run inference in Python; browse the feature explorer locally in the browser."""
from pathlib import Path
import base64
import hashlib
import io

import numpy as np
from PIL import Image
import streamlit as st
from streamlit.components.v1 import declare_component
import torch

from inference import encode_pixels, explorer_data, validate_upload

torch.set_num_threads(2)
st.set_page_config(page_title="U-Net Feature Explorer", layout="wide")
st.markdown("""<style>
.block-container { padding: 16px 20px; }
.block-container [data-testid="stVerticalBlock"] { gap: 0; }
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
.st-key-explorer { opacity: 1 !important; }
[data-testid="stMain"] { overflow: hidden; }
iframe { height: calc(100dvh - 32px) !important; }
@media (max-width: 750px) {
    .block-container { padding: 12px; }
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

# Cache the final payload only; raw tensors and model hooks stay request-local.
explorer_data = st.cache_data(max_entries=8, show_spinner=False)(explorer_data)


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
    if not isinstance(request, dict) or request.get("event") == st.session_state.get("input_event"):
        return
    st.session_state["input_error"] = None
    st.session_state["input_event"] = request.get("event")
    if request.get("action") == "example" and request.get("name") in examples:
        st.session_state["example"] = request["name"]
        st.session_state["upload_bytes"] = None
    elif request.get("action") == "upload":
        try:
            st.session_state["upload_bytes"] = validate_upload(request.get("bytes"))
        except ValueError as error:
            st.session_state["input_error"] = str(error)


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
