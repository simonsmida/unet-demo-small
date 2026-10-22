---
title: U-Net Feature Explorer
emoji: 🔬
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Two-stage U-Net Feature Explorer

[Live demo](https://huggingface.co/spaces/k4ilash/unet-demo) · [Source code](https://github.com/simonsmida/unet-demo-small)

A fixed two-stage U-Net for tubule segmentation in cropped cilium microscopy images.

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run src/app.py
```

The Dockerfile serves the same app on port 7860 for Hugging Face Spaces.

| Model | Setting |
| --- | --- |
| Channels | 16 → 32 → 64 → 32 → 16 |
| Blocks | 2 encoders, bottleneck, 2 decoders; 2 × (3×3 convolution + ReLU) per block |
| Sampling | 2×2 max pooling; 2×2 transpose convolutions; concatenate post-ReLU encoder skips |
| Output | 1×1 convolution; sigmoid for foreground probability; binary threshold 0.5 |
| Parameters | 116,753 |
| Data | 382 train crops / 7 TIFFs; 142 validation crops / 2 different TIFFs |
| Validation | Dice 0.873; precision 0.879; recall 0.867 |
| Checkpoint | `../unet-solution/experiments/architecture/runs/d2_w16_c2/best.pt`, update 4600 |

The checkpoint's layer keys were renamed to match the fixed readable model in `src/model.py`.
Strict loading and bit-identical logits were verified against the trained model; all 142
validation crops reproduce Dice 0.8730380315. No retraining or prediction changes.

## Diagram and feature maps

The Simplified diagram switch uses blue feature planes and gray copied skip planes,
including only the skip half of each concatenation. It hides sample selection, the reference and overlay controls.
Shapes and operations stay unchanged; stacks still select their layer in the explorer.
Mask overlay combines the off switch and opacity control: Off shows binary masks;
positive values overlay the masks on the input at the displayed opacity.

- `src/ui/architecture.svg` shows all 19 tensor stages and every channel in fixed order.
  Map planes share a horizontal baseline and use an oblique spatial projection. Regular stacks show channel 1 in front; concat stacks show
  encoder-skip channels on the left and upsampled channels on the right.
  This is display order only: the actual tensor remains [upsampled, encoder skip].
  Hover spreads maps, then previews a channel face-on around its center. Moving left/right
  through the preview browses channels without changing their order. Clicking selects that channel.
  Concat first separates into its source groups, pauses, then allows face-on browsing.
  On exit, the preview closes before the groups fold together. Clicking the page background clears selection.
- Spatial planes use display sizes of 136, 104 and 88 diagram units for 128, 64 and 32 pixel maps.
  Their displayed sizes still decrease with depth; labels always give true tensor dimensions. Spatial dimensions sit at the right and bottom edges of the front-right plane;
  channel count sits below the stack-depth axis. Full previews remain square.
- Desktop panels fit the available viewport height. The diagram scales proportionally;
  short windows can scroll, while the feature grid scrolls within its panel.
- Colored orthogonal arrows identify operations, with technical details on hover.
  Convolution arrows have equal lengths within each resolution level. Encoder/decoder
  convolution pairs share a symmetry axis; the bottleneck pair is centered in the drawing.
  Skip paths are dashed; concat paths are solid.
- Click a face to select its exact channel. SVG, grid and preview share one renderer.
  Concat channels reuse their source color scale and colorbar, so copied maps look identical.
- The ReLU selector controls convolution displays. Pooling, upsampling, encoder skips and
  concatenations always show their actual outputs. Skips remain post-ReLU even when a
  convolution is displayed before ReLU. Changing the display never changes inference.
  Concatenations contain upsampled channels first.
- Signed values use a zero-centered blue/white/red scale. Nonnegative values use grayscale.
  Per-channel scales and endpoints match the existing explorer. Sigmoid uses a fixed 0–1 scale.
- All visible stacks need their selected activation view decoded; the unused alternative
  remains compressed. Channel images are rendered only when shown and then cached.
- Sample changes prepare the new diagram and selected view before replacing the current sample.

Automatic masks are references, not expert annotations. Validation selected the checkpoint
and architecture; these are not independent test results. Dice, precision and recall pool
foreground pixel counts over all 142 validation crops (Dice = 2TP / (2TP + FP + FN)).
The visibly blurred `Tv33_000` crop is omitted from the example browser only; it remains
in the reported validation score. The 18 thumbnail choices are a browsing subset, not a new evaluation set.

[Training run](https://wandb.ai/simon-smida/cilia-unet/runs/l56dyoik)

## Forward-pass illustration

On mobile, **Forward pass** opens a focused operation-by-operation walkthrough.
The initial card shows the input and optional automatic reference only. Play or
step through 29 stages, pause to inspect channels, or choose **Show prediction**.
The final comparison shows input, binary prediction and the available reference.
Manual stepping keeps the threshold stage open until **Compare results** is chosen.

The mobile sequence omits display-only skip copies, follows each convolution with
its own ReLU step, and separates logits, sigmoid and threshold 0.5. Concatenation shows both
real source groups in tensor order (upsampled, then encoder), including channels
from both halves. Every channel is rendered in compact overlapping stacks.
Every tensor is one continuous stack in channel order. Outputs are progressively
revealed, then move into the next operation’s input position. This reveal order
and timing illustrate data flow, not the hardware’s computation schedule.
Copies reuse their source pixels and color scales.
The full grid and selected preview follow the current tensor. Before/after
activation inspection pauses playback; resuming uses actual post-ReLU inputs.
Reference masks never enter the model. Playback uses the already computed tensors,
with no network requests between steps. Animation timing is explanatory.

Mobile controls and card height stay fixed through playback. Reduced-motion users
step through snapshots. Changing the input cancels playback and starts a new card.
The desktop SVG and its playback remain available above the mobile breakpoint.
The model-upload prototype is kept on its separate branch.

Check the mobile stage definitions against the real inference metadata:
`PYTHON=path/to/python node tests/test_mobile_pass.mjs`.

New inputs replay the backend stage order using the computed tensors. Each operation
arrow is traced before its output appears. Encoder skips are copies, and both inputs
arrive before concatenation is revealed. Sigmoid and thresholding precede prediction.
The grid, selected preview and layer controls follow each revealed tensor. The final
view shows sigmoid probabilities; skipping also jumps to that output.
The controls support Forward pass/Skip and Previous/Next stepping. There are 20 steps:
19 tensor outputs, then sigmoid and the binary prediction. Forward pass continues from a
manual step. Previous/Next keep the diagram, grid, preview, and activation view in sync.
Timing is illustrative, not measured inference latency. Replay/skip is available;
selecting a layer or channel ends the illustration. Reduced-motion preferences skip it.

The three resolution levels use mirrored convolution pairs with a centered bottleneck.
Pooling arrows point straight down and upsampling arrows straight up; saved skip paths
remain horizontal and their captions share one aligned column. Vertical spacing
adapts to the available height without stretching feature-map planes. The toolbar
stays 6 px below the diagram.

## Mobile

Phones show the image comparison and feature explorer without the architecture diagram.
Sticky block tabs and layer arrows navigate the same tensors as desktop. Tap a channel
in the four-column uniform grid to inspect it; activation and color scales are unchanged.
Example thumbnails and upload are below the comparison images. Uploads hide unavailable
ground truth. Model details are collapsible; automatic forward-pass playback is disabled.
The desktop diagram and its forward-pass animation remain available on wider screens.

## Code and checks

- `src/app.py`: Streamlit session and input handling.
- `src/inference.py`: preprocessing, inference, and display encoding.
- `src/ui/`: plain HTML, CSS, JavaScript, and the architecture SVG. No build step.

```bash
PYTHONPATH=src python -m unittest discover -s tests
node --check src/ui/explorer.js
```

Only the final inference payload is cached (8 inputs). Skip and concat views reuse
source pixels. Layer, channel, and display controls run entirely in the browser.
The forward-pass playback is illustrative and can be skipped immediately.

Uploads are restricted server-side to PNG/JPEG, 5 MB, and 16 megapixels. Invalid
images keep the previous input. The container limits WebSocket messages to 16 MB
and enables compression. Checkpoints are bundled and loaded with `weights_only=True`.
Pillow 12.3 is pinned within Streamlit’s `<13` compatibility range; remaining
published Pillow advisories concern PDF/GIF/PPM/TIFF/EPS paths, excluded by the
upload format allowlist. Hosting cold starts and network latency still apply.
