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

- `src/ui/architecture.svg` shows all 19 tensor stages and every channel in fixed order.
  Map planes share a horizontal baseline and use an oblique spatial projection. Regular stacks show channel 1 in front; concat stacks show
  encoder-skip channels on the left and upsampled channels on the right.
  This is display order only: the actual tensor remains [upsampled, encoder skip].
  Hover spreads maps, then previews a channel face-on around its center. Moving left/right
  through the preview browses channels without changing their order. Clicking selects that channel.
  Concat first separates into its source groups, pauses, then allows face-on browsing.
  On exit, the preview closes before the groups fold together. Clicking the page background clears selection.
- Lower-resolution planes are enlarged for readability (128px: 1.05×; 64px: 1.5×; 32px: 2.5×).
  Their displayed sizes still decrease with depth; labels always give true tensor dimensions. Spatial dimensions sit at the right and bottom edges of the front-right plane;
  channel count sits below the stack-depth axis. Full previews remain square.
- Colored orthogonal arrows identify operations, with technical details on hover.
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

New inputs replay the backend stage order using the computed tensors. Each operation
arrow is traced before its output appears. Encoder skips are copies, and both inputs
arrive before concatenation is revealed. Sigmoid and thresholding precede prediction.
Timing is illustrative, not measured inference latency. Replay/skip is available;
selecting a layer or channel ends the illustration. Reduced-motion preferences skip it.

The three resolution levels use mirrored convolution pairs with a centered bottleneck.
Pooling arrows point straight down and upsampling arrows straight up; saved skip paths
remain horizontal and their captions share one aligned column.
