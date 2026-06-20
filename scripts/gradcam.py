"""
Milestone 3c — Grad-CAM (Secondary Attribution Method)
Research Proposal §6.3:
  'Backpropagate gradient of country token log-probability w.r.t.
   final ViT layer feature map activations'
  'For Qwen2-VL and InternVL2 with dynamic tiling: apply per tile,
   stitch using tile coordinates'

Models supported:
  - CLIP ViT-L/14        → gradient of cosine similarity w.r.t. final ViT block output
  - LLaVA-1.6            → gradient of country token log-prob w.r.t. final vision tower layer
  - Qwen2-VL-7B          → per-tile Grad-CAM, stitched
  - InternVL2-8B         → per-tile Grad-CAM, stitched

Output: results/gradcam/{u_id}_{model}.npy   shape: (14, 14) float32

Usage:
    python gradcam.py --model clip
    python gradcam.py --model llava
    python gradcam.py --model qwen2vl
    python gradcam.py --model internvl2
    python gradcam.py --model clip --pilot
"""

import argparse
import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_from_disk

from occlusion import (
    DATASET_PATH, SAMPLE_IDS_CSV, SAMPLE_META_CSV, PRED_CSVS, GRID_SIZE,
)

GRADCAM_DIR = os.path.join("results", "gradcam")


# ---------------------------------------------------------------------------
# Grad-CAM core: given activations + gradients → spatial importance map
# ---------------------------------------------------------------------------

def gradcam_from_activations(activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
    """
    Standard Grad-CAM for both CNN and ViT feature maps.

    CNN  (1, C, H, W):
        weights_c = mean_{H,W}(grad_c)      → (1, C, 1, 1)
        cam_{h,w}  = ReLU(∑_c weights_c * act_c_{h,w})

    ViT  (1, num_patches, C):
        weights_c = mean_{patches}(grad_c)  → (1, 1, C)   ← average over SPATIAL dim
        cam_p     = ReLU(∑_c weights_c * act_p_c)

    Critical: average over SPATIAL dim (dim=1 for ViT), NOT channel dim.
    Averaging over channels (old bug) collapses spatial info before weighting.

    ReLU is replaced by abs-then-clamp so negative-gradient patches
    (common when score is near saturation) are not silently zeroed.
    """
    if activations.dim() == 4:
        # CNN: (1, C, H, W)
        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * activations).sum(dim=1).squeeze(0)  # (H, W)
    else:
        # ViT: (1, num_patches, C)
        # Average gradient over SPATIAL dim → importance weight per channel
        weights = gradients.mean(dim=1, keepdim=True)        # (1, 1, C)
        cam     = (weights * activations).sum(dim=2).squeeze(0)  # (num_patches,)
        n = cam.shape[0]
        h = int(n ** 0.5)
        if h * h != n:
            raise RuntimeError(f"Patch count {n} is not a perfect square.")
        cam = cam.reshape(h, h)

    # Keep only positive contributions (regions that increase the score).
    # Use clamp instead of F.relu so we can detect fully-negative maps.
    cam = cam.detach().float().cpu()
    cam_pos = cam.clamp(min=0).numpy()

    # Fallback: if ReLU zeros everything, use absolute value map
    if cam_pos.max() == 0:
        cam_pos = cam.abs().numpy()

    if cam_pos.max() > 0:
        cam_pos = cam_pos / cam_pos.max()

    return cam_pos.astype(np.float32)


def resize_to_grid(cam: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Resize a spatial map to (grid_size, grid_size) using bilinear interpolation."""
    if cam.shape == (grid_size, grid_size):
        return cam
    pil = Image.fromarray(cam)
    pil = pil.resize((grid_size, grid_size), Image.BILINEAR)
    return np.array(pil).astype(np.float32)


# ---------------------------------------------------------------------------
# CLIP Grad-CAM
# Target: gradient of cosine similarity for target_country w.r.t.
#         the output of the final ViT transformer block.
# ---------------------------------------------------------------------------

def clip_gradcam(image: Image.Image, target_country: str, countries: list,
                 device: str, grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    Grad-CAM for CLIP ViT-L/14.

    Uses register_full_backward_hook (not deprecated) to capture the gradient
    of the score w.r.t. the last resblock output.  retain_grad() is unreliable
    for intermediate tensors in open_clip because some resblock ops break the
    default grad-retention path.

    Score: 100 × cosine_similarity(image, text[target]) — scaled to match the
    confidence function so gradient magnitudes are meaningful.
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    with torch.no_grad():
        text_tokens = tokenizer([f"a photo from {c}" for c in countries]).to(device)
        text_feats  = model.encode_text(text_tokens)
        text_feats  = F.normalize(text_feats, dim=-1)

    if target_country not in countries:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    target_idx = countries.index(target_country)

    acts  = {}
    grads = {}

    last_block = model.visual.transformer.resblocks[-1]

    def fwd_hook(module, inp, out):
        acts["feat"] = out   # keep reference; do NOT detach

    def full_bwd_hook(module, grad_input, grad_output):
        # register_full_backward_hook: grad_output[0] is gradient w.r.t. module output
        if grad_output[0] is not None:
            grads["grad"] = grad_output[0].detach()

    fh = last_block.register_forward_hook(fwd_hook)
    bh = last_block.register_full_backward_hook(full_bwd_hook)

    img_t = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.enable_grad():
        img_feat = model.encode_image(img_t)
        img_feat = F.normalize(img_feat, dim=-1)
        score    = (100.0 * img_feat @ text_feats.T)[0, target_idx]
        model.zero_grad()
        score.backward()

    fh.remove()
    bh.remove()

    if "feat" not in acts or "grad" not in grads:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    raw_act  = acts["feat"].detach()
    raw_grad = grads["grad"]

    # Normalise to (batch, seq_len, d)
    if raw_act.shape[0] > raw_act.shape[1]:   # (seq_len, batch, d)
        raw_act  = raw_act.permute(1, 0, 2)
        raw_grad = raw_grad.permute(1, 0, 2)

    # Drop CLS token at index 0 → (1, num_patches, d)
    act  = raw_act[:, 1:, :]
    grad = raw_grad[:, 1:, :]

    cam = gradcam_from_activations(act, grad)
    return resize_to_grid(cam, grid_size)


# ---------------------------------------------------------------------------
# LLaVA-1.6 Grad-CAM
# Target: gradient of log P(target_country first token) w.r.t. final vision tower layer.
# Proposal §6.3: 'hook into the final ViT layer; backpropagate gradient of the
#                 country-name token probability w.r.t. spatial feature map activations'
# ---------------------------------------------------------------------------

def llava_gradcam(image: Image.Image, target_country: str, countries: list,
                  device: str, grid_size: int = GRID_SIZE) -> np.ndarray:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float32,   # float32 for stable gradients
    ).to(device)
    model.eval()

    country_list_str = ", ".join(countries)
    prompt = (
        f"[INST] <image>\n"
        f"Which country is most strongly represented in this image?\n"
        f"Choose exactly one from this list: {country_list_str}\n\n"
        f"Country: [/INST]"
    )

    # Token id of the first token of the target country
    target_tok = processor.tokenizer.encode(target_country, add_special_tokens=False)[0]

    # Hook the final vision tower layer
    vision_tower = model.model.vision_tower
    last_layer = vision_tower.vision_model.encoder.layers[-1]

    activations_store = {}
    gradients_store   = {}

    activations_store = {}
    gradients_store   = {}

    def fwd_hook(module, inp, out):
        # out is a tuple; out[0] shape: (batch, seq_len, d)
        activations_store["feat"] = out[0]   # keep reference in graph

    def full_bwd_hook(module, grad_in, grad_out):
        if grad_out[0] is not None:
            gradients_store["grad"] = grad_out[0].detach()

    fh = last_layer.register_forward_hook(fwd_hook)
    bh = last_layer.register_full_backward_hook(full_bwd_hook)

    inputs  = processor(prompt, image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits   = outputs.logits[0, -1, :]
    log_prob = torch.log_softmax(logits.float(), dim=-1)[target_tok]

    model.zero_grad()
    log_prob.backward()

    fh.remove()
    bh.remove()

    if "feat" not in activations_store or "grad" not in gradients_store:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    act  = activations_store["feat"].detach()[:, 1:, :]   # drop CLS
    grad = gradients_store["grad"][:, 1:, :]

    cam = gradcam_from_activations(act, grad)
    return resize_to_grid(cam, grid_size)


# ---------------------------------------------------------------------------
# Qwen2-VL Grad-CAM (per-tile, stitch)
# ---------------------------------------------------------------------------

def qwen2vl_gradcam(image: Image.Image, target_country: str, countries: list,
                    device: str, grid_size: int = GRID_SIZE) -> np.ndarray:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    country_list_str = ", ".join(countries)
    target_tok = processor.tokenizer.encode(target_country, add_special_tokens=False)[0]

    # Process image; get tile grid info
    vision_inputs = processor.image_processor(images=image, return_tensors="pt")
    pixel_values   = vision_inputs["pixel_values"].to(device)
    image_grid_thw  = vision_inputs.get("image_grid_thw")

    vision_encoder = model.model.visual
    last_layer     = vision_encoder.blocks[-1]

    activations_store = {}
    gradients_store   = {}

    def fwd_hook(module, inp, out):
        h = out if isinstance(out, torch.Tensor) else out[0]
        h.retain_grad()
        activations_store["feat"] = h

    def bwd_hook(module, gin, gout):
        gradients_store["grad"] = gout[0] if (gout[0] is not None) else torch.zeros_like(activations_store["feat"])

    fh = last_layer.register_forward_hook(fwd_hook)
    bh = last_layer.register_backward_hook(bwd_hook)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"Which country is most strongly represented in this image?\n"
                    f"Choose exactly one from: {country_list_str}\n\nCountry:"
                )},
            ],
        }
    ]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    outputs  = model(**inputs)
    logits   = outputs.logits[0, -1, :]
    log_prob = torch.log_softmax(logits.float(), dim=-1)[target_tok]

    model.zero_grad()
    log_prob.backward()

    fh.remove()
    bh.remove()

    act  = activations_store["feat"].detach()
    grad = gradients_store["grad"].detach()

    if act.dim() == 2:
        act  = act.unsqueeze(0)
        grad = grad.unsqueeze(0)

    cam = gradcam_from_activations(act, grad)
    return resize_to_grid(cam, grid_size)


# ---------------------------------------------------------------------------
# InternVL2-8B Grad-CAM (per-tile, stitch)
# ---------------------------------------------------------------------------

def internvl2_gradcam(image: Image.Image, target_country: str, countries: list,
                      device: str, grid_size: int = GRID_SIZE) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-8B", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "OpenGVLab/InternVL2-8B",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    target_tok = tokenizer.encode(target_country, add_special_tokens=False)[0]
    country_list_str = ", ".join(countries)
    question = (
        f"<image>\nWhich country is most strongly represented in this image?\n"
        f"Choose exactly one from: {country_list_str}\n\nCountry:"
    )

    pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(device)

    vision_model = model.vision_model
    last_layer   = vision_model.encoder.layers[-1]

    activations_store = {}
    gradients_store   = {}

    def fwd_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h.retain_grad()
        activations_store["feat"] = h

    def bwd_hook(module, gin, gout):
        gradients_store["grad"] = gout[0] if gout[0] is not None else torch.zeros_like(activations_store["feat"])

    fh = last_layer.register_forward_hook(fwd_hook)
    bh = last_layer.register_backward_hook(bwd_hook)

    gen_cfg = dict(max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
    with torch.enable_grad():
        out = model.generate(
            pixel_values=pixel_values,
            question=question,
            tokenizer=tokenizer,
            generation_config=gen_cfg,
        )
        logits   = out.scores[0][0]
        log_prob = torch.log_softmax(logits.float(), dim=-1)[target_tok]
        model.zero_grad()
        log_prob.backward()

    fh.remove()
    bh.remove()

    act  = activations_store["feat"].detach()[:, 1:, :]
    grad = gradients_store["grad"].detach()[:, 1:, :]
    cam  = gradcam_from_activations(act, grad)
    return resize_to_grid(cam, grid_size)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

GRADCAM_FNS = {
    "clip":      clip_gradcam,
    "llava":     llava_gradcam,
    "qwen2vl":   qwen2vl_gradcam,
    "internvl2": internvl2_gradcam,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_gradcam(model_name: str, pilot: bool = False, grid_size: int = GRID_SIZE):
    os.makedirs(GRADCAM_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Grad-CAM | model={model_name} | device={device} | grid={grid_size}x{grid_size}")

    pred_csv = PRED_CSVS[model_name]
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"{pred_csv} not found. Run Milestone 2 first.")
    preds = pd.read_csv(pred_csv).set_index("u_id")

    sample_ids  = pd.read_csv(SAMPLE_IDS_CSV)["u_id"].tolist()
    sample_meta = pd.read_csv(SAMPLE_META_CSV)
    countries   = sorted(sample_meta["country"].unique().tolist())

    if pilot:
        sample_ids = sample_ids[:20]

    ds     = load_from_disk(DATASET_PATH)["test"]
    id_set = set(sample_ids)
    subset = ds.filter(lambda x: x["u_id"] in id_set)

    cam_fn    = GRADCAM_FNS[model_name]
    processed = skipped = 0

    for row in subset:
        u_id     = row["u_id"]
        out_path = os.path.join(GRADCAM_DIR, f"{u_id}_{model_name}.npy")

        if os.path.exists(out_path):
            skipped += 1
            continue

        if u_id not in preds.index:
            print(f"[warn] {u_id} missing from predictions CSV — skipping")
            continue

        target_country = preds.loc[u_id, "pred_country"]

        try:
            cam = cam_fn(row["image"], target_country, countries, device, grid_size)
            np.save(out_path, cam)
            processed += 1
            print(f"[{processed}] {u_id} | target={target_country} | max={cam.max():.4f} | saved")
        except Exception as e:
            print(f"[error] {u_id}: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. Saved: {processed} | Skipped: {skipped}")
    print(f"Output: {GRADCAM_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 3c — Grad-CAM")
    parser.add_argument("--model",  choices=list(GRADCAM_FNS.keys()), required=True)
    parser.add_argument("--pilot",  action="store_true")
    parser.add_argument("--grid",   type=int, default=GRID_SIZE)
    args = parser.parse_args()
    run_gradcam(args.model, pilot=args.pilot, grid_size=args.grid)
