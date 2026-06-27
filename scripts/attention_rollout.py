"""
Milestone 3c — Attention Rollout (Secondary Attribution Method)
Research Proposal §6.3

Applies to ViT-based vision encoders:
  - CLIP ViT-L/14        (standalone, via HuggingFace CLIPVisionModel)
  - LLaVA-1.6 vision tower (openai/clip-vit-large-patch14-336)
  - Qwen2-VL (returns uniform map if flash attention is active)
  - InternVL2-8B (patches naive attention forward)

Output: results/attention_rollout/{u_id}_{model}.npy
        Shape: (14, 14) float32

Usage:
    python attention_rollout.py --model clip
    python attention_rollout.py --model llava
    python attention_rollout.py --model qwen2vl
    python attention_rollout.py --model internvl2
    python attention_rollout.py --model clip --pilot
"""

import argparse
import os
import gc
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import load_from_disk
import transformers
print(transformers.__version__)

# ── Paths (all absolute — script can be run from any directory) ────────────────
PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH    = os.path.join(PROJECT_ROOT, "data", "CulturalVQA")
SAMPLE_IDS_CSV  = os.path.join(PROJECT_ROOT, "results", "sample_ids.csv")
SAMPLE_META_CSV = os.path.join(PROJECT_ROOT, "results", "sample_metadata.csv")
ROLLOUT_DIR     = os.path.join(PROJECT_ROOT, "results", "attention_rollout")
GRID_SIZE       = 14


# ── Core rollout algorithm ────────────────────────────────────────────────────

def compute_rollout(attention_maps: list) -> np.ndarray:
    """
    attention_maps: list of (heads, seq, seq) or (batch, heads, seq, seq) arrays.
    Returns: (seq, seq) rollout matrix.

    Algorithm (Abnar & Zuidema 2020):
      1. Average across heads at each layer.
      2. Add residual: A_hat = 0.5*A + 0.5*I
      3. Row-normalise A_hat.
      4. Multiply layer matrices: rollout = A_hat_L @ ... @ A_hat_1
    """
    result = None
    for attn in attention_maps:
        if attn.ndim == 4:          # (batch, heads, seq, seq) → drop batch
            attn = attn[0]
        attn_avg = attn.mean(axis=0)                                    # (seq, seq)
        identity = np.eye(attn_avg.shape[0], dtype=attn_avg.dtype)
        attn_hat = 0.5 * attn_avg + 0.5 * identity
        row_sums = attn_hat.sum(axis=-1, keepdims=True)
        attn_hat = attn_hat / np.where(row_sums == 0, 1.0, row_sums)   # safe divide
        result   = attn_hat if result is None else attn_hat @ result
    return result   # (seq, seq)


def rollout_to_spatial(rollout: np.ndarray, num_patches: int,
                        grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    Extract CLS→patch row from rollout and reshape to (grid_size, grid_size).
    rollout shape: (1 + num_patches, 1 + num_patches), index 0 = CLS token.
    """
    cls_to_patches = rollout[0, 1:].copy()   # (num_patches,)
    mn = cls_to_patches.min()
    mx = cls_to_patches.max()
    if mx > mn:
        cls_to_patches = (cls_to_patches - mn) / (mx - mn)
    h = w = int(num_patches ** 0.5)
    spatial = cls_to_patches.reshape(h, w)
    if h != grid_size:
        spatial = np.array(
            Image.fromarray(spatial.astype(np.float32))
                 .resize((grid_size, grid_size), Image.BILINEAR),
            dtype=np.float32,
        )
    return spatial.astype(np.float32)


# ── Model loaders (called ONCE before the image loop) ─────────────────────────

def load_clip():
    """
    Load CLIP via HuggingFace CLIPVisionModel.
    output_attentions=True makes every layer return its attention weight matrix.
    This is correct — do NOT use open_clip here because open_clip calls
    nn.MultiheadAttention with need_weights=False, so hooks capture None weights.
    """
    from transformers import CLIPVisionModel, CLIPImageProcessor
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model     = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-large-patch14", output_attentions=True
    ).eval()
    return model, processor


def load_llava():
    """LLaVA-1.6 vision tower = CLIP ViT-L/14-336."""
    from transformers import CLIPVisionModel, CLIPImageProcessor
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    model     = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-large-patch14-336", output_attentions=True
    ).eval()
    return model, processor


def load_qwen2vl():
    """
    IMPORTANT: attn_implementation="eager" is required.
    Qwen2-VL defaults to flash_attention_2, which never materialises the
    explicit attention weight matrix — hooks capture nothing and the rollout
    returns a constant uniform map (observed as max_score ≈ 0.051 for every
    image regardless of content).  "eager" forces standard scaled-dot-product
    attention so the hooks can capture real attention weights.
    """
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info   # noqa — imported here to verify
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",   # ← disables flash attention so hooks get weights
    ).eval()
    return model, processor


def load_internvl2():
    """
    Full bfloat16 on single GPU — matches predict_internvl2_backup.py.
    BitsAndBytesConfig + device_map="auto" causes cross-device scatter errors.
    """
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    model = AutoModel.from_pretrained(
        "OpenGVLab/InternVL2-8B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda().eval()
    return model, transform


# ── Per-image rollout functions (model already loaded) ─────────────────────────

def clip_rollout_single(image: Image.Image, model, processor,
                         device: str, grid_size: int) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs,output_attentions=True)
    # out.attentions: tuple of (1, n_heads, seq, seq) per layer
    attn_maps   = [a[0].cpu().numpy() for a in out.attentions]
    num_patches = attn_maps[0].shape[-1] - 1
    rollout     = compute_rollout(attn_maps)
    return rollout_to_spatial(rollout, num_patches, grid_size)


def llava_rollout_single(image: Image.Image, model, processor,
                          device: str, grid_size: int) -> np.ndarray:
    # Identical pipeline to CLIP — both use CLIPVisionModel
    return clip_rollout_single(image, model, processor, device, grid_size)


def qwen2vl_rollout_single(image: Image.Image, model, processor,
                            device: str, grid_size: int) -> np.ndarray:
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image.convert("RGB")},
        {"type": "text",  "text": "Describe."},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
    )
    pixel_values   = inputs["pixel_values"].to("cuda")
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to("cuda")

    attn_maps  = []

    with torch.no_grad():

        outputs = model.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            output_attentions=True,
            return_dict=True,
        )

        attn_maps = outputs.attentions

    for h in hooks:
        h.remove()

    if not attn_maps:
        # Flash attention still active despite attn_implementation="eager".
        # This means the attention module's hook captured nothing.
        # Check: model was loaded with attn_implementation="eager"?
        print("[qwen2vl rollout] WARNING: no attention weights captured — "
              "returning uniform map. Ensure attn_implementation='eager' in load_qwen2vl().")
        return np.full((grid_size, grid_size),
                       1.0 / (grid_size * grid_size), dtype=np.float32)

    # Qwen2-VL has no CLS token → average over all token positions
    rollout = compute_rollout(attn_maps)
    spatial = rollout.mean(axis=0)
    mn, mx  = spatial.min(), spatial.max()
    if mx > mn:
        spatial = (spatial - mn) / (mx - mn)
    n = len(spatial)
    h = w = int(n ** 0.5)
    if h * w == n:
        spatial = spatial.reshape(h, w)
    else:
        spatial = spatial[:h*w].reshape(h, w)
    return np.array(
        Image.fromarray(spatial.astype(np.float32))
             .resize((grid_size, grid_size), Image.BILINEAR),
        dtype=np.float32,
    )


def internvl2_rollout_single(image: Image.Image, model, transform,
                              device: str, grid_size: int) -> np.ndarray:
    """
    InternVL2 uses InternVisionAttention which may call _naive_attn internally
    when flash attention is not available.

    Architecture:
      vision_model.encoder.layers[i].attention      = InternAttention wrapper
      vision_model.encoder.layers[i].attention.attn = InternVisionAttention (inner)

    We hook the INNER attention module to capture the weight matrix.
    If InternVisionAttention uses flash attention internally, the hook captures
    the output tensor only (not weights) — handled via fallback.
    """
    from types import MethodType

    vision_model   = model.vision_model
    pixel_values   = transform(image).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    captured_attns = []

    def patched_naive_attn(self, x):
        """Replaces _naive_attn to capture attention weights."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if getattr(self, "qk_normalization", False):
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        scale = getattr(self, "scale", (C // self.num_heads) ** -0.5)
        attn  = (q * scale) @ k.transpose(-2, -1)
        attn  = attn.softmax(dim=-1)
        attn  = self.attn_drop(attn)
        captured_attns.append(attn.float().detach().cpu().numpy())

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # Patch _naive_attn on the INNER attention module
    # InternVisionEncoderLayer → .attention (InternAttention) → .attn (InternVisionAttention)
    patched = []
    for layer in vision_model.encoder.layers:
        inner_attn = getattr(layer, "attention", None)
        if inner_attn is None:
            continue
        attn_module = getattr(inner_attn, "attn", None)
        if attn_module is None:
            continue
        if hasattr(attn_module, "_naive_attn"):
            attn_module._naive_attn = MethodType(patched_naive_attn, attn_module)
            patched.append(attn_module)

    with torch.no_grad():
        vision_model(pixel_values, output_attentions=False)

    # Restore original _naive_attn (not strictly needed since weights are reloaded
    # per image, but good practice)
    for attn_module in patched:
        if hasattr(attn_module.__class__, "_naive_attn"):
            attn_module._naive_attn = attn_module.__class__._naive_attn

    if not captured_attns:
        print("[internvl2 rollout] No attention weights captured. Returning uniform map.")
        return np.full((grid_size, grid_size),
                       1.0 / (grid_size * grid_size), dtype=np.float32)

    # InternVL2 has 1 CLS token
    num_patches = captured_attns[0].shape[-1] - 1
    rollout     = compute_rollout(captured_attns)
    return rollout_to_spatial(rollout, num_patches, grid_size)


# ── Runner ────────────────────────────────────────────────────────────────────

LOADERS = {
    "clip":      load_clip,
    "llava":     load_llava,
    "qwen2vl":   load_qwen2vl,
    "internvl2": load_internvl2,
}

SINGLE_FNS = {
    "clip":      clip_rollout_single,
    "llava":     llava_rollout_single,
    "qwen2vl":   qwen2vl_rollout_single,
    "internvl2": internvl2_rollout_single,
}


def run_rollout(model_name: str, pilot: bool = False, grid_size: int = GRID_SIZE):
    os.makedirs(ROLLOUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Attention Rollout | model={model_name} | device={device} | grid={grid_size}×{grid_size}")

    sample_ids = pd.read_csv(SAMPLE_IDS_CSV)["u_id"].tolist()
    if pilot:
        sample_ids = sample_ids[:20]
        print(f"Pilot mode: {len(sample_ids)} images.")

    # ── Load model ONCE before the loop ────────────────────────────────────────
    print(f"Loading {model_name}...")
    model_obj, aux = LOADERS[model_name]()
    # Move non-quantized models to device
    if model_name in ("clip", "llava"):
        model_obj = model_obj.to(device)
    print(f"  Model loaded.")

    single_fn  = SINGLE_FNS[model_name]
    ds         = load_from_disk(DATASET_PATH)["test"]
    id_set     = set(sample_ids)
    subset     = ds.filter(lambda x: x["u_id"] in id_set)
    processed  = skipped = errors = 0

    for row in subset:
        u_id     = row["u_id"]
        out_path = os.path.join(ROLLOUT_DIR, f"{u_id}_{model_name}.npy")

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            spatial = single_fn(row["image"], model_obj, aux, device, grid_size)
            np.save(out_path, spatial)
            processed += 1
            print(f"  [{processed}] {u_id} | max={spatial.max():.4f} | saved")
        except Exception as e:
            errors += 1
            print(f"  [ERROR] {u_id}: {e}")

    # Free VRAM
    del model_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone. Saved: {processed} | Skipped: {skipped} | Errors: {errors}")
    print(f"Output: {ROLLOUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 3c — Attention Rollout")
    parser.add_argument("--model",  choices=list(LOADERS.keys()), required=True)
    parser.add_argument("--pilot",  action="store_true")
    parser.add_argument("--grid",   type=int, default=GRID_SIZE)
    args = parser.parse_args()
    run_rollout(args.model, pilot=args.pilot, grid_size=args.grid)
