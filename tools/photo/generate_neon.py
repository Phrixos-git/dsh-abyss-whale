#!/usr/bin/env python3
"""参考画像のような「発光する粒子の大鯨」を ComfyUI で作る（img2img）。

写真版の canvas.png（build.py が置いた不透明な鯨）を下地にして、
ネオンの点描・星座の網・ワイヤーフレームの箱・光の筋へ描き直す。
構図と解剖は下地から引き継ぐので、置き場所（CSS の top とマスク）は
写真版と同じまま使える。

使い方:
  ~/AI/ComfyUI/.venv/bin/python generate_neon.py <prompt_file> <tag> <seeds> [denoise] [steps]
例:
  ~/AI/ComfyUI/.venv/bin/python generate_neon.py prompt-neon.txt neon 11,23 0.65 36

ComfyUI が起動している必要があります（テーマの動作には不要）。
出力は out/ に PNG で保存する。
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import requests

SERVER = "http://127.0.0.1:8188"
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CLIENT_ID = str(uuid.uuid4())

UNET = "qwen_image_2.1_int8_convrot.safetensors"
CLIP = "qwen3vl_8b_int8_convrot.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"


def build_workflow(prompt: str, negative: str, image_name: str, seed: int,
                   width: int, height: int, steps: int, denoise: float, prefix: str) -> dict:
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["2", 0], "prompt": prompt,
                         "negative_prompt": negative, "resolution": 1024}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}},
        "6": {"class_type": "ImageScale", "inputs": {
            "image": ["5", 0], "upscale_method": "lanczos",
            "width": width, "height": height, "crop": "disabled"}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "QwenImage21Cache",
              "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["8", 0], "positive": ["4", 0], "negative": ["4", 1],
                         "latent_image": ["7", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": denoise}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


def upload(image_path: Path) -> str:
    with image_path.open('rb') as fh:
        response = requests.post(f"{SERVER}/upload/image",
                                 files={"image": (image_path.name, fh, "image/png")},
                                 data={"overwrite": "true"}, timeout=120)
    response.raise_for_status()
    body = response.json()
    name = body.get("name")
    sub = body.get("subfolder", "")
    if not name:
        raise RuntimeError(body)
    return f"{sub}/{name}" if sub else name


def main() -> None:
    args = sys.argv[1:]
    prompt_file = HERE / (args[0] if args else "prompt-neon.txt")
    tag = args[1] if len(args) > 1 else "neon"
    seeds = [int(s) for s in (args[2] if len(args) > 2 else "11,23").split(",")]
    denoise = float(args[3]) if len(args) > 3 else 0.65
    steps = int(args[4]) if len(args) > 4 else 36
    base = Path(args[5]) if len(args) > 5 else HERE / "art/canvas-black.png"
    width, height = 1440, 672

    text = prompt_file.read_text(encoding="utf-8")
    blocks = {}
    for line in text.splitlines():
        if line.startswith("## "):
            blocks[line[3:].strip()] = []
        elif blocks:
            blocks[list(blocks)[-1]].append(line)
    prompt = "\n".join(blocks.get("positive", [])).strip()
    negative = "\n".join(blocks.get("negative", [])).strip()

    image_name = upload(base)
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        wf = build_workflow(prompt, negative, image_name, seed, width, height, steps, denoise,
                            f"abyss-neon/{tag}_s{seed}")
        started = time.time()
        response = requests.post(f"{SERVER}/prompt", json={"prompt": wf, "client_id": CLIENT_ID}, timeout=60)
        response.raise_for_status()
        body = response.json()
        if "prompt_id" not in body:
            raise RuntimeError(body)
        prompt_id = body["prompt_id"]
        images: list[dict] = []
        deadline = started + 900
        while time.time() < deadline:
            history = requests.get(f"{SERVER}/history/{prompt_id}", timeout=30).json()
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(json.dumps(status, ensure_ascii=False)[:2000])
                images = entry.get("outputs", {}).get("11", {}).get("images", [])
                break
            time.sleep(1.5)
        if not images:
            raise TimeoutError(f"seed {seed} did not finish")
        for image in images:
            blob = requests.get(f"{SERVER}/view", params={
                "filename": image["filename"],
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            }, timeout=180).content
            target = OUT / f"{tag}_s{seed}.png"
            target.write_bytes(blob)
            print(f"{target} {len(blob)}B {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
