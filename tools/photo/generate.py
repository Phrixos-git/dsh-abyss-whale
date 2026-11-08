#!/usr/bin/env python3
"""ComfyUI でアビス・ホエールテーマ用のザトウクジラ画像を生成する。

既存の qwen-whalechan パイプラインと同じ構成（UNETLoader + QwenImage21Cache +
CLIPLoader(qwen3vl, type=qwen_image) + TextEncodeQwenImage21 + EmptyLatentImage +
KSampler euler/simple cfg1 + VAEDecode + SaveImage）。

使い方:
  ~/AI/ComfyUI/.venv/bin/python generate.py <prompt_file> <tag> <seeds> [w] [h] [steps]
例:
  ~/AI/ComfyUI/.venv/bin/python generate.py prompt-whale.txt whale 411,727 1536 896 40

ComfyUI が起動している必要があります（このリポジトリの外のツールなので、
テーマの動作には不要）。出力は photo/out/ に PNG で保存する。
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


def build_workflow(prompt: str, seed: int, width: int, height: int, steps: int, prefix: str) -> dict:
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "", "resolution": 1024}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "QwenImage21Cache",
              "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["6", 0], "positive": ["4", 0], "negative": ["4", 1],
                         "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def main() -> None:
    args = sys.argv[1:]
    prompt_file = HERE / (args[0] if args else "prompt-whale.txt")
    tag = args[1] if len(args) > 1 else "whale"
    seeds = [int(s) for s in (args[2] if len(args) > 2 else "411,727").split(",")]
    width = int(args[3]) if len(args) > 3 else 1536
    height = int(args[4]) if len(args) > 4 else 896
    steps = int(args[5]) if len(args) > 5 else 40

    prompt = prompt_file.read_text(encoding="utf-8").strip()
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        wf = build_workflow(prompt, seed, width, height, steps, f"abyss-whale/{tag}_s{seed}")
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
                images = entry.get("outputs", {}).get("9", {}).get("images", [])
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
