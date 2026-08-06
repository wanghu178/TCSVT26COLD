"""Evaluate adapted RetinexFormer checkpoints on paired EndoVis images."""

import argparse
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from retinexformer import RetinexFormer


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="RetinexFormer EndoVis evaluation")
    parser.add_argument("--opt", required=True, help="Path to a test YAML file")
    parser.add_argument("--weights", help="Override the checkpoint path in the YAML")
    parser.add_argument("--input_dir", help="Override datasets.val.dataroot_lq")
    parser.add_argument("--gt_dir", help="Override datasets.val.dataroot_gt")
    parser.add_argument("--output_dir", default="results", help="Output image directory")
    parser.add_argument("--device", default="cuda", help="Device, for example cuda, cuda:0, or cpu")
    parser.add_argument("--GT_mean", action="store_true", help="Match output mean to the GT mean")
    parser.add_argument("--self_ensemble", action="store_true", help="Use x8 self-ensemble")
    return parser.parse_args()


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def image_paths(directory):
    paths = [path for path in directory.iterdir()
             if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths, key=natural_key)


def load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def psnr(image, target):
    mse = np.mean((image.astype(np.float64) - target.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def _ssim_channel(image, target):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)
    image = image.astype(np.float64)
    target = target.astype(np.float64)
    mu1 = cv2.filter2D(image, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(target, -1, window)[5:-5, 5:-5]
    sigma1 = cv2.filter2D(image ** 2, -1, window)[5:-5, 5:-5] - mu1 ** 2
    sigma2 = cv2.filter2D(target ** 2, -1, window)[5:-5, 5:-5] - mu2 ** 2
    sigma12 = cv2.filter2D(image * target, -1, window)[5:-5, 5:-5] - mu1 * mu2
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)
    return (numerator / denominator).mean()


def ssim(image, target):
    return float(np.mean([_ssim_channel(image[..., channel], target[..., channel])
                          for channel in range(3)]))


def self_ensemble(image, model):
    outputs = []
    for horizontal_flip in (False, True):
        for vertical_flip in (False, True):
            for rotate in (False, True):
                transformed = image
                if horizontal_flip:
                    transformed = torch.flip(transformed, (-2,))
                if vertical_flip:
                    transformed = torch.flip(transformed, (-1,))
                if rotate:
                    transformed = torch.rot90(transformed, dims=(-2, -1))
                output = model(transformed)
                if rotate:
                    output = torch.rot90(output, dims=(-2, -1), k=3)
                if vertical_flip:
                    output = torch.flip(output, (-1,))
                if horizontal_flip:
                    output = torch.flip(output, (-2,))
                outputs.append(output)
    return torch.stack(outputs).mean(dim=0)


def resolve_path(value, release_root):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (release_root / path).resolve()


def load_options(args):
    config_path = Path(args.opt).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        options = yaml.safe_load(stream)
    release_root = Path(__file__).resolve().parent
    validation = options["datasets"]["val"]
    input_dir = resolve_path(args.input_dir or validation["dataroot_lq"], release_root)
    gt_value = args.gt_dir if args.gt_dir is not None else validation.get("dataroot_gt")
    gt_dir = resolve_path(gt_value, release_root) if gt_value else None
    weight_value = args.weights or options["path"]["pretrain_network_g"]
    weights = resolve_path(weight_value, release_root)
    return options, input_dir, gt_dir, weights


def load_model(options, weights, device):
    network_options = dict(options["network_g"])
    network_options.pop("type", None)
    model = RetinexFormer(**network_options)
    checkpoint = torch.load(weights, map_location="cpu")
    state_dict = checkpoint.get("params", checkpoint)
    state_dict = {key[7:] if key.startswith("module.") else key: value
                  for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def main():
    args = parse_args()
    options, input_dir, gt_dir, weights = load_options(args)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if gt_dir is not None and not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory does not exist: {gt_dir}")
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")

    device = torch.device(args.device)
    model = load_model(options, weights, device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    inputs = image_paths(input_dir)
    if not inputs:
        raise RuntimeError(f"No supported images found in {input_dir}")

    targets = {path.stem: path for path in image_paths(gt_dir)} if gt_dir else {}
    if gt_dir:
        missing = [path.name for path in inputs if path.stem not in targets]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} GT image(s), first missing: {missing[0]}")

    psnr_values = []
    ssim_values = []
    print(f"Checkpoint: {weights}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    with torch.inference_mode():
        for input_path in tqdm(inputs, desc="Testing"):
            image_u8 = load_rgb(input_path)
            image = torch.from_numpy(image_u8.astype(np.float32) / 255.0)
            image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            height, width = image.shape[-2:]
            pad_height, pad_width = (-height) % 4, (-width) % 4
            padded = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")
            restored = self_ensemble(padded, model) if args.self_ensemble else model(padded)
            restored = restored[..., :height, :width].clamp(0, 1)
            restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()

            target_u8 = None
            if gt_dir:
                target_u8 = load_rgb(targets[input_path.stem])
                if target_u8.shape != image_u8.shape:
                    raise ValueError(f"Shape mismatch for {input_path.name}: "
                                     f"input {image_u8.shape}, GT {target_u8.shape}")
                target = target_u8.astype(np.float32) / 255.0
                if args.GT_mean:
                    restored_mean = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY).mean()
                    target_mean = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).mean()
                    restored = np.clip(restored * target_mean / max(restored_mean, 1e-8), 0, 1)
                psnr_values.append(psnr(restored, target))

            restored_u8 = np.round(restored * 255.0).astype(np.uint8)
            save_rgb(output_dir / f"{input_path.stem}.png", restored_u8)
            if target_u8 is not None:
                ssim_values.append(ssim(restored_u8, target_u8))

    if psnr_values:
        print(f"PSNR: {np.mean(psnr_values):.6f}")
        print(f"SSIM: {np.mean(ssim_values):.6f}")
    else:
        print(f"Saved {len(inputs)} enhanced image(s). GT was not provided, so metrics were skipped.")


if __name__ == "__main__":
    main()
