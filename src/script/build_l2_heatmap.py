import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import sys
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from constant import IMG_DIR, MODEL_RESIZE, PAIR_PATH
from dataset import LFW


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build per-pair sliding-patch L2 heatmaps between img1 and img2"
    )
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--pair_path", type=str, default=PAIR_PATH)
    parser.add_argument("--model_name", type=str, default="restnet_vggface", choices=list(MODEL_RESIZE.keys()))
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--label", type=int, default=-1, choices=[-1, 0, 1], help="-1 means all labels")
    parser.add_argument("--start_idx", type=int, default=None, help="Optional start index in pairs list")
    parser.add_argument("--num_samples", type=int, default=100, help="-1 means until dataset ends")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="softmax_inv_l2",
        choices=["softmax_inv_l2", "minmax_inv_l2"],
        help="How to convert L2 map to brightness map",
    )
    parser.add_argument("--softmax_temp", type=float, default=1.0)
    parser.add_argument("--vis_percentile_low", type=float, default=2.0, help="Lower percentile for contrast stretch")
    parser.add_argument("--vis_percentile_high", type=float, default=98.0, help="Upper percentile for contrast stretch")
    parser.add_argument("--vis_gamma", type=float, default=0.7, help="Gamma < 1 boosts bright regions")
    parser.add_argument("--colormap", type=str, default="hot", choices=["hot", "jet", "inferno", "turbo"])
    parser.add_argument("--save_raw_npy", action="store_true")
    parser.add_argument("--output_dir", type=str, default="l2_similarity_heatmaps")
    return parser.parse_args()


def l2_map(img1_t: torch.Tensor, img2_t: torch.Tensor, patch_size: int, stride: int, eps: float) -> torch.Tensor:
    # img tensors are [C, H, W]
    b1 = img1_t.unsqueeze(0)
    b2 = img2_t.unsqueeze(0)

    p1 = F.unfold(b1, kernel_size=patch_size, stride=stride)
    p2 = F.unfold(b2, kernel_size=patch_size, stride=stride)

    diff = p1 - p2
    l2 = torch.sqrt((diff * diff).sum(dim=1).squeeze(0) + eps)

    h = img1_t.shape[1]
    w = img1_t.shape[2]
    out_h = (h - patch_size) // stride + 1
    out_w = (w - patch_size) // stride + 1
    return l2.view(out_h, out_w)


def score_map_from_l2(l2: torch.Tensor, mode: str, temp: float, eps: float) -> torch.Tensor:
    inv = 1.0 / (l2 + eps)

    if mode == "softmax_inv_l2":
        flat = (inv.reshape(-1) / max(temp, eps)).float()
        score = torch.softmax(flat, dim=0).view_as(l2)
        return score

    min_v = inv.min()
    max_v = inv.max()
    denom = (max_v - min_v).clamp_min(eps)
    return (inv - min_v) / denom


def _colormap_code(name: str) -> int:
    if name == "hot":
        return cv2.COLORMAP_HOT
    if name == "inferno":
        return cv2.COLORMAP_INFERNO
    if name == "turbo":
        return cv2.COLORMAP_TURBO
    return cv2.COLORMAP_JET


def save_heatmap_png(
    score: torch.Tensor,
    out_path: str,
    percentile_low: float,
    percentile_high: float,
    gamma: float,
    colormap: str,
) -> None:
    arr = score.detach().cpu().numpy()
    p_low = np.percentile(arr, percentile_low)
    p_high = np.percentile(arr, percentile_high)
    if p_high <= p_low:
        arr = np.clip(arr, 0.0, 1.0)
    else:
        arr = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)

    gamma = max(gamma, 1e-6)
    arr = np.power(arr, gamma)

    gray = (arr * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, _colormap_code(colormap))
    cv2.imwrite(out_path, color)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    heat_dir = os.path.join(args.output_dir, "heat_png")
    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(heat_dir, exist_ok=True)
    if args.save_raw_npy:
        os.makedirs(raw_dir, exist_ok=True)

    size = MODEL_RESIZE[args.model_name]
    data = LFW(IMG_DIR=args.img_dir, MASK_DIR="", PAIR_PATH=args.pair_path, transform=None)
    to_tensor = transforms.ToTensor()

    start_idx = args.start_idx
    if start_idx is None:
        if args.label == 0:
            start_idx = 0
        elif args.label == 1:
            start_idx = 300
        else:
            start_idx = 0

    end_idx = len(data) if args.num_samples < 0 else min(len(data), start_idx + args.num_samples)

    saved = 0
    for idx in tqdm(range(start_idx, end_idx), desc="Building L2 heatmaps"):
        img1, img2, label = data[idx]
        if args.label != -1 and label != args.label:
            continue

        img1 = img1.resize((size, size))
        img2 = img2.resize((size, size))
        img1_t = to_tensor(img1)
        img2_t = to_tensor(img2)

        l2 = l2_map(
            img1_t=img1_t,
            img2_t=img2_t,
            patch_size=args.patch_size,
            stride=args.stride,
            eps=args.eps,
        )
        score = score_map_from_l2(
            l2=l2,
            mode=args.score_mode,
            temp=args.softmax_temp,
            eps=args.eps,
        )

        out_name = f"pair_{idx:04d}_label_{label}.png"
        save_heatmap_png(
            score=score,
            out_path=os.path.join(heat_dir, out_name),
            percentile_low=args.vis_percentile_low,
            percentile_high=args.vis_percentile_high,
            gamma=args.vis_gamma,
            colormap=args.colormap,
        )

        if args.save_raw_npy:
            np.save(
                os.path.join(raw_dir, f"pair_{idx:04d}_label_{label}_l2.npy"),
                l2.detach().cpu().numpy(),
            )
            np.save(
                os.path.join(raw_dir, f"pair_{idx:04d}_label_{label}_score.npy"),
                score.detach().cpu().numpy(),
            )

        saved += 1

    print(f"Done. Saved heatmaps: {saved}")
    print(f"Range: start={start_idx}, end={end_idx}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
