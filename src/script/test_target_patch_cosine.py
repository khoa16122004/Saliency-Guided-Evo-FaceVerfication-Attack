import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from constant import IMG_DIR, MODEL_RESIZE, PAIR_PATH
from dataset import LFW
from get_architech import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick test: does adding a target-face patch increase cosine(img1, img2)?"
    )
    parser.add_argument("--model_name", type=str, default="restnet_vggface")
    parser.add_argument("--pair_path", type=str, default=PAIR_PATH)
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--label", type=int, choices=[0, 1], default=1)
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument(
        "--placement",
        type=str,
        choices=["center", "random", "best_of_random"],
        default="center",
        help="Where to place the patch on img1.",
    )
    parser.add_argument(
        "--num_random_locations",
        type=int,
        default=20,
        help="Used only when placement=best_of_random.",
    )
    parser.add_argument("--seed", type=int, default=22520691)
    return parser.parse_args()


def pick_indices_by_label(data: LFW, target_label: int, num_samples: int) -> list[int]:
    indices: list[int] = []
    for idx, line in enumerate(data.lines):
        label = 0 if len(line) == 3 else 1
        if label == target_label:
            indices.append(idx)
            if len(indices) >= num_samples:
                break
    return indices


def apply_patch(img: torch.Tensor, patch: torch.Tensor, x_min: int, y_min: int) -> torch.Tensor:
    patch_size = patch.shape[-1]
    adv = img.clone()
    adv[:, x_min : x_min + patch_size, y_min : y_min + patch_size] = patch
    return adv


def get_locations(size: int, patch_size: int, mode: str, num_random: int) -> list[tuple[int, int]]:
    if patch_size > size:
        raise ValueError(f"patch_size ({patch_size}) must be <= input size ({size})")

    if mode == "center":
        x = (size - patch_size) // 2
        y = (size - patch_size) // 2
        return [(x, y)]

    if mode == "random":
        x = random.randint(0, size - patch_size)
        y = random.randint(0, size - patch_size)
        return [(x, y)]

    locations = [((size - patch_size) // 2, (size - patch_size) // 2)]
    for _ in range(max(1, num_random)):
        x = random.randint(0, size - patch_size)
        y = random.randint(0, size - patch_size)
        locations.append((x, y))
    return locations


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = get_model(args.model_name)
    size = MODEL_RESIZE[args.model_name]
    to_tensor = transforms.ToTensor()

    data = LFW(
        IMG_DIR=args.img_dir,
        MASK_DIR="",
        PAIR_PATH=args.pair_path,
        transform=None,
    )

    selected_indices = pick_indices_by_label(data, args.label, args.num_samples)
    if not selected_indices:
        raise RuntimeError(f"No samples found for label={args.label}")

    clean_cosines: list[float] = []
    patched_cosines: list[float] = []
    deltas: list[float] = []

    with torch.no_grad():
        for pair_idx in tqdm(selected_indices, desc="Testing target patch effect"):
            img1, img2, label = data[pair_idx]
            if label != 1:
                continue
            img1 = img1.resize((size, size))
            img2 = img2.resize((size, size))

            img1_t = to_tensor(img1).to(next(model.parameters()).device)
            img2_t = to_tensor(img2).to(next(model.parameters()).device)

            patch = F.interpolate(
                img2_t.unsqueeze(0),
                size=(args.patch_size, args.patch_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            img1_feat = model(img1_t.unsqueeze(0))
            img2_feat = model(img2_t.unsqueeze(0))
            clean_cos = F.cosine_similarity(img1_feat, img2_feat, dim=1).item()

            best_cos = -1e9
            for x_min, y_min in get_locations(size, args.patch_size, args.placement, args.num_random_locations):
                adv = apply_patch(img1_t, patch, x_min, y_min)
                adv_feat = model(adv.unsqueeze(0))
                adv_cos = F.cosine_similarity(adv_feat, img2_feat, dim=1).item()
                if adv_cos > best_cos:
                    best_cos = adv_cos

            clean_cosines.append(clean_cos)
            patched_cosines.append(best_cos)
            deltas.append(best_cos - clean_cos)

    clean_arr = np.asarray(clean_cosines)
    patched_arr = np.asarray(patched_cosines)
    delta_arr = np.asarray(deltas)
    improved = (delta_arr > 0).sum()

    print("\n=== Target Patch Quick Test ===")
    print(f"Model: {args.model_name}")
    print(f"Label: {args.label}")
    print(f"Samples: {len(clean_arr)}")
    print(f"Patch size: {args.patch_size}")
    print(f"Placement: {args.placement}")
    if args.placement == "best_of_random":
        print(f"Num random locations: {args.num_random_locations}")
    print(f"Mean cosine (clean):   {clean_arr.mean():.6f}")
    print(f"Mean cosine (patched): {patched_arr.mean():.6f}")
    print(f"Mean delta:            {delta_arr.mean():.6f}")
    print(f"Median delta:          {np.median(delta_arr):.6f}")
    print(f"Improved count:        {improved}/{len(delta_arr)}")
    print(f"Improved ratio:        {100.0 * improved / len(delta_arr):.2f}%")
    print(f"Delta > 0.01 count:    {(delta_arr > 0.01).sum()}/{len(delta_arr)}")


if __name__ == "__main__":
    main()
