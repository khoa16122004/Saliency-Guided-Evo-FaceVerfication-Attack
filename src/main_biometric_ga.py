import argparse
import os
import pickle as pkl
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from algorithm import GA
from constant import IMG_DIR, MODEL_RESIZE, OUTPUT_DIR, PAIR_PATH
from dataset import LFW
from fitness import Fitness
from get_architech import get_model
from population_biometric import BiometricPopulation


MASK_FILE_MAP = {
    "binary": "binary_mask.png",
    "eyes": "eyes.png",
    "eyebrows": "eyebrows.png",
    "nose": "nose.png",
    "mouth": "mouth.png",
}


def parse_args():
    default_mask_dir = os.path.join("../", "mask")

    parser = argparse.ArgumentParser(description="Biometric-region constrained GA attack")
    parser.add_argument("--pop_size", type=int, default=100, help="Population size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--prob_mutate_location", type=float, default=0.9, help="Probability of mutating patch location")
    parser.add_argument("--prob_mutate_patch", type=float, default=0.9, help="Probability of mutating patch content")
    parser.add_argument("--mutate_mode", type=str, choices=["single_rectangle", "multiple_rectangles"], default="single_rectangle")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of GA iterations")
    parser.add_argument("--tourament_size", type=int, default=4, help="Tournament size")
    parser.add_argument("--recons_w", type=float, default=0.0, help="Weight for reconstruction objective")
    parser.add_argument("--attack_w", type=float, default=1.0, help="Weight for attack objective")
    parser.add_argument("--update_location_iterval", "--update_location_interval", dest="update_location_iterval", type=int, default=50)
    parser.add_argument("--crossover_type", type=str, choices=["UX", "Blended"], default="Blended")
    parser.add_argument("--fitness_type", type=str, choices=["normal", "adaptive"], default="normal")

    parser.add_argument("--mask_dir", type=str, default=default_mask_dir, help="Root directory of per-pair biometric masks")
    parser.add_argument(
        "--mask_parts",
        type=str,
        default="binary",
        help="Comma-separated mask parts from {binary,eyes,eyebrows,nose,mouth}",
    )
    parser.add_argument(
        "--min_mask_coverage",
        type=float,
        default=1.0,
        help="Minimum average mask coverage in a patch window for valid location (1.0 means fully inside mask)",
    )
    parser.add_argument(
        "--allow_relax_coverage",
        action="store_true",
        help="If no valid location with min coverage, relax threshold automatically",
    )

    parser.add_argument("--label", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num_samples", type=int, default=100, help="Number of matching-label pairs to run")
    parser.add_argument("--start_idx", type=int, default=None, help="Optional start index in pairs list")
    parser.add_argument(
        "--init_from_img2",
        action="store_true",
        help="Initialize a portion of population patches from img2 crop at sampled locations",
    )
    parser.add_argument(
        "--img2_seed_ratio",
        type=float,
        default=0.5,
        help="Fraction of individuals initialized from img2 when init_from_img2 is enabled",
    )
    parser.add_argument(
        "--img2_seed_blend",
        type=float,
        default=0.7,
        help="Blend weight for img2 crop in initialized patch (1.0 means pure img2 crop)",
    )
    parser.add_argument(
        "--img2_seed_only_label1",
        action="store_true",
        help="Apply img2-based initialization only when attack label is 1",
    )
    parser.add_argument("--seed", type=int, default=22520691)
    parser.add_argument("--log", type=str, default="log_biometric")
    parser.add_argument("--pair_path", type=str, default=PAIR_PATH)
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--model_name", type=str, default="restnet_vggface", help="Model to attack")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output root")
    return parser.parse_args()


def parse_mask_parts(mask_parts_raw: str) -> list[str]:
    parts = [p.strip().lower() for p in mask_parts_raw.split(",") if p.strip()]
    if not parts:
        return ["binary"]
    for p in parts:
        if p not in MASK_FILE_MAP:
            raise ValueError(f"Unknown mask part '{p}'. Valid values: {list(MASK_FILE_MAP.keys())}")
    return parts


def load_pair_mask(mask_root: str, pair_idx: int, target_size: int, mask_parts: list[str]) -> np.ndarray:
    pair_mask_dir = os.path.join(mask_root, str(pair_idx))
    if not os.path.isdir(pair_mask_dir):
        raise FileNotFoundError(f"Mask folder not found: {pair_mask_dir}")

    merged = None
    for part in mask_parts:
        file_name = MASK_FILE_MAP[part]
        mask_path = os.path.join(pair_mask_dir, file_name)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise ValueError(f"Could not read mask image: {mask_path}")

        resized = cv2.resize(mask_gray, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        binary = (resized > 127).astype(np.uint8)
        if merged is None:
            merged = binary
        else:
            merged = np.maximum(merged, binary)

    return merged


def build_valid_locations(mask_binary: np.ndarray, patch_size: int, min_coverage: float) -> list[tuple[int, int]]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    mask_t = torch.from_numpy(mask_binary).float().unsqueeze(0).unsqueeze(0)
    cov = F.avg_pool2d(mask_t, kernel_size=patch_size, stride=1).squeeze(0).squeeze(0)
    ys, xs = torch.where(cov >= float(min_coverage))

    valid = [(int(y.item()), int(x.item())) for y, x in zip(ys, xs)]
    return valid


def maybe_relax_locations(mask_binary: np.ndarray, patch_size: int, min_coverage: float) -> tuple[list[tuple[int, int]], float]:
    valid = build_valid_locations(mask_binary, patch_size, min_coverage)
    if valid:
        return valid, min_coverage

    thresholds = [0.75, 0.5, 0.3, 0.1]
    for th in thresholds:
        if th >= min_coverage:
            continue
        valid = build_valid_locations(mask_binary, patch_size, th)
        if valid:
            return valid, th

    return [], min_coverage


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mask_parts = parse_mask_parts(args.mask_parts)

    run_tag = (
        f"s{args.seed}_{args.log}_bioGA_it{args.n_iter}_lb{args.label}"
        f"_ps{args.patch_size}_pop{args.pop_size}_ts{args.tourament_size}"
        f"_mw{args.mask_parts.replace(',', '-')}_cov{args.min_mask_coverage}"
    )
    output_dir = os.path.join(args.output_dir, args.model_name, run_tag)
    output_img_dir = os.path.join(output_dir, "img")
    output_pickle_dir = os.path.join(output_dir, "pickle")
    output_mask_dir = os.path.join(output_dir, "mask_debug")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_pickle_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    model = get_model(args.model_name)
    data = LFW(
        IMG_DIR=args.img_dir,
        MASK_DIR="",
        PAIR_PATH=args.pair_path,
        transform=None,
    )

    to_tensor = transforms.ToTensor()
    size = MODEL_RESIZE[args.model_name]

    start_idx = args.start_idx
    if start_idx is None:
        start_idx = 0 if args.label == 0 else 300

    success_count = 0
    skipped_no_mask = 0
    saved_count = 0

    for pair_idx in tqdm(range(start_idx, start_idx + args.num_samples), desc="Biometric-constrained GA"):
        img1, img2, label = data[pair_idx]
        if label != args.label:
            continue

        sample_seed = args.seed + pair_idx
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)

        img1 = img1.resize((size, size))
        img2 = img2.resize((size, size))
        img1_torch, img2_torch = to_tensor(img1), to_tensor(img2)

        try:
            mask_binary = load_pair_mask(args.mask_dir, pair_idx, size, mask_parts)
        except Exception as err:
            print(f"[Pair {pair_idx}] SKIP | mask load error: {err}")
            skipped_no_mask += 1
            continue

        valid_locations = build_valid_locations(mask_binary, args.patch_size, args.min_mask_coverage)
        used_coverage = args.min_mask_coverage

        if not valid_locations and args.allow_relax_coverage:
            valid_locations, used_coverage = maybe_relax_locations(mask_binary, args.patch_size, args.min_mask_coverage)

        if not valid_locations:
            print(
                f"[Pair {pair_idx}] SKIP | no valid biometric location "
                f"for patch_size={args.patch_size}, min_cov={args.min_mask_coverage}"
            )
            skipped_no_mask += 1
            continue

        fitness = Fitness(
            patch_size=args.patch_size,
            img1=img1_torch,
            img2=img2_torch,
            model=model,
            label=label,
            recons_w=args.recons_w,
            attack_w=args.attack_w,
            fitness_type=args.fitness_type,
            saliency_w=0.0,
            use_saliency_guidance=False,
        )

        population = BiometricPopulation(
            pop_size=args.pop_size,
            patch_size=args.patch_size,
            img_shape=(size, size),
            prob_mutate_location=args.prob_mutate_location,
            prob_mutate_patch=args.prob_mutate_patch,
            valid_locations=valid_locations,
            mutate_mode=args.mutate_mode,
            seed_patch_source=img2_torch,
            use_img2_seed_init=(
                args.init_from_img2
                and (label == 1 or not args.img2_seed_only_label1)
            ),
            img2_seed_ratio=args.img2_seed_ratio,
            img2_seed_blend=args.img2_seed_blend,
        )

        algo = GA(
            n_iter=args.n_iter,
            population=population,
            fitness=fitness,
            tourament_size=args.tourament_size,
            interval_update=args.update_location_iterval,
            crossover_type=args.crossover_type,
        )

        _, adv_img, adv_score, pnsr_score, full_log = algo.solve()

        save_image(adv_img, os.path.join(output_img_dir, f"{pair_idx}.png"))
        algo.save_location_heatmap(output_dir, pair_idx)
        cv2.imwrite(os.path.join(output_mask_dir, f"{pair_idx}.png"), mask_binary * 255)

        result = {
            "adv_score": adv_score,
            "pnsr_score": pnsr_score,
            "log": full_log,
            "pair_idx": pair_idx,
            "label": int(label),
            "num_valid_locations": len(valid_locations),
            "mask_parts": mask_parts,
            "min_mask_coverage": float(used_coverage),
        }

        is_success = adv_score > 0
        if is_success:
            success_count += 1
        status = "SUCCESS" if is_success else "FAIL"
        print(
            f"[Pair {pair_idx}] {status} | adv_score={adv_score:.6f} | "
            f"psnr={pnsr_score:.6f} | valid_locs={len(valid_locations)} | cov={used_coverage:.2f}"
        )

        output_pickle = os.path.join(output_pickle_dir, f"{pair_idx}.pkl")
        with open(output_pickle, "wb") as f:
            pkl.dump(result, f)

        saved_count += 1

    success_rate = (success_count / saved_count) if saved_count > 0 else 0.0
    print(
        f"Run completed. Saved={saved_count}, Success={success_count}, "
        f"SuccessRate={success_rate:.4f}, Skipped={skipped_no_mask}"
    )
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
