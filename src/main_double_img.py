import argparse
import os
import pickle as pkl
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from algorithm import GA, NSGAII
from constant import IMG_DIR, MODEL_RESIZE, OUTPUT_DIR, PAIR_PATH
from dataset import LFW
from fitness import FitnessDouble
from get_architech import get_model
from population import Population
from population_biometric import BiometricPopulation


MASK_FILE_MAP = {
    "binary": "binary_mask.png",
    "eyes": "eyes.png",
    "eyebrows": "eyebrows.png",
    "nose": "nose.png",
    "mouth": "mouth.png",
}


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
    mask_t = torch.from_numpy(mask_binary).float().unsqueeze(0).unsqueeze(0)
    cov = F.avg_pool2d(mask_t, kernel_size=patch_size, stride=1).squeeze(0).squeeze(0)
    ys, xs = torch.where(cov >= float(min_coverage))
    return [(int(y.item()), int(x.item())) for y, x in zip(ys, xs)]


def maybe_relax_locations(mask_binary: np.ndarray, patch_size: int, min_coverage: float) -> tuple[list[tuple[int, int]], float]:
    valid = build_valid_locations(mask_binary, patch_size, min_coverage)
    if valid:
        return valid, min_coverage

    for th in [0.75, 0.5, 0.3, 0.1]:
        if th >= min_coverage:
            continue
        valid = build_valid_locations(mask_binary, patch_size, th)
        if valid:
            return valid, th

    return [], min_coverage


def parse_args():
    parser = argparse.ArgumentParser(description="Double-image patch attack (patch on both img1 and img2)")

    parser.add_argument("--mode", type=str, choices=["ga", "biometric"], default="ga", help="Population mode")
    parser.add_argument("--baseline", type=str, choices=["GA", "NSGAII"], default="GA")

    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--prob_mutate_location", type=float, default=0.2)
    parser.add_argument("--prob_mutate_patch", type=float, default=0.9)
    parser.add_argument("--mutate_mode", type=str, choices=["single_rectangle", "multiple_rectangles"], default="single_rectangle")
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--tourament_size", type=int, default=4)
    parser.add_argument("--recons_w", type=float, default=0.0)
    parser.add_argument("--attack_w", type=float, default=1.0)
    parser.add_argument("--update_location_iterval", "--update_location_interval", dest="update_location_iterval", type=int, default=50)
    parser.add_argument("--crossover_type", type=str, choices=["UX", "Blended"], default="Blended")
    parser.add_argument("--fitness_type", type=str, choices=["normal", "adaptive"], default="normal")

    parser.add_argument("--label", type=int, choices=[0, 1], default=1)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=22520691)
    parser.add_argument("--pair_path", type=str, default=PAIR_PATH)
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--model_name", type=str, default="restnet_vggface")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--log", type=str, default="double_img")

    parser.add_argument("--init_from_img2", action="store_true")
    parser.add_argument("--img2_seed_ratio", type=float, default=0.5)
    parser.add_argument("--img2_seed_only_label1", action="store_true")

    parser.add_argument("--mask_dir", type=str, default=os.path.join("../", "mask"))
    parser.add_argument("--mask_parts", type=str, default="binary")
    parser.add_argument("--min_mask_coverage", type=float, default=1.0)
    parser.add_argument("--allow_relax_coverage", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_tag = (
        f"s{args.seed}_{args.log}_{args.mode}_{args.baseline}_double_it{args.n_iter}_lb{args.label}"
        f"_ps{args.patch_size}_pop{args.pop_size}_ts{args.tourament_size}"
        f"_rw{args.recons_w}_aw{args.attack_w}_ft{args.fitness_type}"
    )
    output_dir = os.path.join(args.output_dir, args.model_name, run_tag)
    out_img1_dir = os.path.join(output_dir, "img1")
    out_img2_dir = os.path.join(output_dir, "img2")
    out_pickle_dir = os.path.join(output_dir, "pickle")
    out_mask_dir = os.path.join(output_dir, "mask_debug")
    os.makedirs(out_img1_dir, exist_ok=True)
    os.makedirs(out_img2_dir, exist_ok=True)
    os.makedirs(out_pickle_dir, exist_ok=True)
    if args.mode == "biometric":
        os.makedirs(out_mask_dir, exist_ok=True)

    model = get_model(args.model_name)
    data = LFW(IMG_DIR=args.img_dir, MASK_DIR="", PAIR_PATH=args.pair_path, transform=None)

    to_tensor = transforms.ToTensor()
    size = MODEL_RESIZE[args.model_name]
    mask_parts = parse_mask_parts(args.mask_parts)

    processed = 0
    saved = 0
    success = 0
    skipped_no_mask = 0

    for pair_idx in tqdm(range(len(data)), desc=f"Double-image attack [{args.mode}]"):
        if processed >= args.num_samples:
            break

        img1, img2, label = data[pair_idx]
        if label != args.label:
            continue

        processed += 1

        sample_seed = args.seed + pair_idx
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)

        img1 = img1.resize((size, size))
        img2 = img2.resize((size, size))
        img1_torch, img2_torch = to_tensor(img1), to_tensor(img2)

        valid_locations = None
        used_coverage = args.min_mask_coverage
        if args.mode == "biometric":
            try:
                mask_binary = load_pair_mask(args.mask_dir, pair_idx, size, mask_parts)
            except Exception as err:
                print(f"[Pair {pair_idx}] SKIP | mask load error: {err}")
                skipped_no_mask += 1
                continue

            valid_locations = build_valid_locations(mask_binary, args.patch_size, args.min_mask_coverage)
            if not valid_locations and args.allow_relax_coverage:
                valid_locations, used_coverage = maybe_relax_locations(mask_binary, args.patch_size, args.min_mask_coverage)

            if not valid_locations:
                print(
                    f"[Pair {pair_idx}] SKIP | no valid biometric location "
                    f"for patch_size={args.patch_size}, min_cov={args.min_mask_coverage}"
                )
                skipped_no_mask += 1
                continue

        fitness = FitnessDouble(
            patch_size=args.patch_size,
            img1=img1_torch,
            img2=img2_torch,
            model=model,
            label=label,
            recons_w=args.recons_w,
            attack_w=args.attack_w,
            fitness_type=args.fitness_type,
        )

        use_img2_seed = args.init_from_img2 and (label == 1 or not args.img2_seed_only_label1)
        if args.mode == "ga":
            population = Population(
                pop_size=args.pop_size,
                patch_size=args.patch_size,
                img_shape=(size, size),
                prob_mutate_location=args.prob_mutate_location,
                prob_mutate_patch=args.prob_mutate_patch,
                guidance=fitness.get_guidance(),
                use_saliency_guidance=False,
                saliency_noise_scale=0.15,
                mutate_mode=args.mutate_mode,
                target_patch_source=img2_torch,
                use_img2_seed_init=use_img2_seed,
                img2_seed_ratio=args.img2_seed_ratio,
            )
        else:
            population = BiometricPopulation(
                pop_size=args.pop_size,
                patch_size=args.patch_size,
                img_shape=(size, size),
                prob_mutate_location=args.prob_mutate_location,
                prob_mutate_patch=args.prob_mutate_patch,
                valid_locations=valid_locations,
                mutate_mode=args.mutate_mode,
                seed_patch_source=img2_torch,
                use_img2_seed_init=use_img2_seed,
                img2_seed_ratio=args.img2_seed_ratio,
            )

        if args.baseline == "GA":
            algo = GA(
                n_iter=args.n_iter,
                population=population,
                fitness=fitness,
                tourament_size=args.tourament_size,
                interval_update=args.update_location_iterval,
                crossover_type=args.crossover_type,
            )
        else:
            algo = NSGAII(
                n_iter=args.n_iter,
                population=population,
                fitness=fitness,
                tourament_size=args.tourament_size,
                interval_update=args.update_location_iterval,
                crossover_type=args.crossover_type,
            )

        P, _, _, _, full_log = algo.solve()

        combined, adv_scores, psnr_scores, _ = fitness.benchmark(P)
        best_idx = int(torch.argmax(combined).item())
        best_ind = P[best_idx]

        adv1 = fitness.apply_patch(fitness.img1, best_ind.patch, best_ind.location)
        adv2 = fitness.apply_patch(fitness.img2, best_ind.patch, best_ind.location)
        best_adv = float(adv_scores[best_idx].item())
        best_psnr = float(psnr_scores[best_idx].item())

        save_image(adv1, os.path.join(out_img1_dir, f"{pair_idx}.png"))
        save_image(adv2, os.path.join(out_img2_dir, f"{pair_idx}.png"))
        if hasattr(algo, "save_location_heatmap"):
            algo.save_location_heatmap(output_dir, pair_idx)
        if args.mode == "biometric":
            cv2.imwrite(os.path.join(out_mask_dir, f"{pair_idx}.png"), mask_binary * 255)

        result = {
            "pair_idx": pair_idx,
            "label": int(label),
            "adv_score": best_adv,
            "pnsr_score": best_psnr,
            "location": best_ind.location,
            "log": full_log,
            "mode": args.mode,
            "baseline": args.baseline,
            "mask_parts": mask_parts if args.mode == "biometric" else None,
            "min_mask_coverage": float(used_coverage) if args.mode == "biometric" else None,
            "num_valid_locations": len(valid_locations) if args.mode == "biometric" else None,
        }

        with open(os.path.join(out_pickle_dir, f"{pair_idx}.pkl"), "wb") as f:
            pkl.dump(result, f)

        if best_adv > 0:
            success += 1
        saved += 1

    success_rate = (success / saved) if saved > 0 else 0.0
    print(
        f"Run completed. Processed={processed}, Saved={saved}, Success={success}, "
        f"SuccessRate={success_rate:.4f}, SkippedMask={skipped_no_mask}"
    )
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
