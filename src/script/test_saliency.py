import argparse
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from constant import IMG_DIR, OUTPUT_DIR, PAIR_PATH
from dataset import LFW
from get_architech import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and save Grad-CAM saliency maps for LFW pairs")
    parser.add_argument("--pair_path", type=str, default=PAIR_PATH)
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--model_name", type=str, default="restnet_vggface")
    parser.add_argument("--label", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=22520691)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--overlay_alpha", type=float, default=0.4)
    parser.add_argument("--attack_dir", type=str, default="output")
    parser.add_argument("--output_adv_dir", type=str)
    parser.add_argument(
        "--smoothgrad_samples",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--smoothgrad_std",
        type=float,
        default=0.10,
    )

    return parser.parse_args()


class SmoothGradSimilarityExtractor:

    def __init__(
        self,
        model,
        num_samples=30,
        noise_std=0.10,
    ):
        self.model = model
        self.num_samples = num_samples
        self.noise_std = noise_std

    def generate_pair(self, image1, image2):

        device = next(self.model.parameters()).device

        image1 = image1.unsqueeze(0).to(device)
        image2 = image2.unsqueeze(0).to(device)

        accumulated_grad1 = torch.zeros_like(image1)
        accumulated_grad2 = torch.zeros_like(image2)

        cosine_score = None

        for _ in range(self.num_samples):

            noisy1 = image1 + torch.randn_like(image1) * self.noise_std
            noisy2 = image2 + torch.randn_like(image2) * self.noise_std

            noisy1.requires_grad_(True)
            noisy2.requires_grad_(True)

            self.model.zero_grad()

            x = torch.cat(
                [noisy1, noisy2],
                dim=0,
            )

            embeddings = self.model(x)

            score = F.cosine_similarity(
                embeddings[0:1],
                embeddings[1:2],
                dim=1,
            )

            loss = -score
            loss.backward()

            accumulated_grad1 += noisy1.grad.detach().abs()
            accumulated_grad2 += noisy2.grad.detach().abs()

            cosine_score = score.item()

        saliency1 = accumulated_grad1.mean(dim=1)[0]
        saliency2 = accumulated_grad2.mean(dim=1)[0]

        saliency1 -= saliency1.min()
        saliency1 /= saliency1.max().clamp_min(1e-8)

        saliency2 -= saliency2.min()
        saliency2 /= saliency2.max().clamp_min(1e-8)

        return (
            cosine_score,
            saliency1.cpu(),
            saliency2.cpu(),
        )

    def remove(self):
        pass


def get_target_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    target_layer = getattr(model, layer_name, None)
    if target_layer is None:
        raise ValueError(f"Unknown target layer: {layer_name}")
    return target_layer


def tensor_to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    image_np = np.clip(image_np, 0.0, 1.0)
    return (image_np * 255).astype(np.uint8)


def saliency_to_heatmap(saliency_map: torch.Tensor) -> np.ndarray:
    saliency_map = saliency_map.detach().cpu()
    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max().clamp_min(1e-8)
    saliency_np = saliency_map.numpy()
    saliency_uint8 = np.clip(saliency_np * 255, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(image: torch.Tensor, saliency_map: torch.Tensor, alpha: float) -> np.ndarray:
    image_rgb = tensor_to_uint8_rgb(image)
    heatmap_rgb = saliency_to_heatmap(saliency_map)
    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def save_pair_outputs(output_dir: str,
                      pair_idx: int,
                      label: int,
                      cosine_score: float,
                      img1: torch.Tensor,
                      img2: torch.Tensor,
                      cam1: torch.Tensor,
                      cam2: torch.Tensor,
                      overlay_alpha: float) -> None:
    pair_dir = os.path.join(output_dir, f"pair_{pair_idx:04d}_label_{label}")
    os.makedirs(pair_dir, exist_ok=True)

    save_image(img1, os.path.join(pair_dir, "img1.png"))
    save_image(img2, os.path.join(pair_dir, "img2.png"))
    save_image(cam1.unsqueeze(0), os.path.join(pair_dir, "img1_saliency.png"))
    save_image(cam2.unsqueeze(0), os.path.join(pair_dir, "img2_saliency.png"))

    cv2.imwrite(
        os.path.join(pair_dir, "img1_heatmap.png"),
        cv2.cvtColor(saliency_to_heatmap(cam1), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(pair_dir, "img2_heatmap.png"),
        cv2.cvtColor(saliency_to_heatmap(cam2), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(pair_dir, "img1_overlay.png"),
        cv2.cvtColor(overlay_heatmap(img1, cam1, overlay_alpha), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(pair_dir, "img2_overlay.png"),
        cv2.cvtColor(overlay_heatmap(img2, cam2, overlay_alpha), cv2.COLOR_RGB2BGR),
    )

    with open(os.path.join(pair_dir, "meta.txt"), "w", encoding="utf-8") as file_obj:
        file_obj.write(f"pair_index: {pair_idx}\n")
        file_obj.write(f"label: {label}\n")
        file_obj.write(f"cosine_similarity: {cosine_score:.6f}\n")


def extract_pair_saliency_maps(
    model: torch.nn.Module,
    extractor: SmoothGradSimilarityExtractor,
    img1: torch.Tensor,
    img2: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
    return extractor.generate_pair(img1, img2)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = os.path.join(
        args.output_dir,
        f"gradcam_{args.model_name}_label={args.label}_num={args.num_samples}_seed={args.seed}",
    )
    output_adv_dir = os.path.join(
        args.output_adv_dir,
        f"gradcam_{args.model_name}_label={args.label}_num={args.num_samples}_seed={args.seed}",
    )
    os.makedirs(output_dir, exist_ok=True)

    model = get_model(args.model_name)
    data = LFW(
        IMG_DIR=args.img_dir,
        MASK_DIR="",
        PAIR_PATH=args.pair_path,
        transform=None,
    )
    to_tensor = transforms.ToTensor()
    extractor = SmoothGradSimilarityExtractor(
        model=model,
        num_samples=args.smoothgrad_samples,
        noise_std=args.smoothgrad_std,
    )
    if args.start_idx is not None:
        start_idx = args.start_idx
    else:
        start_idx = 0 if args.label == 0 else 300

    saved_count = 0
    try:
        for pair_idx in range(start_idx, len(data)):
            if saved_count >= args.num_samples:
                break

            img1, img2, label = data[pair_idx]
            adv_img1 = Image.open(
                os.path.join(args.attack_dir, f"{pair_idx}.png")
            ).convert("RGB")
            if label != args.label:
                continue

            img1 = img1.resize((args.image_size, args.image_size))
            adv_img1 = adv_img1.resize((args.image_size, args.image_size))
            img2 = img2.resize((args.image_size, args.image_size))
            img1_tensor = to_tensor(img1)
            adv_img1_tensor = to_tensor(adv_img1)
            img2_tensor = to_tensor(img2)
            

            cosine_score, cam1, cam2 = extract_pair_saliency_maps(
                model=model,
                extractor=extractor,
                img1=img1_tensor,
                img2=img2_tensor,
            )
            cosine_adv_score, cam_adv1, cam_adv2 = extract_pair_saliency_maps(
                model=model,
                extractor=extractor,
                img1=adv_img1_tensor,
                img2=img2_tensor,
            )

            save_pair_outputs(
                output_dir=output_dir,
                pair_idx=pair_idx,
                label=label,
                cosine_score=cosine_score,
                img1=img1_tensor,
                img2=img2_tensor,
                cam1=cam1,
                cam2=cam2,
                overlay_alpha=args.overlay_alpha,
            )
            save_pair_outputs(
                output_dir=output_adv_dir,
                pair_idx=pair_idx,
                label=label,
                cosine_score=cosine_score,
                img1=adv_img1_tensor,
                img2=img2_tensor,
                cam1=cam_adv1,
                cam2=cam_adv2,
                overlay_alpha=args.overlay_alpha,
            )
            saved_count += 1
            print(f"Saved pair {pair_idx} with cosine similarity {cosine_score:.6f}")
    finally:
        extractor.remove()

    print(f"Finished extracting saliency maps for {saved_count} pairs")


if __name__ == "__main__":
    main()