import argparse
import json
import os
import sys
from tqdm import tqdm
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
    
from constant import MODEL_RESIZE


import torch
import torch.nn.functional as F
from torchvision import transforms

from constant import IMG_DIR, PAIR_PATH
from dataset import LFW
from get_architech import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate clean face verification accuracy on LFW pairs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save summary and id lists")
    parser.add_argument("--img_dir", type=str, default=IMG_DIR, help="Directory containing LFW images")
    parser.add_argument("--pair_path", type=str, default=PAIR_PATH, help="Path to LFW pairs.txt")
    parser.add_argument("--model_name", type=str, default="restnet_vggface", help="Model name to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold for same-person prediction")
    parser.add_argument("--image_size", type=int, default=160, help="Image resize used before inference")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda or cpu")
    return parser.parse_args()


def predict_label_from_similarity(similarity: float, threshold: float) -> int:
    return 0 if similarity > threshold else 1


def resolve_device(model: torch.nn.Module, device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return next(model.parameters()).device


def evaluate_dataset(model: torch.nn.Module,
                     dataset: LFW,
                     threshold: float,
                     image_size: int,
                     device: torch.device) -> dict[str, object]:
    to_tensor = transforms.ToTensor()

    total = 0
    total_correct = 0
    class_totals = {0: 0, 1: 0}
    class_correct = {0: 0, 1: 0}
    correct_ids = {0: [], 1: []}
    wrong_ids = {0: [], 1: []}
    records = []

    for sample_idx in tqdm(range(len(dataset)), desc="Evaluating dataset"):
        img1, img2, label = dataset[sample_idx]
        img1 = img1.resize((image_size, image_size))
        img2 = img2.resize((image_size, image_size))

        img1_tensor = to_tensor(img1).unsqueeze(0).to(device)
        img2_tensor = to_tensor(img2).unsqueeze(0).to(device)

        with torch.no_grad():
            img1_feature = model(img1_tensor)
            img2_feature = model(img2_tensor)
            similarity = F.cosine_similarity(img1_feature, img2_feature, dim=1).item()

        predicted_label = predict_label_from_similarity(similarity, threshold)
        correct = predicted_label == label

        total += 1
        class_totals[label] += 1
        if correct:
            total_correct += 1
            class_correct[label] += 1
            correct_ids[label].append(sample_idx)
        else:
            wrong_ids[label].append(sample_idx)

        records.append(
            {
                "sample_idx": sample_idx,
                "label": label,
                "predicted_label": predicted_label,
                "similarity": similarity,
                "correct": correct,
            }
        )

    overall_accuracy = total_correct / total if total else 0.0
    class_accuracy = {
        str(label): (class_correct[label] / class_totals[label] if class_totals[label] else 0.0)
        for label in class_totals
    }

    return {
        "total_samples": total,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "class_totals": {str(label): class_totals[label] for label in class_totals},
        "class_correct": {str(label): class_correct[label] for label in class_correct},
        "class_accuracy": class_accuracy,
        "correct_ids": {str(label): correct_ids[label] for label in correct_ids},
        "wrong_ids": {str(label): wrong_ids[label] for label in wrong_ids},
        "records": records,
    }


def write_id_list(file_path: str, sample_ids: list[int]) -> None:
    with open(file_path, "w", encoding="utf-8") as file_obj:
        for sample_idx in sample_ids:
            file_obj.write(f"{sample_idx}\n")


def write_summary(output_dir: str, summary: dict[str, object]) -> None:
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    write_id_list(os.path.join(output_dir, "correct_ids_class_0.txt"), summary["correct_ids"]["0"])
    write_id_list(os.path.join(output_dir, "correct_ids_class_1.txt"), summary["correct_ids"]["1"])
    write_id_list(os.path.join(output_dir, "wrong_ids_class_0.txt"), summary["wrong_ids"]["0"])
    write_id_list(os.path.join(output_dir, "wrong_ids_class_1.txt"), summary["wrong_ids"]["1"])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = get_model(args.model_name)
    size = MODEL_RESIZE[args.model_name]
    device = resolve_device(model, args.device)
    model = model.to(device).eval()

    dataset = LFW(
        IMG_DIR=args.img_dir,
        MASK_DIR="",
        PAIR_PATH=args.pair_path,
        transform=None,
    )

    summary = evaluate_dataset(
        model=model,
        dataset=dataset,
        threshold=args.threshold,
        image_size=size,
        device=device,
    )
    write_summary(args.output_dir, summary)

    print(f"Total samples: {summary['total_samples']}")
    print(f"Total correct: {summary['total_correct']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.4f}")
    print(f"Class 0 accuracy: {summary['class_accuracy']['0']:.4f}")
    print(f"Class 1 accuracy: {summary['class_accuracy']['1']:.4f}")
    print(f"Class 0 correct ids saved: {len(summary['correct_ids']['0'])}")
    print(f"Class 1 correct ids saved: {len(summary['correct_ids']['1'])}")


if __name__ == "__main__":
    main()
