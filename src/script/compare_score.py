import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_selected_file(file_path):
    """
    selected_i.txt

    adv psnr
    adv psnr
    ...
    """
    adv_scores = []
    psnr_scores = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                continue

            adv, psnr = map(float, line.split())

            adv_scores.append(adv)
            psnr_scores.append(psnr)

    return np.array(adv_scores), np.array(psnr_scores)


def load_method(selected_dir):

    all_adv = []
    all_psnr = []

    files = sorted(
        [
            f for f in os.listdir(selected_dir)
            if f.startswith("selected_") and f.endswith(".txt")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    for fname in tqdm(files, desc=os.path.basename(os.path.dirname(selected_dir))):

        fpath = os.path.join(selected_dir, fname)

        adv, psnr = load_selected_file(fpath)

        all_adv.append(adv)
        all_psnr.append(psnr)

    min_len = min(len(x) for x in all_adv)

    all_adv = np.array([x[:min_len] for x in all_adv])
    all_psnr = np.array([x[:min_len] for x in all_psnr])

    mean_adv = all_adv.mean(axis=0)
    std_adv = all_adv.std(axis=0)

    mean_psnr = all_psnr.mean(axis=0)
    std_psnr = all_psnr.std(axis=0)

    return {
        "mean_adv": mean_adv,
        "std_adv": std_adv,
        "mean_psnr": mean_psnr,
        "std_psnr": std_psnr,
        "num_samples": len(all_adv),
    }


def plot_adv(method_results, output_dir):

    plt.figure(figsize=(10, 6))

    for method_name, result in method_results.items():

        y = result["mean_adv"]
        x = np.arange(len(y))

        plt.plot(
            x,
            y,
            linewidth=2,
            label=method_name,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Adversarial Score")
    plt.title("Average Adversarial Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(output_dir, "compare_adv.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


def plot_psnr(method_results, output_dir):

    plt.figure(figsize=(10, 6))

    for method_name, result in method_results.items():

        y = result["mean_psnr"]
        x = np.arange(len(y))

        plt.plot(
            x,
            y,
            linewidth=2,
            label=method_name,
        )

    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.title("Average PSNR")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(output_dir, "compare_psnr.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


def plot_tradeoff(method_results, output_dir):

    plt.figure(figsize=(8, 8))

    for method_name, result in method_results.items():

        plt.plot(
            result["mean_psnr"],
            result["mean_adv"],
            linewidth=2,
            label=method_name,
        )

        plt.scatter(
            result["mean_psnr"][-1],
            result["mean_adv"][-1],
            s=60,
        )

    plt.xlabel("PSNR")
    plt.ylabel("Adversarial Score")
    plt.title("PSNR vs Adversarial Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(output_dir, "compare_tradeoff.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    method_results = {}

    methods = [
        ("GA_single_objective", args.ga_so),
        ("GA_normal", args.ga_normal),
        ("GA_adaptive", args.ga_adaptive),
        # ("NSGAII", args.nsga),
    ]

    for method_name, result_dir in methods:

        if result_dir is None:
            continue

        selected_dir = os.path.join(result_dir, "selected")

        print(f"\nLoading {method_name}")

        method_results[method_name] = load_method(selected_dir)

    plot_adv(method_results, args.output_dir)

    plot_psnr(method_results, args.output_dir)

    plot_tradeoff(method_results, args.output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ga_normal", type=str)
    parser.add_argument("--ga_adaptive", type=str)

    parser.add_argument("--nsga_normal", type=str)
    parser.add_argument("--nsga_adaptive", type=str)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="compare_results"
    )

    args = parser.parse_args()

    main(args)