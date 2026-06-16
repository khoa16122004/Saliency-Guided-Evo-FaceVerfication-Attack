import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=22520691)

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["GA", "NSGAII"]
    )

    parser.add_argument(
        "--fitness_types",
        nargs="+",
        default=["normal", "adaptive"]
    )

    parser.add_argument("--recons_w", type=float, default=0.5)
    parser.add_argument("--attack_w", type=float, default=0.5)
    parser.add_argument("--saliency_w", type=float, default=0.0)

    parser.add_argument("--label", type=int, default=-1)
    parser.add_argument("--niter", type=int, default=1000)
    parser.add_argument("--popsize", type=int, default=100)
    parser.add_argument("--toursize", type=int, default=4)
    parser.add_argument("--patchsize", type=int, default=16)

    parser.add_argument("--prob_location_mutate", type=float, default=0.2)
    parser.add_argument("--prob_patch_mutate", type=float, default=0.9)

    parser.add_argument("--guided", type=int, default=0)

    parser.add_argument("--output_root", type=str, default="output")

    parser.add_argument(
        "--save_dir",
        type=str,
        default="compare_results"
    )

    return parser.parse_args()


def compute_final_statistics(exp_dir):

    selected_dir = os.path.join(exp_dir, "selected")

    adv_scores = []
    psnr_scores = []

    files = sorted(
        [
            f for f in os.listdir(selected_dir)
            if f.startswith("selected_")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    for fname in files:

        filepath = os.path.join(
            selected_dir,
            fname
        )

        with open(filepath, "r") as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if len(lines) == 0:
            continue

        adv, psnr = map(
            float,
            lines[-1].split()
        )

        adv_scores.append(adv)
        psnr_scores.append(psnr)

    adv_scores = np.asarray(adv_scores)
    psnr_scores = np.asarray(psnr_scores)

    return {
        "final_adv_mean": adv_scores.mean(),
        "final_adv_std": adv_scores.std(),

        "final_psnr_mean": psnr_scores.mean(),
        "final_psnr_std": psnr_scores.std(),

        "asr": 100.0 * np.mean(adv_scores > 0),

        "num_samples": len(adv_scores)
    }


def build_exp_dir(args, method, fitness_type):

    return (
        f"{args.output_root}/"
        f"seed={args.seed}"
        f"_log_{method}"
        f"_niter={args.niter}"
        f"_label={args.label}"
        f"_reconsw={args.recons_w}"
        f"_attackw={args.attack_w}"
        f"_saliencyw={args.saliency_w}"
        f"_guided={args.guided}"
        f"_popsize={args.popsize}"
        f"_toursize={args.toursize}"
        f"_patchsize={args.patchsize}"
        f"_problocationmutate={args.prob_location_mutate}"
        f"_probpatchmutate={args.prob_patch_mutate}"
        f"_fitnesstype={fitness_type}"
    )


def load_selected_file(txt_file):

    adv = []
    psnr = []

    with open(txt_file, "r") as f:

        for line in f:

            vals = line.strip().split()

            if len(vals) < 2:
                continue

            adv.append(float(vals[0]))
            psnr.append(float(vals[1]))

    return np.array(adv), np.array(psnr)


def load_experiment(exp_dir):

    selected_dir = os.path.join(exp_dir, "selected")

    if not os.path.exists(selected_dir):
        raise FileNotFoundError(selected_dir)

    all_adv = []
    all_psnr = []

    files = sorted(
        [
            f for f in os.listdir(selected_dir)
            if f.startswith("selected_")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    for fname in tqdm(files, leave=False):

        adv, psnr = load_selected_file(
            os.path.join(selected_dir, fname)
        )

        all_adv.append(adv)
        all_psnr.append(psnr)

    min_len = min(len(x) for x in all_adv)

    all_adv = np.array(
        [x[:min_len] for x in all_adv]
    )

    all_psnr = np.array(
        [x[:min_len] for x in all_psnr]
    )

    curve_result = {
        "mean_adv": all_adv.mean(axis=0),
        "std_adv": all_adv.std(axis=0),

        "mean_psnr": all_psnr.mean(axis=0),
        "std_psnr": all_psnr.std(axis=0),
    }

    final_result = compute_final_statistics(
        exp_dir
    )
    
    curve_result.update(final_result)

    return curve_result


def print_summary(results):

    print("\n")
    print("=" * 100)

    header = (
        f"{'Method':<20}"
        f"{'ASR (%)':>12}"
        f"{'Adv Mean':>15}"
        f"{'Adv Std':>12}"
        f"{'PSNR Mean':>15}"
        f"{'PSNR Std':>12}"
    )

    print(header)
    print("-" * 100)

    for name, result in results.items():

        print(
            f"{name:<20}"
            f"{result['asr']:>12.2f}"
            f"{result['final_adv_mean']:>15.4f}"
            f"{result['final_adv_std']:>12.4f}"
            f"{result['final_psnr_mean']:>15.4f}"
            f"{result['final_psnr_std']:>12.4f}"
        )

    print("=" * 100)

def plot_adv(results, save_dir):

    plt.figure(figsize=(10, 6))

    for name, result in results.items():

        x = np.arange(len(result["mean_adv"]))

        mean = result["mean_adv"]
        std = result["std_adv"]

        plt.plot(x, mean, label=name)

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.15
        )

    plt.xlabel("Iteration")
    plt.ylabel("Adversarial Score")
    plt.title("Average Adversarial Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "compare_adv.png"),
        dpi=300
    )

    plt.close()


def plot_psnr(results, save_dir):

    plt.figure(figsize=(10, 6))

    for name, result in results.items():

        x = np.arange(len(result["mean_psnr"]))

        mean = result["mean_psnr"]
        std = result["std_psnr"]

        plt.plot(x, mean, label=name)

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.15
        )

    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.title("Average PSNR")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "compare_psnr.png"),
        dpi=300
    )

    plt.close()


def plot_tradeoff(results, save_dir):

    plt.figure(figsize=(8, 8))

    for name, result in results.items():

        plt.plot(
            result["mean_psnr"],
            result["mean_adv"],
            label=name,
            linewidth=2
        )

        plt.scatter(
            result["mean_psnr"][-1],
            result["mean_adv"][-1]
        )

    plt.xlabel("PSNR")
    plt.ylabel("Adversarial Score")
    plt.title("PSNR vs Adversarial Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "compare_tradeoff.png"),
        dpi=300
    )

    plt.close()


def main():

    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    results = {}

    for method in args.methods:

        for fitness_type in args.fitness_types:

            exp_dir = build_exp_dir(
                args,
                method,
                fitness_type
            )

            if not os.path.exists(exp_dir):

                print(
                    f"Skip: {exp_dir}"
                )

                continue

            name = f"{method}_{fitness_type}"

            print(f"Loading {name}")

            results[name] = load_experiment(
                exp_dir
            )
    print_summary(results)
    plot_adv(results, args.save_dir)
    plot_psnr(results, args.save_dir)
    plot_tradeoff(results, args.save_dir)

    print(
        f"Saved figures to {args.save_dir}"
    )


if __name__ == "__main__":
    main()