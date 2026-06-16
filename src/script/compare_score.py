import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=22520691)
    parser.add_argument("--method", type=str, default="GA")

    parser.add_argument("--recons_w", type=float, default=0.5)
    parser.add_argument("--attack_w", type=float, default=0.5)
    parser.add_argument("--saliency_w", type=float, default=0.0)

    parser.add_argument("--niter", type=int, default=1000)
    parser.add_argument("--popsize", type=int, default=100)
    parser.add_argument("--toursize", type=int, default=4)
    parser.add_argument("--patchsize", type=int, default=16)

    parser.add_argument("--prob_location_mutate", type=float, default=0.2)
    parser.add_argument("--prob_patch_mutate", type=float, default=0.9)

    parser.add_argument("--guided", type=int, default=0)

    return parser.parse_args()


def build_path(args, label, fitness_type):
    return (
        f"output/seed={args.seed}"
        f"_log_{args.method}"
        f"_niter={args.niter}"
        f"_label={label}"
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


def load_mean_adv_curve(args, label, fitness_type):
    path = build_path(args, label, fitness_type)

    final_dir = os.path.join(path, "final_selected")

    if not os.path.exists(final_dir):
        print(f"Missing: {final_dir}")
        return None

    all_best_adv_curves = []

    for file_name in sorted(os.listdir(final_dir)):
        if not file_name.endswith(".txt"):
            continue

        final_path = os.path.join(final_dir, file_name)
        process_path = final_path.replace(
            "final_selected",
            "selected"
        )

        if not os.path.exists(process_path):
            continue

        with open(process_path, "r") as f:
            data = [
                list(map(float, line.strip().split()))
                for line in f
                if line.strip()
            ]

        if len(data) == 0:
            continue

        adv_scores = np.array([row[0] for row in data])
        psnr_scores = np.array([row[1] for row in data])

        if fitness_type == "adaptive":
            fitness_adv_scores = np.maximum(adv_scores, 0)
        else:
            fitness_adv_scores = adv_scores

        fitness_scores = (
            args.attack_w * fitness_adv_scores
            + args.recons_w * psnr_scores
        )

        best_adv_curve = []
        best_idx = 0

        for i in range(len(fitness_scores)):
            if fitness_scores[i] > fitness_scores[best_idx]:
                best_idx = i

            best_adv_curve.append(
                adv_scores[best_idx]
            )

        all_best_adv_curves.append(best_adv_curve)

    if len(all_best_adv_curves) == 0:
        return None

    min_len = min(
        len(curve)
        for curve in all_best_adv_curves
    )

    curves = np.array([
        curve[:min_len]
        for curve in all_best_adv_curves
    ])

    return curves.mean(axis=0)


def main():
    args = parse_args()

    labels = [0, 1]
    fitness_types = ["normal", "adaptive"]

    fig, axes = plt.subplots(
        1,
        len(labels),
        figsize=(12, 5),
        sharey=True
    )

    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):

        for fitness_type in fitness_types:

            curve = load_mean_adv_curve(
                args,
                label,
                fitness_type
            )

            if curve is None:
                continue

            ax.plot(
                curve,
                linewidth=2,
                label=fitness_type.capitalize()
            )

        ax.set_title(f"Label {label}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Adv Score")
        ax.grid(True)
        ax.legend()

    plt.suptitle(
        f"{args.method}: Normal vs Adaptive Fitness"
    )

    plt.tight_layout()

    save_path = (
        f"comparison_"
        f"{args.method}_"
        f"reconsw={args.recons_w}_"
        f"attackw={args.attack_w}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()