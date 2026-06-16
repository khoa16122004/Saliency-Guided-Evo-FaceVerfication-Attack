import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


LABELS = [0, 1]
FITNESS_TYPES = ["normal", "adaptive"]


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


def load_mean_curves(args, label, fitness_type):
    path = build_path(args, label, fitness_type)

    final_dir = os.path.join(path, "final_selected")

    if not os.path.exists(final_dir):
        print(f"[WARNING] Missing folder: {final_dir}")
        return None, None

    all_best_adv_curves = []
    all_best_psnr_curves = []

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

        # ==================================================
        # Fitness definition
        # ==================================================

        if fitness_type == "adaptive":
            # positive adv contributes 0
            adv_for_fitness = np.minimum(adv_scores, 0)
        else:
            adv_for_fitness = adv_scores

        fitness_scores = (
            args.attack_w * adv_for_fitness
            + args.recons_w * psnr_scores
        )

        # ==================================================
        # Best-so-far according to fitness
        # ==================================================

        best_adv_curve = []
        best_psnr_curve = []

        best_idx = 0

        for i in range(len(fitness_scores)):
            if fitness_scores[i] > fitness_scores[best_idx]:
                best_idx = i

            # IMPORTANT:
            # plot REAL scores, not clipped scores
            best_adv_curve.append(
                adv_scores[best_idx]
            )

            best_psnr_curve.append(
                psnr_scores[best_idx]
            )

        all_best_adv_curves.append(best_adv_curve)
        all_best_psnr_curves.append(best_psnr_curve)

    if len(all_best_adv_curves) == 0:
        return None, None

    min_len = min(
        min(len(c) for c in all_best_adv_curves),
        min(len(c) for c in all_best_psnr_curves),
    )

    adv_curves = np.array([
        c[:min_len]
        for c in all_best_adv_curves
    ])

    psnr_curves = np.array([
        c[:min_len]
        for c in all_best_psnr_curves
    ])

    mean_adv_curve = adv_curves.mean(axis=0)
    mean_psnr_curve = psnr_curves.mean(axis=0)

    return mean_adv_curve, mean_psnr_curve


def plot_metric_comparison(
    args,
    metric_name,
    save_path,
    curve_loader,
):
    fig, axes = plt.subplots(
        1,
        len(LABELS),
        figsize=(12, 5),
        sharey=True,
    )

    if len(LABELS) == 1:
        axes = [axes]

    for ax, label in zip(axes, LABELS):

        for fitness_type in FITNESS_TYPES:

            adv_curve, psnr_curve = curve_loader(
                args,
                label,
                fitness_type,
            )

            if adv_curve is None:
                continue

            curve = (
                adv_curve
                if metric_name == "Adv Score"
                else psnr_curve
            )

            ax.plot(
                curve,
                linewidth=2,
                label=fitness_type.capitalize(),
            )

        ax.set_title(f"Label {label}")
        ax.set_xlabel("Iteration")
        ax.grid(True)

        if metric_name == "Adv Score":
            ax.set_ylabel("Adv Score")
        else:
            ax.set_ylabel("PSNR")

        ax.legend()

    plt.suptitle(
        f"{args.method}: Normal vs Adaptive ({metric_name})"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {save_path}")


def main():
    args = parse_args()

    adv_save_path = (
        f"comparison_adv_"
        f"{args.method}_"
        f"rw={args.recons_w}_"
        f"aw={args.attack_w}.png"
    )

    psnr_save_path = (
        f"comparison_psnr_"
        f"{args.method}_"
        f"rw={args.recons_w}_"
        f"aw={args.attack_w}.png"
    )

    plot_metric_comparison(
        args=args,
        metric_name="Adv Score",
        save_path=adv_save_path,
        curve_loader=load_mean_curves,
    )

    plot_metric_comparison(
        args=args,
        metric_name="PSNR",
        save_path=psnr_save_path,
        curve_loader=load_mean_curves,
    )


if __name__ == "__main__":
    main()