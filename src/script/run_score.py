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
    parser.add_argument("--label", type=int, choices=[0, 1], default=0)
    parser.add_argument("--fitness_type", type=str, default="normal")

    parser.add_argument("--niter", type=int, default=1000)
    parser.add_argument("--popsize", type=int, default=100)
    parser.add_argument("--toursize", type=int, default=4)
    parser.add_argument("--patchsize", type=int, default=16)

    parser.add_argument("--prob_location_mutate", type=float, default=0.2)
    parser.add_argument("--prob_patch_mutate", type=float, default=0.9)

    parser.add_argument("--guided", type=int, default=0)

    return parser.parse_args()


def save_curve(curve, ylabel, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(curve)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


args = parse_args()

path = (
    f"output/seed={args.seed}"
    f"_log_{args.method}"
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
    f"_fitnesstype={args.fitness_type}"
)

final_dir = os.path.join(path, "final_selected")

success_count = 0
total_adv_score = 0.0
total_psnr = 0.0
num_samples = 0

all_adv_curves = []
all_psnr_curves = []

for file_name in sorted(os.listdir(final_dir)):

    if not file_name.endswith(".txt"):
        continue

    final_path = os.path.join(final_dir, file_name)
    process_path = final_path.replace("final_selected", "selected")

    # =============================
    # Final result
    # =============================
    with open(final_path, "r") as f:
        adv_score, psnr_score = map(
            float,
            f.readline().strip().split()
        )

    total_adv_score += adv_score
    total_psnr += psnr_score

    if adv_score > 0:
        success_count += 1

    num_samples += 1

    # =============================
    # Optimization process
    # =============================
    with open(process_path, "r") as f:

        data = [
            list(map(float, line.split()))
            for line in f
            if line.strip()
        ]

    if len(data) == 0:
        continue

    data = np.asarray(data)

    adv_scores = data[:, 0]
    psnr_scores = data[:, 1]

    all_adv_curves.append(adv_scores)
    all_psnr_curves.append(psnr_scores)

  

# ==================================================
# Statistics
# ==================================================

success_rate = success_count / num_samples
avg_adv_score = total_adv_score / num_samples
avg_psnr = total_psnr / num_samples

print(f"Num Samples : {num_samples}")
print(f"Success Rate: {success_rate:.4f}")
print(f"Avg Adv     : {avg_adv_score:.4f}")
print(f"Avg PSNR    : {avg_psnr:.4f}")

# ===================================
# Visualization
# ===================================

min_len = min(
    min(len(curve) for curve in all_adv_curves),
    min(len(curve) for curve in all_psnr_curves),
)

all_adv_curves = np.array(
    [curve[:min_len] for curve in all_adv_curves]
)

all_psnr_curves = np.array(
    [curve[:min_len] for curve in all_psnr_curves]
)

mean_adv_curve = all_adv_curves.mean(axis=0)

mean_psnr_curve = all_psnr_curves.mean(axis=0)

save_curve(
    mean_adv_curve,
    ylabel="Adv Score",
    title=f"{args.method} Adv Evolution",
    save_path=os.path.join(path, "process_adv.png"),
)

save_curve(
    mean_psnr_curve,
    ylabel="PSNR",
    title=f"{args.method} PSNR Evolution",
    save_path=os.path.join(path, "process_psnr.png"),
)

print("Saved:")
print(os.path.join(path, "process_adv.png"))
print(os.path.join(path, "process_psnr.png"))

# ==================================================
# Save summary
# ==================================================

summary_path = os.path.join(path, "summary.txt")

with open(summary_path, "w") as f:
    f.write(f"Num Samples   : {num_samples}\n")
    f.write(f"Success Rate  : {success_rate:.4f}\n")
    f.write(f"Avg Adv Score : {avg_adv_score:.4f}\n")
    f.write(f"Avg PSNR      : {avg_psnr:.4f}\n")
    f.write(f"Attack Weight : {args.attack_w}\n")
    f.write(f"Recon Weight  : {args.recons_w}\n")

print(f"Saved summary to {summary_path}")