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
        default=["GA", 
                #  "NSGAII",
                 "GA_SO"]
    )

    parser.add_argument(
        "--fitness_types",
        nargs="+",
        default=["normal", "adaptive"]
    )

    parser.add_argument("--saliency_w", type=float, default=0.0)

    parser.add_argument("--label", type=int, default=0)

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


def get_weights(method):
    """
    Folder naming convention:

    GA / NSGAII:
        recons_w = 0.5
        attack_w = 0.5

    GA_SO:
        recons_w = 0.0
        attack_w = 1.0
    """

    if method == "GA_SO":
        return 0.0, 1.0

    return 0.5, 0.5


def build_exp_dir(args, method, fitness_type, label):

    recons_w, attack_w = get_weights(method)

    if method == "GA_SO":
        method = "GA"

    return (
        f"{args.output_root}/"
        f"seed={args.seed}"
        f"_log_{method}"
        f"_niter={args.niter}"
        f"_label={label}"
        f"_reconsw={recons_w}"
        f"_attackw={attack_w}"
        f"_saliencyw={args.saliency_w}"
        f"_guided={args.guided}"
        f"_popsize={args.popsize}"
        f"_toursize={args.toursize}"
        f"_patchsize={args.patchsize}"
        f"_problocationmutate={args.prob_location_mutate}"
        f"_probpatchmutate={args.prob_patch_mutate}"
        f"_fitnesstype={fitness_type}"
    )

def merge_results(result_list):

    all_adv = []
    all_psnr = []

    for result in result_list:

        for curve in result["all_adv"]:
            all_adv.append(curve)

        for curve in result["all_psnr"]:
            all_psnr.append(curve)

    min_len = min(len(x) for x in all_adv)

    all_adv = np.array(
        [x[:min_len] for x in all_adv]
    )

    all_psnr = np.array(
        [x[:min_len] for x in all_psnr]
    )

    return {
        "mean_adv": all_adv.mean(axis=0),
        "std_adv": all_adv.std(axis=0),

        "mean_psnr": all_psnr.mean(axis=0),
        "std_psnr": all_psnr.std(axis=0),

        "num_samples": len(all_adv),

        # giữ raw để merge tiếp nếu cần
        "all_adv": all_adv,
        "all_psnr": all_psnr,
    }


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

    return np.asarray(adv), np.asarray(psnr)


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
            and f.endswith(".txt")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    if len(files) == 0:
        raise RuntimeError(f"No selected files found in {selected_dir}")

    for fname in tqdm(files, leave=False):

        adv, psnr = load_selected_file(
            os.path.join(selected_dir, fname)
        )

        if len(adv) == 0:
            continue

        all_adv.append(adv)
        all_psnr.append(psnr)

    if len(all_adv) == 0:
        raise RuntimeError(f"No valid curves found in {selected_dir}")

    min_len = min(len(x) for x in all_adv)

    all_adv = np.array(
        [x[:min_len] for x in all_adv]
    )

    all_psnr = np.array(
        [x[:min_len] for x in all_psnr]
    )

    return {
        "mean_adv": all_adv.mean(axis=0),
        "std_adv": all_adv.std(axis=0),

        "mean_psnr": all_psnr.mean(axis=0),
        "std_psnr": all_psnr.std(axis=0),

        "num_samples": len(all_adv),

        # raw curves
        "all_adv": all_adv,
        "all_psnr": all_psnr,
    }

def plot_adv(results, save_dir):

    plt.figure(figsize=(10, 6))

    for name, result in results.items():

        x = np.arange(len(result["mean_adv"]))

        mean = result["mean_adv"]
        std = result["std_adv"]

        plt.plot(
            x,
            mean,
            linewidth=2,
            label=name
        )

        # plt.fill_between(
        #     x,
        #     mean - std,
        #     mean + std,
        #     alpha=0.15
        # )

    plt.xlabel("Iteration")
    plt.ylabel("Adversarial Score")
    plt.title("Average Adversarial Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "compare_adv.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_psnr(results, save_dir):

    plt.figure(figsize=(10, 6))

    for name, result in results.items():

        x = np.arange(len(result["mean_psnr"]))

        mean = result["mean_psnr"]
        std = result["std_psnr"]

        plt.plot(
            x,
            mean,
            linewidth=2,
            label=name
        )

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.15
        )

    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.title("Average PSNR")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "compare_psnr.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_tradeoff(results, save_dir):

    plt.figure(figsize=(8, 8))

    for name, result in results.items():

        plt.plot(
            result["mean_psnr"],
            result["mean_adv"],
            linewidth=2,
            label=name
        )

        plt.scatter(
            result["mean_psnr"][-1],
            result["mean_adv"][-1],
            s=50
        )

    plt.xlabel("PSNR")
    plt.ylabel("Adversarial Score")
    plt.title("PSNR vs Adversarial Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "compare_tradeoff.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():

    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    results = {}

    for method in args.methods:

        # ---------------------------------
        # GA_SO: only one configuration
        # ---------------------------------
        if method == "GA_SO":

            fitness_type = "normal"

            labels = [0, 1] if args.label == -1 else [args.label]

            loaded_results = []

            for label in labels:

                exp_dir = build_exp_dir(
                    args,
                    method,
                    fitness_type,
                    label
                )

                if not os.path.exists(exp_dir):
                    continue

                loaded_results.append(
                    load_experiment(exp_dir)
                )

            if len(loaded_results) == 0:
                continue

            results[method] = merge_results(
                loaded_results
            )
            if not os.path.exists(exp_dir):

                print(f"Skip: {exp_dir}")
                continue

            print(f"\nLoading {method}")

            try:

                results[method] = load_experiment(
                    exp_dir
                )

            except Exception as e:

                print(
                    f"Failed loading {method}: {e}"
                )

            continue

        # ---------------------------------
        # GA / NSGAII
        # ---------------------------------
        for fitness_type in args.fitness_types:

            exp_dir = build_exp_dir(
                args,
                method,
                fitness_type
            )

            if not os.path.exists(exp_dir):

                print(f"Skip: {exp_dir}")
                continue

            name = f"{method}_{fitness_type}"

            print(f"\nLoading {name}")

            try:

                results[name] = load_experiment(
                    exp_dir
                )

            except Exception as e:

                print(
                    f"Failed loading {name}: {e}"
                )

    if len(results) == 0:

        print("No experiment found.")
        return

    print("\nLoaded experiments:")

    for name in results:

        print(
            f"  {name} "
            f"(N={results[name]['num_samples']})"
        )

    plot_adv(results, args.save_dir)
    plot_psnr(results, args.save_dir)
    plot_tradeoff(results, args.save_dir)

    print(
        f"\nSaved figures to: {args.save_dir}"
    )


if __name__ == "__main__":
    main()