import argparse
import os
import random
import pickle as pkl

from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image

from fitness import Fitness
from algorithm import LOAP
from get_architech import get_model
from dataset import LFW
from constant import IMG_DIR, PAIR_PATH, OUTPUT_DIR, MODEL_RESIZE


def parse_args():
    parser = argparse.ArgumentParser(description="LOAP attack for face verification")
    parser.add_argument('--patch_size', type=int, default=16, help="Patch size")
    parser.add_argument('--n_iter', type=int, default=1000, help="Number of LOAP iterations")
    parser.add_argument('--recons_w', type=float, default=0.0, help="Deprecated for LOAP. Kept for compatibility.")
    parser.add_argument('--attack_w', type=float, default=1.0, help="Deprecated for LOAP. Kept for compatibility.")
    parser.add_argument('--fitness_type', type=str, choices=['normal', 'adaptive'], default='normal')
    parser.add_argument('--use_saliency_guidance', action='store_true')
    parser.add_argument('--saliency_w', type=float, default=0.0)

    parser.add_argument('--epsilon', type=float, default=0.05, help="LOAP patch update step size")
    parser.add_argument('--stride', type=int, default=1, help="LOAP location move stride")
    parser.add_argument('--optimize_location', action='store_true', help="Enable LOAP location optimization")
    parser.add_argument('--optimize_location_type', type=str, choices=['full', 'random'], default='full')
    parser.add_argument('--signed_grad', action='store_true', help="Use sign(grad) update in LOAP")
    parser.add_argument('--track_best', action='store_true', help="Return the best patch over iterations instead of final iteration")
    parser.add_argument('--print_iter', action='store_true', help="Print LOAP loss/score during optimization iterations")
    parser.add_argument('--print_every', type=int, default=1, help="Print every N iterations when --print_iter is enabled")

    parser.add_argument('--label', type=int, choices=[0, 1], default=0)
    parser.add_argument('--log', type=str, default='log_LOAP')
    parser.add_argument('--seed', type=int, default=22520691)
    parser.add_argument('--pair_path', type=str, default=PAIR_PATH)
    parser.add_argument('--img_dir', type=str, default=IMG_DIR)
    parser.add_argument('--model_name', type=str, default='restnet_vggface', help="Model to attack")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help="Output root")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    output_dir = os.path.join(
        args.output_dir,
        args.model_name,
        (
            f"seed={args.seed}_{args.log}_LOAP_niter={args.n_iter}_label={args.label}"
            f"_attackw={args.attack_w}_reconsw={args.recons_w}"
            f"_patchsize={args.patch_size}"
            f"_epsilon={args.epsilon}_stride={args.stride}_locopt={int(args.optimize_location)}"
            f"_loctype={args.optimize_location_type}_signedgrad={int(args.signed_grad)}"
            f"_trackbest={int(args.track_best)}"
            f"_printiter={int(args.print_iter)}_printevery={args.print_every}"
        ),
    )
    output_img_dir = os.path.join(output_dir, 'img')
    output_pickle_dir = os.path.join(output_dir, 'pickle')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_pickle_dir, exist_ok=True)

    model = get_model(args.model_name)
    data = LFW(
        IMG_DIR=args.img_dir,
        MASK_DIR='',
        PAIR_PATH=args.pair_path,
        transform=None,
    )

    to_tensor = transforms.ToTensor()
    size = MODEL_RESIZE[args.model_name]
    success_count = 0

    start = 0 if args.label == 0 else 300
    end = start + 100

    for i in tqdm(range(start, end), desc='Processing samples'):
        random.seed(args.seed)
        img1, img2, label = data[i]
        img1, img2 = img1.resize((size, size)), img2.resize((size, size))
        img1_torch, img2_torch = to_tensor(img1), to_tensor(img2)

        # LOAP here is configured as attack-only objective.
        fitness = Fitness(
            patch_size=args.patch_size,
            img1=img1_torch,
            img2=img2_torch,
            model=model,
            label=label,
            recons_w=0.0,
            attack_w=1.0,
            fitness_type=args.fitness_type,
            saliency_w=args.saliency_w,
            use_saliency_guidance=args.use_saliency_guidance,
        )

        algo = LOAP(
            n_iter=args.n_iter,
            fitness=fitness,
            epsilon=args.epsilon,
            stride=args.stride,
            optimize_location=args.optimize_location,
            optimize_location_type=args.optimize_location_type,
            signed_grad=args.signed_grad,
            track_best=args.track_best,
            print_iter=args.print_iter,
            print_every=args.print_every,
        )

        _, _, adv_img, adv_score, pnsr_score, full_log = algo.solve(sample_idx=i)

        save_image(adv_img, os.path.join(output_img_dir, f'{i}.png'))
        algo.save_location_heatmap(output_dir, i)

        result = {
            'adv_score': adv_score,
            'pnsr_score': pnsr_score,
            'log': full_log,
        }

        is_success = adv_score > 0
        status = 'SUCCESS' if is_success else 'FAIL'
        print(f"[Sample {i}] {status} | adv_score={adv_score:.6f} | psnr={pnsr_score:.6f}")

        if is_success:
            success_count += 1

        output_pickle = os.path.join(output_pickle_dir, f'{i}.pkl')
        with open(output_pickle, 'wb') as f:
            pkl.dump(result, f)

    total = end - start
    success_rate = success_count / total
    print(f"Run completed. Success: {success_count}/{total} ({success_rate:.4f})")