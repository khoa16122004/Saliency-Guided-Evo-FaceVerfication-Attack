import argparse
import os
import random
import pickle as pkl

from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image

from population import Population
from fitness import Fitness
from algorithm import LOAP
from get_architech import get_model
from dataset import LFW
from constant import IMG_DIR, PAIR_PATH, OUTPUT_DIR, MODEL_RESIZE


def parse_args():
    parser = argparse.ArgumentParser(description="LOAP attack for face verification")
    parser.add_argument('--pop_size', type=int, default=100, help="Population size")
    parser.add_argument('--patch_size', type=int, default=16, help="Patch size")
    parser.add_argument('--prob_mutate_location', type=float, default=0.2, help="Probability of mutating patch location for initialization utilities")
    parser.add_argument('--prob_mutate_patch', type=float, default=0.9, help="Probability of mutating patch values for initialization utilities")
    parser.add_argument('--mutate_mode', type=str, choices=['single_rectangle', 'multiple_rectangles'], default='single_rectangle')
    parser.add_argument('--n_iter', type=int, default=1000, help="Number of LOAP iterations")
    parser.add_argument('--tourament_size', type=int, default=4)
    parser.add_argument('--recons_w', type=float, default=0.0, help="Set to 0.0 for LOAP attack-only objective")
    parser.add_argument('--attack_w', type=float, default=1.0, help="Set to 1.0 for LOAP attack-only objective")
    parser.add_argument('--update_location_iterval', '--update_location_interval', dest='update_location_iterval', type=int, default=1)
    parser.add_argument('--crossover_type', type=str, choices=['UX', 'Blended'], default='Blended')
    parser.add_argument('--fitness_type', type=str, choices=['normal', 'adaptive'], default='normal')
    parser.add_argument('--use_saliency_guidance', action='store_true')
    parser.add_argument('--saliency_w', type=float, default=0.0)
    parser.add_argument('--saliency_noise_scale', type=float, default=0.15)

    parser.add_argument('--epsilon', type=float, default=0.05, help="LOAP patch update step size")
    parser.add_argument('--stride', type=int, default=1, help="LOAP location move stride")
    parser.add_argument('--optimize_location', action='store_true', help="Enable LOAP location optimization")
    parser.add_argument('--optimize_location_type', type=str, choices=['full', 'random'], default='full')
    parser.add_argument('--signed_grad', action='store_true', help="Use sign(grad) update in LOAP")

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
            f"_popsize={args.pop_size}_patchsize={args.patch_size}"
            f"_epsilon={args.epsilon}_stride={args.stride}_locopt={int(args.optimize_location)}"
            f"_loctype={args.optimize_location_type}_signedgrad={int(args.signed_grad)}"
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
    success_rate = 0

    start = 0 if args.label == 0 else 300
    end = start + 100

    for i in tqdm(range(start, end), desc='Processing samples'):
        random.seed(args.seed)
        img1, img2, label = data[i]
        img1, img2 = img1.resize((size, size)), img2.resize((size, size))
        img1_torch, img2_torch = to_tensor(img1), to_tensor(img2)

        fitness = Fitness(
            patch_size=args.patch_size,
            img1=img1_torch,
            img2=img2_torch,
            model=model,
            label=label,
            recons_w=args.recons_w,
            attack_w=args.attack_w,
            fitness_type=args.fitness_type,
            saliency_w=args.saliency_w,
            use_saliency_guidance=args.use_saliency_guidance,
        )

        population = Population(
            pop_size=args.pop_size,
            patch_size=args.patch_size,
            img_shape=(size, size),
            prob_mutate_location=args.prob_mutate_location,
            prob_mutate_patch=args.prob_mutate_patch,
            guidance=fitness.get_guidance(),
            use_saliency_guidance=args.use_saliency_guidance,
            saliency_noise_scale=args.saliency_noise_scale,
            mutate_mode=args.mutate_mode,
        )

        algo = LOAP(
            n_iter=args.n_iter,
            population=population,
            fitness=fitness,
            tourament_size=args.tourament_size,
            interval_update=args.update_location_iterval,
            crossover_type=args.crossover_type,
            epsilon=args.epsilon,
            stride=args.stride,
            optimize_location=args.optimize_location,
            optimize_location_type=args.optimize_location_type,
            signed_grad=args.signed_grad,
        )

        _, adv_img, adv_score, pnsr_score, full_log = algo.solve()

        save_image(adv_img, os.path.join(output_img_dir, f'{i}.png'))

        result = {
            'adv_score': adv_score,
            'pnsr_score': pnsr_score,
            'log': full_log,
        }

        if adv_score > 0:
            success_rate += 1

        output_pickle = os.path.join(output_pickle_dir, f'{i}.pkl')
        with open(output_pickle, 'wb') as f:
            pkl.dump(result, f)

    print(f"Success rate: {success_rate / 100}")
