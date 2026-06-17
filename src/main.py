import argparse

from tqdm import tqdm
from population import Population
from individual import Individual
from algorithm import GA, NSGAII
from fitness import Fitness
from get_architech import get_model
from dataset import LFW
import cv2
import numpy as np
import os
from torchvision import transforms
import random
from torchvision.utils import save_image
import pickle as pkl
from constant import IMG_DIR, PAIR_PATH, RESNET_VGGFACE, RESNET_WEBFACE, OUTPUT_DIR, MODEL_RESIZE
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Image Patch Manipulation")
    parser.add_argument('--pop_size', type=int, default=100, help="Population size")
    parser.add_argument('--patch_size', type=int, default=16, help="Size of the patch")
    parser.add_argument('--prob_mutate_location', type=float, default=0.2, help="Probability of mutating the patch location")
    parser.add_argument('--prob_mutate_patch', type=float, default=0.9, help="Probability of mutating the patch itself")
    parser.add_argument('--mutate_mode', type=str, choices=['single_rectangle', 'multiple_rectangles'], default="single_rectangle", help="Mode of mutation for the patch")
    parser.add_argument('--n_iter', type=int, default=1000, help="Number of iterations for the genetic algorithm")
    parser.add_argument('--tourament_size', type=int, default=4, help="Tournament size for selection")
    parser.add_argument('--recons_w', type=float, default=0.5, help="Weight for reconstruction fitness")
    parser.add_argument('--attack_w', type=float, default=0.5, help="Weight for attack fitness")
    parser.add_argument('--baseline', type=str, default='GA', choices=['GA','GA_rules', 'NSGAII'])
    parser.add_argument('--update_location_iterval', '--update_location_interval', dest='update_location_iterval', type=int, default=50)
    parser.add_argument('--crossover_type', type=str, choices=['UX', 'Blended'], default="Blended")
    parser.add_argument('--fitness_type', type=str, choices=['normal', 'adaptive'], help="the type of fitness function", default='normal')
    parser.add_argument('--use_saliency_guidance', action='store_true', help="Enable saliency-guided initialization, mutation, and selection")
    parser.add_argument('--saliency_w', type=float, default=0.0, help="Weight for the saliency objective in GA")
    parser.add_argument('--saliency_noise_scale', type=float, default=0.15, help="Base noise scale for saliency-adaptive patch mutation")
    parser.add_argument('--label', type=int, choices=[0, 1], default=0) 
    parser.add_argument('--log', type=str, default="log")
    parser.add_argument('--seed', type=int, default=22520691)
    parser.add_argument('--pair_path', type=str, default=PAIR_PATH)
    parser.add_argument('--img_dir', type=str, default=IMG_DIR)
    parser.add_argument("--model_name", type=str, default="restnet_vggface", help="the name of the model to attack")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="the directory to save the results")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # save
    output_dir = os.path.join(args.output_dir, args.model_name, f"seed={args.seed}_{args.log}_{args.baseline}_niter={args.n_iter}_label={args.label}_reconsw={args.recons_w}_attackw={args.attack_w}_saliencyw={args.saliency_w}_guided={int(args.use_saliency_guidance)}_popsize={args.pop_size}_toursize={args.tourament_size}_patchsize={args.patch_size}_problocationmutate={args.prob_mutate_location}_probpatchmutate={args.prob_mutate_patch}_fitnesstype={args.fitness_type}_mutatemode={args.mutate_mode}")
    output_img_dir = os.path.join(output_dir, "img")
    output_pickle_dir = os.path.join(output_dir, "pickle")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_pickle_dir, exist_ok=True)
    
    MODEL = get_model(args.model_name)
    DATA = LFW(IMG_DIR=args.img_dir,
               MASK_DIR="", 
               PAIR_PATH=args.pair_path,
               transform=None)
    
    toTensor = transforms.ToTensor()
    size = MODEL_RESIZE[args.model_name]
    success_rate = 0
    results = []
   
    if args.label == 0:
        start = 0
    else:
        start = 300

    end = start + 100

    for i in tqdm(range(start, end), desc="Processing samples"):
        if i == end:
            break
        
        random.seed(args.seed)
        img1, img2, label = DATA[i]
        
        img1, img2 = img1.resize((size, size)), img2.resize((size, size))
        
        img1_torch, img2_torch = toTensor(img1), toTensor(img2)
        img1_feature = MODEL(img1_torch.unsqueeze(0).cuda())
        img2_feature = MODEL(img2_torch.unsqueeze(0).cuda())
        sims = F.cosine_similarity(img1_feature, img2_feature, dim=1)
        # print("Similarity: ", sims.item())

        fitness = Fitness(patch_size=args.patch_size,
                        img1=img1_torch, img2=img2_torch,
                        model=MODEL,
                        label=label,
                        recons_w=args.recons_w,
                        attack_w=args.attack_w,
                        fitness_type=args.fitness_type,
                        saliency_w=args.saliency_w,
                        use_saliency_guidance=args.use_saliency_guidance)

        population = Population(pop_size=args.pop_size,
                                patch_size=args.patch_size,
                                img_shape=(size, size),
                                prob_mutate_location=args.prob_mutate_location,
                                prob_mutate_patch=args.prob_mutate_patch,
                                guidance=fitness.get_guidance(),
                                use_saliency_guidance=args.use_saliency_guidance,
                                saliency_noise_scale=args.saliency_noise_scale,
                                mutate_mode=args.mutate_mode)
        
       
        best_psnr_success = None
        best_ind_success = None

        if args.baseline in ["GA", 'GA_flag', 'GA_rules']:
            algo = GA(n_iter=args.n_iter,
                    population=population,
                    fitness=fitness,
                    tourament_size=args.tourament_size,
                    interval_update=args.update_location_iterval,
                    crossover_type=args.crossover_type)
        
        elif args.baseline == "NSGAII":
            algo = NSGAII(n_iter=args.n_iter,
                        population=population,
                        fitness=fitness,
                        tourament_size=args.tourament_size,
                        interval_update=args.update_location_iterval,
                        crossover_type=args.crossover_type)
        
        if args.baseline == 'GA_sequence':
                P, adv_img, adv_score, pnsr_score = algo.solve_sequential()
        elif args.baseline == 'GA':
                P, adv_img, adv_score, pnsr_score, full_P = algo.solve()

        elif args.baseline == 'GA_flag':
                P, adv_img, adv_score, pnsr_score, best_psnr_success, best_ind_success = algo.solve_save_best()

        elif args.baseline == 'GA_rules':
               P, adv_img, adv_score, pnsr_score, full_P = algo.solve_rule() 

        elif args.baseline == "NSGAII":
                P, adv_img, adv_score, pnsr_score, full_P = algo.solve()
        
        
        # save_image
        save_image(adv_img, os.path.join(output_img_dir, f"{i}.png"))

        
        result = {
                "adv_score": adv_score,
                "pnsr_score": pnsr_score,
                "best_psnr_success": best_psnr_success,
                "best_ind_success": best_ind_success,
                "log": full_P
                }
        
        if adv_score > 0:
                success_rate += 1
        # break
        output_pickle = os.path.join(output_pickle_dir, f'{i}.pkl')            

        with open(output_pickle, 'wb') as f:
                pkl.dump(result, f)
                        
    print(f"Success rate: {success_rate / 100}")
