from population import Population
from fitness import Fitness
from individual import Individual
import torch
import random
import os
import cv2
from torchvision.utils import save_image
from tqdm import tqdm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np
from pymoo.util.randomized_argsort import randomized_argsort


def _init_location_grid(img_shape: tuple[int, int], grid_size: int) -> np.ndarray:
    img_h, img_w = img_shape
    grid_h = (img_h + grid_size - 1) // grid_size
    grid_w = (img_w + grid_size - 1) // grid_size
    return np.full((grid_h, grid_w), np.nan, dtype=np.float32)


def _location_to_grid_index(location: tuple[int, int, int, int], grid_size: int, grid_shape: tuple[int, int]) -> tuple[int, int]:
    x_min, x_max, y_min, y_max = location
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    row = center_x // grid_size
    col = center_y // grid_size

    row = max(0, min(row, grid_shape[0] - 1))
    col = max(0, min(col, grid_shape[1] - 1))
    return row, col


def _update_location_grid_from_pool(
    location_grid_best: np.ndarray,
    pool: list['Individual'],
    adv_scores: torch.Tensor,
    grid_size: int,
) -> np.ndarray:
    adv_np = adv_scores.detach().cpu().numpy()
    for ind, adv_score in zip(pool, adv_np):
        row, col = _location_to_grid_index(ind.location, grid_size, location_grid_best.shape)
        current = location_grid_best[row, col]
        if np.isnan(current) or adv_score > current:
            location_grid_best[row, col] = float(adv_score)
    return location_grid_best


def _save_location_heatmap(
    location_grid_best: np.ndarray,
    grid_size: int,
    output_root: str,
    sample_id: int,
) -> str:
    valid_mask = ~np.isnan(location_grid_best)
    vis = np.zeros_like(location_grid_best, dtype=np.float32)

    if valid_mask.any():
        valid_scores = location_grid_best[valid_mask]
        score_min = float(valid_scores.min())
        score_max = float(valid_scores.max())
        if score_max > score_min:
            vis[valid_mask] = (valid_scores - score_min) / (score_max - score_min)
        else:
            vis[valid_mask] = 1.0

    heat_u8 = np.clip(vis * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_HOT)
    heat_color = cv2.resize(
        heat_color,
        (heat_color.shape[1] * grid_size, heat_color.shape[0] * grid_size),
        interpolation=cv2.INTER_NEAREST,
    )

    output_location_dir = os.path.join(output_root, "location")
    os.makedirs(output_location_dir, exist_ok=True)
    save_path = os.path.join(output_location_dir, f"elite_{sample_id}.png")
    cv2.imwrite(save_path, heat_color)
    return save_path

class GA:
    def __init__(self, n_iter: int, 
                 population: 'Population', 
                 fitness: 'Fitness',
                 tourament_size: int, 
                 interval_update: int,
                 crossover_type: str):
        self.n_iter = n_iter
        self.pop = population
        self.args = self.pop.get_params
        self.tourament_size = tourament_size
        self.fitness = fitness
        self.interval_update = interval_update
        self.crossover_type = crossover_type
        self.arkive = [] 
        self.location_grid_size = max(1, int(self.pop.patch_size))
        self.location_grid_best = _init_location_grid(self.pop.img_shape, self.location_grid_size)

    def _update_location_grid(self, pool: list['Individual'], adv_scores: torch.Tensor) -> None:
        self.location_grid_best = _update_location_grid_from_pool(
            self.location_grid_best,
            pool,
            adv_scores,
            self.location_grid_size,
        )

    def save_location_heatmap(self, output_root: str, sample_id: int) -> str:
        return _save_location_heatmap(
            self.location_grid_best,
            self.location_grid_size,
            output_root,
            sample_id,
        )

    def solve(self):
        """
        Parallel update the location and content of individuals
        """
        
        P = self.pop.P
        
        # arkive = []
        
        for i in tqdm(range(self.n_iter)):
            # if i == 4:
              #  raise
            O_P = [] # list['individual']
            for j in range(self.pop.pop_size // 2):
                parent1, parent2 = random.sample(self.pop.P, 2)  
                if self.crossover_type == 'Blended':              
                    offstring_1, offstring_2 = parent1.crossover_blended(parent2)
                elif self.crossover_type == 'UX':
                    offstring_1, offstring_2 = parent1.crossover_UX(parent2)
               
                offstring_1.mutate()
                offstring_2.mutate()
                
                O_P.append(offstring_1)
                O_P.append(offstring_2)

            O_P.extend(self.pop.P)            
            P_final = []
            for i in range(2):
                # shuffle 2 lần nên lưu ra 2 x iterations
                random.shuffle(O_P)
                P_ = self.tourament_selection(O_P)
                P_final.extend(P_)
            self.pop.P = P_final
            # print(len(P_final))    
            if self.has_converged(P):
                print(f"Convergence reached at iteration {i+1}. Terminating early.")
                break
            
            
        adv_img, adv_score, psnr_score= self.save_best(self.pop.P)
   
        return self.pop.P, adv_img, adv_score, psnr_score, self.arkive   


    def solve_rule(self):
        """
        Parallel update the location and content of individuals
        """
        
        P = self.pop.P
        #arkive = []
        for i in tqdm(range(self.n_iter)):  
            O_P = [] # list['individual']
            for j in range(self.pop.pop_size//2):
                parent1, parent2 = random.sample(self.pop.P, 2)  
                if self.crossover_type == 'Blended':              
                    offstring_1, offstring_2 = parent1.crossover_blended(parent2)
                elif self.crossover_type == 'UX':
                    offstring_1, offstring_2 = parent1.crossover_UX(parent2)
                
                offstring_1.mutate()
                offstring_2.mutate()
                
                O_P.append(offstring_1)
                O_P.append(offstring_2)

            O_P.extend(self.pop.P)            
            P_final = []
            for i in range(2):
                random.shuffle(O_P)
                P_ = self.tourament_selection(O_P)
                P_final.extend(P_)

            self.pop.P = P_final
            if self.has_converged(P):
                print(f"Convergence reached at iteration {i+1}. Terminating early.")
                break
            
            
        adv_img, adv_score, pnsr_score= self.save_best(self.pop.P)
   
        return self.pop.P, adv_img, adv_score, pnsr_score,self.arkive 

    

    def update_arkive(self, arkive: list['Individual'], new_ind: 'Individual'):

        if len(arkive) == 0:
            return [new_ind]
        to_remove = []

        for i, item in enumerate(arkive):
            dominant_status = self.is_dominant(item, new_ind)
            if dominant_status == 1:
                return arkive

            elif dominant_status == 2:
                to_remove.append(i)
 
        for idx in reversed(to_remove):
            arkive.pop(idx)
        arkive.append(new_ind)

        return arkive

    def solve_save_best(self):
        """
        Parallel update the location and content of individuals
        """
        
        P = self.pop.P

        best_success_psnr_score = -100000
        best_success_psnr_indi = None


        for i in tqdm(range(self.n_iter)):  
            O_P = [] # list['individual']
            for j in range(self.pop.pop_size // 2):
                parent1, parent2 = random.sample(self.pop.P, 2)  
                if self.crossover_type == 'Blended':              
                    offstring_1, offstring_2 = parent1.crossover_blended(parent2)
                elif self.crossover_type == 'UX':
                    offstring_1, offstring_2 = parent1.crossover_UX(parent2)
                
                offstring_1.mutate()
                offstring_2.mutate()
                
                O_P.append(offstring_1)
                O_P.append(offstring_2)

            O_P.extend(self.pop.P)
            random.shuffle(O_P)
            P, current_success_psnr_score, current_success_psnr_indi = self.tourament_selection_take_best(O_P)
            
            if current_success_psnr_score:
                if best_success_psnr_score < current_success_psnr_score:
                    best_success_psnr_score = current_success_psnr_score
                    best_success_psnr_indi = current_success_psnr_indi 
            # print("Best psnr with successfully: ", best_success_psnr_score)
            self.pop.P = P
            # self.save_best(P)
            
            if self.has_converged(P):
                print(f"Convergence reached at iteration {i+1}. Terminating early.")
                break
            
            
        adv_img, adv_score, pnsr_score= self.save_best(self.pop.P)
   
        return self.pop.P, adv_img, adv_score, pnsr_score, best_success_psnr_score, best_success_psnr_indi     




    def solve_sequential(self):
        """
        Sequential update the location and content of individuals
        """
        self.prob_mutate_location = 0.0 # mutate for random_location 
        P = self.pop.P
        
        for i in tqdm(range(self.n_iter)):  
            O_P = [] # list['individual']
            for j in range(self.pop.pop_size // 2):
                parent1, parent2 = random.sample(self.pop.P, 2)                
                offstring_1, offstring_2 = parent1.crossover(parent2)
                
                if i % self.interval_update == 0:
                    print("Mutate location")
                    offstring_1.mutate_location()
                    offstring_2.mutate_location()
                else:
                    offstring_1.mutate_content()
                    offstring_2.mutate_content()
                
                O_P.append(offstring_1)
                O_P.append(offstring_2)

            O_P.extend(self.pop.P)
            random.shuffle(O_P)
            P = self.tourament_selection(O_P)
            self.pop.P = P
            # self.save_best(P)
            
            if self.has_converged(P):
                print(f"Convergence reached at iteration {i+1}. Terminating early.")
                break
            
        adv_img, adv_score, pnsr_score= self.save_best(self.pop.P)
   
        return self.pop.P, adv_img, adv_score, pnsr_score   
   
    
    def has_converged(self, population: list['Individual']) -> bool:
        """
        Check if all patches in the population are identical.
        If they are, the population has converged.
        """
        first_patch = population[0].patch
        for individual in population[1:]:
            if not torch.equal(first_patch, individual.patch):  # Check if patches are identical
                return False
        return True
    
         
    def tourament_selection(self, pool: list['Individual']) -> list['Individual']:
        pool_fitness, adv_scores, psnr_scores, saliency_scores = self.fitness.benchmark(pool)
        self._update_location_grid(pool, adv_scores)
        # self.arkive.append({'adv_scores': adv_scores.cpu(), 'psnr_scores': psnr_scores.cpu()})
        winner = []
        adv_scores_save = []
        psnr_scores_save = []
        saliency_scores_save = []
        for i in range(0, len(pool), self.tourament_size):

            idx = i + torch.argmax(pool_fitness[i:i+self.tourament_size])
            best_ind = pool[idx]
            adv_scores_save.append(adv_scores[idx].cpu())
            psnr_scores_save.append(psnr_scores[idx].cpu())
            saliency_scores_save.append(saliency_scores[idx].cpu())
    
            winner.append(best_ind)
        # print(len(winner)) 
        self.arkive.append({"adv_scores_log": adv_scores_save, "psnr_scores_log": psnr_scores_save, "saliency_scores_log": saliency_scores_save}) 
        return winner

    def tourament_selection_rules(self, pool: list['Individual']) -> list['Individual']:
        pool_fitness, adv_scores, psnr_scores, saliency_scores = self.fitness.benchmark(pool)
        self._update_location_grid(pool, adv_scores)
        adv_scores_save = []
        psnr_scores_save = []
        saliency_scores_save = []

        winners = []
        

        for i in range(0, len(pool), self.tourament_size):
            current_indices = torch.arange(i, min(i + self.tourament_size, len(pool))).cuda()
            adv_success_indices = current_indices[torch.where(adv_scores[current_indices] >= 0)[0]]

            if adv_success_indices.shape[0] > 0:
                best_index = adv_success_indices[torch.argmax(psnr_scores[adv_success_indices])]
            else:
                best_index = current_indices[torch.argmax(adv_scores[current_indices])]
            winners.append(pool[best_index])
            adv_scores_save.append(adv_scores[best_index].cpu())
            psnr_scores_save.append(psnr_scores[best_index].cpu())
            saliency_scores_save.append(saliency_scores[best_index].cpu())

        self.arkive.append({"adv_scores_log": adv_scores_save, "psnr_scores_log": psnr_scores_save, "saliency_scores_log": saliency_scores_save})

        return winners


    def tourament_selection_take_best(self, pool: list['Individual']) -> list['Individual']:
        pool_fitness, adv_scores, psnr_scores, _ = self.fitness.benchmark(pool)
        self._update_location_grid(pool, adv_scores)
        winner = []

        current_best_success_psnr = None 
        current_best_success_ind = None

        for i in range(0, len(pool), self.tourament_size):
            idx = i + torch.argmax(pool_fitness[i:i+self.tourament_size])
            winner.append(pool[idx])
        
        best_adv_score_idx = torch.argmax(adv_scores)
        if adv_scores[best_adv_score_idx] >= 0:
            current_best_success_psnr = psnr_scores[best_adv_score_idx]
            current_best_success_ind = pool[best_adv_score_idx]            
    
        return winner, current_best_success_psnr, current_best_success_ind
   

    def save_best(self, P: list['Individual']) -> None:
        fitness, adv_scores, psnr_scores, _ = self.fitness.benchmark(P)
        best_idx = torch.argmax(fitness)
        best_patch = P[best_idx]
        best_adv_img = self.fitness.apply_patch_to_image(best_patch.patch, best_patch.location)
        best_adv_score = adv_scores[best_idx]
        best_psnr_score = psnr_scores[best_idx]
        print(f"Best_adv: {best_adv_score}, Best_psnr: {best_psnr_score}")
        # save_image(best_adv_img, 'process.png')
        return best_adv_img, best_adv_score.item(), best_psnr_score.item()
        

selector = NonDominatedSorting()

class NSGAII:
    def __init__(self, n_iter: int, 
                 population: 'Population', 
                 fitness: 'Fitness',
                 tourament_size: int, 
                 interval_update: int,
                 crossover_type: str):
        self.n_iter = n_iter
        self.pop = population
        self.args = self.pop.get_params
        self.tourament_size = tourament_size
        self.fitness = fitness
        self.interval_update = interval_update
        self.crossover_type = crossover_type
        self.arkive = []  
        self.location_grid_size = max(1, int(self.pop.patch_size))
        self.location_grid_best = _init_location_grid(self.pop.img_shape, self.location_grid_size)

    def _update_location_grid(self, pool: list['Individual'], adv_scores: torch.Tensor) -> None:
        self.location_grid_best = _update_location_grid_from_pool(
            self.location_grid_best,
            pool,
            adv_scores,
            self.location_grid_size,
        )

    def save_location_heatmap(self, output_root: str, sample_id: int) -> str:
        return _save_location_heatmap(
            self.location_grid_best,
            self.location_grid_size,
            output_root,
            sample_id,
        )

    def solve(self):
        """
        Parallel update the location and content of individuals
        """
        
        P = self.pop.P
        #arkive = []
        for i in tqdm(range(self.n_iter)):  
            O_P = [] # list['individual']
            for j in range(self.pop.pop_size // 2):
                parent1, parent2 = random.sample(self.pop.P, 2)  
                if self.crossover_type == 'Blended':              
                    offstring_1, offstring_2 = parent1.crossover_blended(parent2)
                elif self.crossover_type == 'UX':
                    offstring_1, offstring_2 = parent1.crossover_UX(parent2)
                
                offstring_1.mutate()
                offstring_2.mutate()
                
                O_P.append(offstring_1)
                O_P.append(offstring_2)

            O_P.extend(self.pop.P)
            random.shuffle(O_P)
            P = self.selection(O_P)
            self.pop.P = P
            #arkive.append(P)
            # self.save_best(P)
            # for ind in P:
            #    arkive = self.update_arkive(arkive, ind)
            # print(len(arkive)) 
            if self.has_converged(P):
                print(f"Convergence reached at iteration {i+1}. Terminating early.")
                break
            
        adv_img, adv_score, pnsr_score= self.save_best(self.pop.P)
   
        return self.pop.P, adv_img, adv_score, pnsr_score,self.arkive  

    def update_arkive(self, arkive: list['Individual'], new_ind: 'Individual'):

        if len(arkive) == 0:
            return [new_ind]
        to_remove = []

        for i, item in enumerate(arkive):
            dominant_status = self.is_dominant(item, new_ind)
            if dominant_status == 1:
                return arkive

            elif dominant_status == 2:
                to_remove.append(i)
 
        for idx in reversed(to_remove):
            print("POP up")
            arkive.pop(idx)
        arkive.append(new_ind)

        return arkive
   
    
    def has_converged(self, population: list['Individual']) -> bool:
        """
        Check if all patches in the population are identical.
        If they are, the population has converged.
        """
        first_patch = population[0].patch
        for individual in population[1:]:
            if not torch.equal(first_patch, individual.patch):  # Check if patches are identical
                return False
        return True
    
    
    def selection(self, pool: list['Individual']) -> list['Individual']:
        _, adv_scores, fsnr_scores, saliency_scores = self.fitness.benchmark(pool)
        self._update_location_grid(pool, adv_scores)
        adv_scores_save = []
        psnr_scores_save = []
        saliency_scores_save = []

       
        # selection minimize for NSGAII
        objective_terms = [-adv_scores, -fsnr_scores]
        if self.fitness.saliency_w > 0 or self.fitness.use_saliency_guidance:
            objective_terms.append(-saliency_scores)
        F = np.array(torch.stack(objective_terms, dim=1).cpu().detach())
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=self.pop.pop_size)
        survivors = []

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calculating_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pool[i].rank = k
                pool[i].crowding = crowding_of_front[j]

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > self.pop.pop_size:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(self.pop.pop_size - len(survivors))]
 
            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])
            index = list(front[I])
            adv_scores_save.append(adv_scores[index])
            psnr_scores_save.append(fsnr_scores[index])
            saliency_scores_save.append(saliency_scores[index])

            # adv_scores_save.append(pool[front[I]])
            # psnr_scores_save.append(pool[front[I]])
        
        self.arkive.append({"adv_scores_log": adv_scores_save, "psnr_scores_log": psnr_scores_save, "saliency_scores_log": saliency_scores_save}) 
        return [pool[i] for i in survivors]

    
    def tourament_selection(self, pool: list['Individual']) -> list['Individual']:
        pool_fitness, _, _, _ = self.fitness.benchmark(pool)
        winner = []
        for i in range(0, len(pool), self.tourament_size):
            idx = i + torch.argmax(pool_fitness[i:i+self.tourament_size])
            winner.append(pool[idx])
            
        return winner
    
    
    
    
    def save_best(self, P: list['Individual']) -> None:
        fitness, adv_scores, psnr_scores, _ = self.fitness.benchmark(P)
        success_mask = adv_scores > 0
        if success_mask.any():
            candidate_idx = torch.where(success_mask)[0]
            local_best = torch.argmax(psnr_scores[candidate_idx])
            best_idx = candidate_idx[local_best]
        else:
            best_idx = torch.argmax(adv_scores)

        best_patch = P[best_idx]

        best_adv_img = self.fitness.apply_patch_to_image(
            best_patch.patch,
            best_patch.location
        )


        return (
            best_adv_img,
            adv_scores[best_idx].item(),
            psnr_scores[best_idx].item(),
        )
                    
def calculating_crowding_distance(F):
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # sort each column and get index
        I = np.argsort(F, axis=0, kind='mergesort')

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F])

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity
    return crowding



class LOAP:
    def __init__(self, n_iter: int,
                 fitness: 'Fitness',
                 epsilon: float = 0.05,
                 stride: int = 1,
                 optimize_location: bool = True,
                 optimize_location_type: str = 'full',
                 signed_grad: bool = True,
                 track_best: bool = True,
                 print_iter: bool = False,
                 print_every: int = 1):
        self.n_iter = n_iter
        self.fitness = fitness
        self.epsilon = epsilon
        self.stride = stride
        self.optimize_location = optimize_location
        self.optimize_location_type = optimize_location_type
        self.signed_grad = signed_grad
        self.track_best = track_best
        self.print_iter = print_iter
        self.print_every = max(1, print_every)
        self.arkive = []
        _, self.img_h, self.img_w = self.fitness.img1.shape
        self.patch_size = self.fitness.patch_size
        self.location_grid_size = max(1, int(self.patch_size))
        self.location_grid_best = _init_location_grid((self.img_h, self.img_w), self.location_grid_size)

    def _update_location_grid(self, location: tuple[int, int, int, int], adv_score: float) -> None:
        row, col = _location_to_grid_index(location, self.location_grid_size, self.location_grid_best.shape)
        current = self.location_grid_best[row, col]
        if np.isnan(current) or adv_score > current:
            self.location_grid_best[row, col] = float(adv_score)

    def save_location_heatmap(self, output_root: str, sample_id: int) -> str:
        return _save_location_heatmap(
            self.location_grid_best,
            self.location_grid_size,
            output_root,
            sample_id,
        )

    def solve(self, sample_idx: int | None = None):
        patch = torch.rand(
            3,
            self.patch_size,
            self.patch_size,
            device=self.fitness.device,
        )
        location = self._random_location()

        best_patch = patch.clone()
        best_location = location
        best_adv = self._attack_score(best_patch, best_location)
        self._update_location_grid(best_location, best_adv)

        for i in tqdm(range(self.n_iter)):
            current_loss = self._attack_score(patch, location)
            self._update_location_grid(location, current_loss)
            patch = self._update_patch(patch, location)
            if self.optimize_location:
                location = self.next_location(patch, location, current_loss)

            updated_adv = self._attack_score(patch, location)
            self._update_location_grid(location, updated_adv)
            if updated_adv > best_adv:
                best_adv = updated_adv
                best_patch = patch.clone()
                best_location = location

            if self.print_iter and ((i + 1) % self.print_every == 0 or i == 0):
                prefix = f"[Sample {sample_idx}] " if sample_idx is not None else ""
                print(
                    f"{prefix}Iter {i+1}/{self.n_iter} | "
                    f"loss_before={current_loss:.6f} | "
                    f"score_after={updated_adv:.6f} | "
                    f"best={best_adv:.6f} | "
                    f"loc={location}"
                )

            self._log_iteration(updated_adv, location)

        if self.track_best:
            final_patch = best_patch
            final_location = best_location
            final_adv = best_adv
        else:
            final_patch = patch
            final_location = location
            final_adv = self._attack_score(patch, location)
        self._update_location_grid(final_location, final_adv)

        adv_img = self.fitness.apply_patch_to_image(final_patch, final_location)
        psnr_score = self._single_psnr(adv_img)
        return final_patch, final_location, adv_img, final_adv, psnr_score, self.arkive

    def _update_patch(self, patch: torch.Tensor, location: tuple[int, int, int, int]) -> torch.Tensor:
        self.fitness.model.zero_grad(set_to_none=True)

        patch_var = patch.detach().clone().requires_grad_(True)
        adv_objective = self.fitness.evaluate_adv_single_with_grad(
            patch_var,
            location,
        )
        adv_objective.backward()

        grad = patch_var.grad
        if grad is None:
            return patch

        if self.signed_grad:
            grad = torch.sign(grad)

        return torch.clamp(patch_var + self.epsilon * grad, 0.0, 1.0).detach()

    def next_location(self, patch: torch.Tensor, location: tuple[int, int, int, int], current_loss: float) -> tuple[int, int, int, int]:
        directions = ['up', 'down', 'left', 'right']
        if self.optimize_location_type == 'random':
            directions = [random.choice(directions)]

        best_location = location
        best_score = current_loss

        for direction in directions:
            candidate_location = self._move_location(location, direction)
            if candidate_location == location:
                continue

            candidate_score = self._attack_score(patch, candidate_location)
            self._update_location_grid(candidate_location, candidate_score)
            if candidate_score > best_score:
                best_score = candidate_score
                best_location = candidate_location

        return best_location

    def _move_location(self, location: tuple[int, int, int, int], direction: str) -> tuple[int, int, int, int]:
        x_min, _, y_min, _ = location

        if direction == 'up':
            x_min -= self.stride
        elif direction == 'down':
            x_min += self.stride
        elif direction == 'left':
            y_min -= self.stride
        elif direction == 'right':
            y_min += self.stride

        x_min = max(0, min(x_min, self.img_h - self.patch_size))
        y_min = max(0, min(y_min, self.img_w - self.patch_size))
        return (x_min, x_min + self.patch_size, y_min, y_min + self.patch_size)

    def _random_location(self) -> tuple[int, int, int, int]:
        x_min = random.randint(0, self.img_h - self.patch_size)
        y_min = random.randint(0, self.img_w - self.patch_size)
        return (x_min, x_min + self.patch_size, y_min, y_min + self.patch_size)

    def _attack_score(self, patch: torch.Tensor, location: tuple[int, int, int, int]) -> float:
        with torch.no_grad():
            return self.fitness.evaluate_adv_single_with_grad(patch, location).item()

    def _single_psnr(self, adv_img: torch.Tensor) -> float:
        mse = torch.mean((adv_img - self.fitness.img1) ** 2)
        psnr = torch.log10(1 / (mse + 1e-8)) / 10
        return psnr.item()

    def _log_iteration(self, adv_score: float, location: tuple[int, int, int, int]) -> None:
        self.arkive.append({
            "adv_score": adv_score,
            "location": location,
        })



                