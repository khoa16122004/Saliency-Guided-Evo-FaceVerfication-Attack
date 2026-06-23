import torch
from population import Population
from individual import Individual
import torch.nn.functional as F
from torch import nn

class Fitness:
    
    def __init__(self, patch_size: int, img1:torch.Tensor, img2: torch.Tensor, model: nn.Module, label: int, recons_w: float, attack_w: float, fitness_type: str, saliency_w: float = 0.0, use_saliency_guidance: bool = False) -> None:
        self.device = next(model.parameters()).device
        self.img1 = img1.to(self.device)
        self.img2 = img2.to(self.device)
        self.model = model.eval()
        self.patch_size = patch_size
        
        self.attack_w = attack_w
        self.recons_w = recons_w
        # Keep signature compatibility, but disable saliency-related effects.
        self.saliency_w = 0.0
        self.label = label
        self.fitness_type = fitness_type
        self.use_saliency_guidance = False
        self.enable_saliency = False
        self.max_psnr, self.min_psnr = None, None
        self.max_adv, self.max_adv = None, None
        with torch.no_grad():
            self.img2_feature = model(self.img2.unsqueeze(0)).detach()
        self.saliency_map = None
        self.location_scores = None
        self.location_probs = None

    def apply_patch_to_image(self, patch: torch.Tensor, location: tuple[int, int, int, int]):
        img_copy = self.img1.clone()
        x_min, x_max, y_min, y_max = location
        img_copy[:, x_min : x_max, y_min : y_max] = patch
        return img_copy

    def attack_objective(self, sims: torch.Tensor) -> torch.Tensor:
        """
        Attack objective used across algorithms. Higher is better for attack success.
        """
        return (1 - self.label) * (0.5 - sims) + self.label * (sims - 0.5)

    def evaluate_adv_single_with_grad(self, patch: torch.Tensor, location: tuple[int, int, int, int]) -> torch.Tensor:
        """
        Differentiable attack objective for a single patch/location pair.
        """
        adv_img = self.apply_patch_to_image(patch, location).unsqueeze(0)
        adv_features = self.model(adv_img)
        sims = F.cosine_similarity(adv_features, self.img2_feature, dim=1)
        return self.attack_objective(sims).squeeze(0)

    def evaluate_adv_single(self, patch: torch.Tensor, location: tuple[int, int, int, int]) -> torch.Tensor:
        """
        Non-differentiable (no-grad) attack objective for a single patch/location pair.
        """
        with torch.no_grad():
            adv_img = self.apply_patch_to_image(patch, location).unsqueeze(0)
            adv_features = self.model(adv_img)
            sims = F.cosine_similarity(adv_features, self.img2_feature, dim=1)
            return self.attack_objective(sims).squeeze(0)

    def _compute_saliency_map(self) -> torch.Tensor:
        guided_img = self.img1.detach().clone().unsqueeze(0).requires_grad_(True)
        similarity = F.cosine_similarity(self.model(guided_img), self.img2_feature, dim=1).mean()
        saliency = torch.autograd.grad(similarity, guided_img, retain_graph=False, create_graph=False)[0]
        saliency = saliency.abs().mean(dim=1).squeeze(0)
        saliency = saliency - saliency.min()
        saliency = saliency / (saliency.max() + 1e-8)
        return saliency.detach()

    def _build_location_distribution(self) -> tuple[torch.Tensor, torch.Tensor]:
        saliency = self.saliency_map.unsqueeze(0).unsqueeze(0)
        scores = F.avg_pool2d(saliency, kernel_size=self.patch_size, stride=1).squeeze(0).squeeze(0)
        flat_scores = scores.reshape(-1)
        probs = flat_scores / flat_scores.sum().clamp_min(1e-8)
        return scores.detach(), probs.detach()

    def get_guidance(self) -> dict | None:
        return None

    def get_location_saliency(self, location: tuple[int, int, int, int]) -> torch.Tensor:
        if self.saliency_map is None:
            return torch.tensor(0.0, device=self.device)
        x_min, x_max, y_min, y_max = location
        return self.saliency_map[x_min:x_max, y_min:y_max].mean()
    
        
    def evaluate_adv(self, P: list['Individual']):
        adv_imgs = torch.stack([self.apply_patch_to_image(ind.patch, ind.location) for ind in P])
        
        with torch.no_grad():
            adv_batch = adv_imgs.to(self.device)
            adv_features = self.model(adv_batch)
            sims = F.cosine_similarity(adv_features, self.img2_feature, dim=1)
            adv_scores = self.attack_objective(sims)
           
           # if self.fitness_type == "adaptive":
            #    adv_scores = torch.where(adv_scores > 0, torch.tensor(0.0, device=adv_scores.device), adv_scores)
            
            return adv_scores
            
    def evaluate_psnr(self, P: list['Individual']) -> torch.Tensor:
        adv_imgs = torch.stack([self.apply_patch_to_image(ind.patch, ind.location) for ind in P])
        mse = F.mse_loss(adv_imgs, self.img1.expand_as(adv_imgs), reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1) 
        psnr_scores = torch.log10(1 / (mse + 1e-8))
        
        return psnr_scores / 10

    def evaluate_saliency(self, P: list['Individual']) -> torch.Tensor:
        if self.saliency_map is None:
            return torch.zeros(len(P), device=self.device)
        return torch.stack([self.get_location_saliency(ind.location) for ind in P])
    
    def update_min_max(self, adv_scores: torch.Tensor, psnr_scores:torch.Tensor) -> None: 
        self.min_psnr = torch.min(psnr_scores.min(), self.min_psnr)
        self.max_psnr = torch.max(psnr_scores.max(), self.max_psnr)
        self.min_adv = torch.min(adv_scores.min(), self.min_adv)
        self.max_adv = torch.max(psnr_scores.min(), self.min_adv)
 

    def benchmark(self, P: list['Individual']) -> torch.Tensor:
        real_adv_scores = self.evaluate_adv(P)
        clip_adv_scores = real_adv_scores.clone()
        real_psnr_scores = self.evaluate_psnr(P)
        saliency_scores = self.evaluate_saliency(P)
        
        for i in range(len(P)):
            P[i].adv_score = real_adv_scores[i]
            P[i].psnr_score = real_psnr_scores[i]
            P[i].saliency_score = saliency_scores[i]
        #if self.fitness_type == "normalize":
            # normalize each score
            # self.update_min_max(adv_scores, psnr_scores)
            # adv_scores_normalize = (adv_scores - self.min_adv) / (self.max_adv - self.min_adv)
            # psnr_scores_normalize = (adv_scores - self.min_psnr) / (self.max_psnr - self.min_psnr)
            # return adv_scores_normalize + psnr_scores_normalize, adv_scores, psnr_scores
        if self.fitness_type == "adaptive":
            clip_adv_scores = torch.where(real_adv_scores > 0, torch.tensor(0.0, device=real_adv_scores.device), real_adv_scores)
        combined = clip_adv_scores * self.attack_w + real_psnr_scores * self.recons_w
        return combined, real_adv_scores, real_psnr_scores, saliency_scores
    
    def benchmark_not_adaptive(self, P: list['Individual']) -> torch.Tensor:
        adv_scores = self.evaluate_adv(P)
        psnr_scores = self.evaluate_psnr(P)
        saliency_scores = self.evaluate_saliency(P)
        
        for i in range(len(P)):
            P[i].adv_score = adv_scores[i]
            P[i].psnr_score = psnr_scores[i]
            P[i].saliency_score = saliency_scores[i]
        #if self.fitness_type == "normalize":
            # normalize each score
            # self.update_min_max(adv_scores, psnr_scores)
            # adv_scores_normalize = (adv_scores - self.min_adv) / (self.max_adv - self.min_adv)
            # psnr_scores_normalize = (adv_scores - self.min_psnr) / (self.max_psnr - self.min_psnr)
            # return adv_scores_normalize + psnr_scores_normalize, adv_scores, psnr_scores


        combined = adv_scores * self.attack_w + psnr_scores * self.recons_w
        return combined, adv_scores, psnr_scores, saliency_scores        
        
