from individual import Individual
class Population:
    
    def __init__(self, pop_size: int, 
                 patch_size: int, img_shape: tuple[int, int], 
                 prob_mutate_patch: float, prob_mutate_location: float,
                 guidance: dict | None = None,
                 use_saliency_guidance: bool = False,
                 saliency_noise_scale: float = 0.15,
                 mutate_mode: str = "single_rectangle",
                 target_patch_source=None) -> None:
        
        self.pop_size = pop_size
        self.patch_size = patch_size
        self.img_shape = img_shape
        self.prob_mutate_patch = prob_mutate_patch
        self.prob_mutate_location = prob_mutate_location
        self.guidance = guidance
        self.use_saliency_guidance = use_saliency_guidance
        self.saliency_noise_scale = saliency_noise_scale
        self.mutate_mode = mutate_mode
        self.target_patch_source = target_patch_source
        
        self._create_population(patch_size, img_shape, prob_mutate_patch, prob_mutate_location, mutate_mode=self.mutate_mode)
        
    def _create_population(self, patch_size: int, img_shape: tuple[int, int], prob_mutate_patch: float, prob_mutate_location: float, mutate_mode: str) -> None:
        self.P = [
            Individual(
                patch_size,
                img_shape,
                prob_mutate_patch,
                prob_mutate_location,
                guidance=self.guidance,
                use_saliency_guidance=self.use_saliency_guidance,
                saliency_noise_scale=self.saliency_noise_scale,
                mutate_mode=mutate_mode,
                target_patch_source=self.target_patch_source,
            )
            for _ in range(self.pop_size)
        ]
    
    def get_params(self):
        return [
            self.pop_size,
            self.patch_size,
            self.img_shape,
            self.prob_mutate_patch,
            self.prob_mutate_location,
        ]
