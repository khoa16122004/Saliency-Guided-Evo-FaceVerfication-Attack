import random
import torch
from torchvision.utils import save_image

class Individual:
    def __init__(self, patch_size: int, img_shape: tuple[int, int], prob_mutate_patch: float, prob_mutate_location: float, guidance: dict | None = None, use_saliency_guidance: bool = False, saliency_noise_scale: float = 0.15, mutate_mode: str = "single_rectangle") -> None:
        """
        Initialize an individual with a random patch and location.
        """
        self.patch_size = patch_size 
        self.img_shape = img_shape
        self.prob_mutate_patch = prob_mutate_patch
        self.prob_mutate_location = prob_mutate_location
        self.guidance = guidance or {}
        self.use_saliency_guidance = use_saliency_guidance
        self.saliency_noise_scale = saliency_noise_scale
        self.rank = None
        self.crowding = None
        self.device = self._resolve_device()
        
        
        self.mutate_mode = mutate_mode
        self._random_location()
        self._random_patch()
        
        self.psnr_score = None
        self.adv_score = None
        self.saliency_score = None

    def _resolve_device(self) -> torch.device:
        saliency_map = self.guidance.get("saliency_map")
        if saliency_map is not None:
            return saliency_map.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _spawn(self) -> 'Individual':
        return Individual(
            self.patch_size,
            self.img_shape,
            self.prob_mutate_patch,
            self.prob_mutate_location,
            guidance=self.guidance,
            use_saliency_guidance=self.use_saliency_guidance,
            saliency_noise_scale=self.saliency_noise_scale,
        )

    def _location_from_start(self, x_min: int, y_min: int) -> tuple[int, int, int, int]:
        x_min = max(0, min(x_min, self.img_shape[0] - self.patch_size))
        y_min = max(0, min(y_min, self.img_shape[1] - self.patch_size))
        return (x_min, x_min + self.patch_size, y_min, y_min + self.patch_size)

    def _guided_random_location(self) -> tuple[int, int, int, int]:
        location_probs = self.guidance.get("location_probs")
        if location_probs is None:
            x_min = random.randint(0, self.img_shape[0] - self.patch_size)
            y_min = random.randint(0, self.img_shape[1] - self.patch_size)
            return self._location_from_start(x_min, y_min)

        flat_idx = torch.multinomial(location_probs, 1).item()
        width = self.img_shape[1] - self.patch_size + 1
        x_min = flat_idx // width
        y_min = flat_idx % width
        return self._location_from_start(x_min, y_min)

    def _current_patch_saliency(self) -> float:
        saliency_map = self.guidance.get("saliency_map")
        if saliency_map is None:
            return 0.0
        x_min, x_max, y_min, y_max = self.location
        return saliency_map[x_min:x_max, y_min:y_max].mean().item()

    def _guided_location_mutation(self) -> None:
        current_saliency = self._current_patch_saliency()
        if random.random() < max(0.0, 1.0 - current_saliency):
            self.location = self._guided_random_location()
            return

        x_min, _, y_min, _ = self.location
        sigma = max(1.0, self.patch_size / 4)
        new_x = int(round(random.gauss(x_min, sigma)))
        new_y = int(round(random.gauss(y_min, sigma)))
        self.location = self._location_from_start(new_x, new_y)

    def _guided_patch_mutation(self) -> None:
        current_saliency = self._current_patch_saliency()
        std = self.saliency_noise_scale * max(0.1, current_saliency)
        noise = torch.randn_like(self.patch) * std
        self.patch = torch.clamp(self.patch + noise, 0.0, 1.0)

    def _random_location(self) -> None:
        """
        Generates a random location (x_min, x_max, y_min, y_max) within image bounds.
        """
        if self.use_saliency_guidance:
            self.location = self._guided_random_location()
            return

        x_min = random.randint(0, self.img_shape[0] - self.patch_size)
        y_min = random.randint(0, self.img_shape[1] - self.patch_size)
        self.location = self._location_from_start(x_min, y_min)
    
    def _random_patch(self) -> None:
        """
        Generates a random patch of shape (3, patch_size, patch_size).
        """
        self.patch = torch.rand(3, self.patch_size, self.patch_size, device=self.device)
    
    def mutate(self) -> None:
        """
        Apply a mutation to the individual: add a rectangle or circle shape to the patch.
        """
        if random.random() < self.prob_mutate_patch:  # Add rectangle
            if self.use_saliency_guidance:
                self._guided_patch_mutation()
            else:
                self._add_rectangle()
        
        if random.random() < self.prob_mutate_location:
            if self.use_saliency_guidance:
                self._guided_location_mutation()
            else:
                self._random_location()
 
    def mutate_location(self) -> None:
        """
        Apply a mutation location to the individual
        """
        if self.use_saliency_guidance:
            self._guided_location_mutation()
        else:
            self._random_location()
    
    def mutate_content(self) -> None:
        """
        Apply a mutation to the individual: add a rectangle or circle shape to the patch.
        """
        if random.random() < self.prob_mutate_patch:  
            if self.use_saliency_guidance:
                self._guided_patch_mutation()
            else:
                self._add_rectangle()
    
    def _add_rectangle(self) -> None:
        """
        Add a rectangle to the patch.
        """
        print("Adding rectangle mutation...")
        print(self.mutate_mode)
        raise
        
        if self.mutate_mode == "single_rectangle":     
            x_min = random.randint(0, self.patch_size - 1)
            y_min = random.randint(0, self.patch_size - 1)
            width = random.randint(2, 5)
            color = torch.rand(3, device=self.device)  # Random RGB color

            self.patch[:, x_min: x_min + width, y_min: y_min + width] = color.unsqueeze(1).unsqueeze(2)
        
        elif self.mutate_mode == "multiple_rectangles":
            print("Mutating with multiple rectangles...")
            n_rects = torch.randint(
                low=10,
                high=50,
                size=(1,),
                device=self.device
            ).item()

            # kích thước rectangle
            widths = torch.randint(
                low=2,
                high=max(3, self.patch_size // 4),
                size=(n_rects,),
                device=self.device
            )

            heights = torch.randint(
                low=2,
                high=max(3, self.patch_size // 4),
                size=(n_rects,),
                device=self.device
            )

            # màu RGB
            colors = torch.rand(
                n_rects,
                3,
                device=self.device
            )

            # độ mạnh khi cộng vào
            alphas = 0.05 + 0.45 * torch.rand(
                n_rects,
                device=self.device
            )

            for i in range(n_rects):

                w = widths[i].item()
                h = heights[i].item()

                x = torch.randint(
                    0,
                    self.patch_size - w + 1,
                    (1,),
                    device=self.device
                ).item()

                y = torch.randint(
                    0,
                    self.patch_size - h + 1,
                    (1,),
                    device=self.device
                ).item()

                color = colors[i].view(3, 1, 1)
                alpha = alphas[i]

                self.patch[:, x:x+w, y:y+h] += alpha * color

            self.patch.clamp_(0.0, 1.0)            


        
    
    

    def crossover_UX(self, parent2: 'Individual') -> tuple['Individual', 'Individual']:
        """
        Perform crossover with another individual to produce two offspring.

        :param parent2: Another Individual object.
        :return: Two new Individual objects.
        """
        offstring1_patch = self.patch.clone()
        offstring2_patch = parent2.patch.clone()

        offstring1 = self._spawn()
        offstring2 = self._spawn()

        cut_point = random.randint(0, self.patch_size)
        offstring1_patch[:, :cut_point, :] = parent2.patch[:, :cut_point, :]
        offstring2_patch[:, :cut_point, :] = self.patch[:, :cut_point, :]

        if self.use_saliency_guidance:
            if random.random() < 0.5:
                offstring1.location = self.location
                offstring2.location = parent2.location
            else:
                offstring1.location = parent2.location
                offstring2.location = self.location
        elif random.random() < 0.05:
            offstring1.location = parent2.location
            offstring2.location = self.location 
      
        offstring1.patch = offstring1_patch
        offstring2.patch = offstring2_patch

        return offstring1, offstring2
    
    def crossover_blended(self, parent2: 'Individual', alpha=0.5) -> tuple['Individual', 'Individual']:
        """
        using crossover_blended
        o1 = alpha * p1 + (1 - alpha) * p2
        o2 = alpha *p2 + (1 - alpha) * p1
        """    
        offstring1_patch = alpha * self.patch + (1 - alpha) * parent2.patch
        offstring2_patch = alpha * parent2.patch + (1 - alpha) * self.patch

        offstring1 = self._spawn()
        offstring2 = self._spawn()

        if self.use_saliency_guidance:
            if random.random() < 0.5:
                offstring1.location = self.location
                offstring2.location = parent2.location
            else:
                offstring1.location = parent2.location
                offstring2.location = self.location
        elif random.random() < 0.05:
            offstring1.location = parent2.location
            offstring2.location = self.location
        
        offstring1.patch = offstring1_patch
        offstring2.patch = offstring2_patch

        return offstring1, offstring2
        
        
