import random

import torch


class BiometricIndividual:
    def __init__(
        self,
        patch_size: int,
        img_shape: tuple[int, int],
        prob_mutate_patch: float,
        prob_mutate_location: float,
        valid_locations: list[tuple[int, int]],
        mutate_mode: str = "single_rectangle",
        seed_patch_source=None,
        use_img2_seed_init: bool = False,
        img2_seed_ratio: float = 0.5,
        img2_seed_blend: float = 0.7,
    ) -> None:
        self.patch_size = int(patch_size)
        self.img_shape = img_shape
        self.prob_mutate_patch = float(prob_mutate_patch)
        self.prob_mutate_location = float(prob_mutate_location)
        self.valid_locations = valid_locations
        self.mutate_mode = mutate_mode
        self.seed_patch_source = seed_patch_source
        self.use_img2_seed_init = bool(use_img2_seed_init)
        self.img2_seed_ratio = float(max(0.0, min(1.0, img2_seed_ratio)))
        self.img2_seed_blend = float(max(0.0, min(1.0, img2_seed_blend)))
        self.rank = None
        self.crowding = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.valid_locations:
            raise ValueError("valid_locations is empty. Cannot sample biometric-only locations.")

        self._random_location()
        self._random_patch()

        self.psnr_score = None
        self.adv_score = None
        self.saliency_score = None

    def _spawn(self) -> "BiometricIndividual":
        return BiometricIndividual(
            self.patch_size,
            self.img_shape,
            self.prob_mutate_patch,
            self.prob_mutate_location,
            valid_locations=self.valid_locations,
            mutate_mode=self.mutate_mode,
            seed_patch_source=self.seed_patch_source,
            use_img2_seed_init=self.use_img2_seed_init,
            img2_seed_ratio=self.img2_seed_ratio,
            img2_seed_blend=self.img2_seed_blend,
        )

    def _location_from_start(self, x_min: int, y_min: int) -> tuple[int, int, int, int]:
        x_min = max(0, min(int(x_min), self.img_shape[0] - self.patch_size))
        y_min = max(0, min(int(y_min), self.img_shape[1] - self.patch_size))
        return (x_min, x_min + self.patch_size, y_min, y_min + self.patch_size)

    def _random_location(self) -> None:
        x_min, y_min = random.choice(self.valid_locations)
        self.location = self._location_from_start(x_min, y_min)

    def _random_patch(self) -> None:
        self.patch = torch.rand(3, self.patch_size, self.patch_size, device=self.device)

        if not self.use_img2_seed_init:
            return
        if self.seed_patch_source is None:
            return
        if random.random() >= self.img2_seed_ratio:
            return

        x_min, x_max, y_min, y_max = self.location
        crop = self.seed_patch_source[:, x_min:x_max, y_min:y_max].to(self.device)
        if crop.shape[-2:] != (self.patch_size, self.patch_size):
            return

        self.patch = torch.clamp(
            self.img2_seed_blend * crop + (1.0 - self.img2_seed_blend) * self.patch,
            0.0,
            1.0,
        )

    def mutate(self) -> None:
        if random.random() < self.prob_mutate_patch:
            self._add_rectangle()

        if random.random() < self.prob_mutate_location:
            self._random_location()

    def mutate_location(self) -> None:
        self._random_location()

    def mutate_content(self) -> None:
        if random.random() < self.prob_mutate_patch:
            self._add_rectangle()

    def _add_rectangle(self) -> None:
        if self.mutate_mode == "single_rectangle":
            x_min = random.randint(0, self.patch_size - 1)
            y_min = random.randint(0, self.patch_size - 1)
            width = random.randint(2, 5)
            color = torch.rand(3, device=self.device)
            self.patch[:, x_min : x_min + width, y_min : y_min + width] = color.unsqueeze(1).unsqueeze(2)
        elif self.mutate_mode == "multiple_rectangles":
            num_rectangles = random.randint(1, 10)
            for _ in range(num_rectangles):
                x_min = random.randint(0, self.patch_size - 1)
                y_min = random.randint(0, self.patch_size - 1)
                width = random.randint(2, 5)
                color = torch.rand(3, device=self.device)
                self.patch[:, x_min : x_min + width, y_min : y_min + width] = color.unsqueeze(1).unsqueeze(2)

    def crossover_UX(self, parent2: "BiometricIndividual") -> tuple["BiometricIndividual", "BiometricIndividual"]:
        offstring1_patch = self.patch.clone()
        offstring2_patch = parent2.patch.clone()

        offstring1 = self._spawn()
        offstring2 = self._spawn()

        cut_point = random.randint(0, self.patch_size)
        offstring1_patch[:, :cut_point, :] = parent2.patch[:, :cut_point, :]
        offstring2_patch[:, :cut_point, :] = self.patch[:, :cut_point, :]

        if random.random() < 0.05:
            offstring1.location = parent2.location
            offstring2.location = self.location
        else:
            offstring1.location = self.location
            offstring2.location = parent2.location

        offstring1.patch = offstring1_patch
        offstring2.patch = offstring2_patch
        return offstring1, offstring2

    def crossover_blended(self, parent2: "BiometricIndividual", alpha: float = 0.5) -> tuple["BiometricIndividual", "BiometricIndividual"]:
        offstring1_patch = alpha * self.patch + (1 - alpha) * parent2.patch
        offstring2_patch = alpha * parent2.patch + (1 - alpha) * self.patch

        offstring1 = self._spawn()
        offstring2 = self._spawn()

        if random.random() < 0.05:
            offstring1.location = parent2.location
            offstring2.location = self.location
        else:
            offstring1.location = self.location
            offstring2.location = parent2.location

        offstring1.patch = offstring1_patch
        offstring2.patch = offstring2_patch
        return offstring1, offstring2
