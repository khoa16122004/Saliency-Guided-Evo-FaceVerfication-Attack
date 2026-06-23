from individual_biometric import BiometricIndividual


class BiometricPopulation:
    def __init__(
        self,
        pop_size: int,
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
        self.pop_size = int(pop_size)
        self.patch_size = int(patch_size)
        self.img_shape = img_shape
        self.prob_mutate_patch = float(prob_mutate_patch)
        self.prob_mutate_location = float(prob_mutate_location)
        self.valid_locations = valid_locations
        self.mutate_mode = mutate_mode
        self.seed_patch_source = seed_patch_source
        self.use_img2_seed_init = bool(use_img2_seed_init)
        self.img2_seed_ratio = float(img2_seed_ratio)
        self.img2_seed_blend = float(img2_seed_blend)

        self._create_population(
            patch_size=self.patch_size,
            img_shape=self.img_shape,
            prob_mutate_patch=self.prob_mutate_patch,
            prob_mutate_location=self.prob_mutate_location,
            valid_locations=self.valid_locations,
            mutate_mode=self.mutate_mode,
            seed_patch_source=self.seed_patch_source,
            use_img2_seed_init=self.use_img2_seed_init,
            img2_seed_ratio=self.img2_seed_ratio,
            img2_seed_blend=self.img2_seed_blend,
        )

    def _create_population(
        self,
        patch_size: int,
        img_shape: tuple[int, int],
        prob_mutate_patch: float,
        prob_mutate_location: float,
        valid_locations: list[tuple[int, int]],
        mutate_mode: str,
        seed_patch_source,
        use_img2_seed_init: bool,
        img2_seed_ratio: float,
        img2_seed_blend: float,
    ) -> None:
        self.P = [
            BiometricIndividual(
                patch_size,
                img_shape,
                prob_mutate_patch,
                prob_mutate_location,
                valid_locations=valid_locations,
                mutate_mode=mutate_mode,
                seed_patch_source=seed_patch_source,
                use_img2_seed_init=use_img2_seed_init,
                img2_seed_ratio=img2_seed_ratio,
                img2_seed_blend=img2_seed_blend,
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
