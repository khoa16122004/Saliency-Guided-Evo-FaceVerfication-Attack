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
    ) -> None:
        self.pop_size = int(pop_size)
        self.patch_size = int(patch_size)
        self.img_shape = img_shape
        self.prob_mutate_patch = float(prob_mutate_patch)
        self.prob_mutate_location = float(prob_mutate_location)
        self.valid_locations = valid_locations
        self.mutate_mode = mutate_mode

        self._create_population(
            patch_size=self.patch_size,
            img_shape=self.img_shape,
            prob_mutate_patch=self.prob_mutate_patch,
            prob_mutate_location=self.prob_mutate_location,
            valid_locations=self.valid_locations,
            mutate_mode=self.mutate_mode,
        )

    def _create_population(
        self,
        patch_size: int,
        img_shape: tuple[int, int],
        prob_mutate_patch: float,
        prob_mutate_location: float,
        valid_locations: list[tuple[int, int]],
        mutate_mode: str,
    ) -> None:
        self.P = [
            BiometricIndividual(
                patch_size,
                img_shape,
                prob_mutate_patch,
                prob_mutate_location,
                valid_locations=valid_locations,
                mutate_mode=mutate_mode,
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
