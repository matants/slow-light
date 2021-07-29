import time

import numpy as np
import torch

from slow_light_device import SlowLightDevice


class SlowLightDeviceWithPerturbations(SlowLightDevice):
    def __init__(
            self,
            alpha: float = 0.001,
            gamma: float = 100,
            refractive_index: float = 2.,
            N: int = 10,
            batch_size: int = 256,
            weight_of_p_finish_loss: float = 1000,
            p_finish_wanted: float = 0.1,
            L: float = 1.,
            min_d: float = 0.01,
            min_L: float = 0.01,
            L_init: list = None,
            d_init: list = None,
            lr: float = 1e-5,
            sgd_momentum: float = 0,
            is_optimize: bool = True,
            log_dir: str = None,
            log_dir_suffix: str = None,
            seed: int = None,
            use_tensorboard: bool = True,
            device: torch.device = torch.device('cpu'),
            # Arguments just for this version
            num_perturbed_versions: int = 16,
            perturbation_max_size: float = 0.01,
    ):
        self.num_perturbed_versions = num_perturbed_versions
        self.perturbation_max_size = perturbation_max_size
        super().__init__(
            alpha,
            gamma,
            refractive_index,
            N,
            batch_size,
            weight_of_p_finish_loss,
            p_finish_wanted,
            L,
            min_d,
            min_L,
            L_init,
            d_init,
            lr,
            sgd_momentum,
            is_optimize,
            log_dir,
            log_dir_suffix,
            seed,
            use_tensorboard,
            device,
        )

        self.loss_building_time = 0  # for debugging purposes

    def check_params(self):
        max_move_prob = np.exp(-self.gamma * (self.min_d - self.perturbation_max_size))
        max_slab_length = self.L + self.perturbation_max_size - \
                          (self.N + 1) * (self.min_d - self.perturbation_max_size) - \
                          (self.N - 1) * (self.min_L - self.perturbation_max_size)
        max_scatter_prob = self.alpha * max_slab_length
        return super().check_params() and \
               (max_move_prob * 2 + max_scatter_prob <= 1) and \
               (self.perturbation_max_size < self.min_L) and \
               (self.perturbation_max_size < self.min_d)

    def make_perturbed_version(self):
        perturbations_Lt = (torch.rand(self.Lt.shape, device=self.device) * 2 - 1) * self.perturbation_max_size
        perturbations_dt = (torch.rand(self.dt.shape, device=self.device) * 2 - 1) * self.perturbation_max_size
        new_Lt = self.Lt + perturbations_Lt
        new_dt = self.dt + perturbations_dt
        # new_Lt, new_dt = self._clamp_not_inplace(new_Lt, new_dt)
        # ^ I do not want to enforce the clamp on the noised versions, so this line is commented out.
        # check_params makes sure i don't get negative transition probabilities (instead of enforcing
        # d_min), and I disregard the total length of 1 for the noised versions.
        return new_Lt, new_dt

    def build_loss_mean(self):
        init_time = time.time()
        loss = torch.zeros(self.batch_size, device=self.device)
        for _ in range(self.num_perturbed_versions):
            new_Lt, new_dt = self.make_perturbed_version()
            new_chain, _, _, _, _, _, _, _ = self.initialize_P(new_Lt, new_dt)
            loss += self._build_loss(new_Lt, new_chain)
        loss /= self.num_perturbed_versions
        self.loss_building_time += time.time() - init_time
        return loss

    def build_loss(self):
        return self.build_loss_mean()


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # device = torch.device('cpu')
    # sld = SlowLightDeviceWithPerturbations(
    #     seed=73,
    #     batch_size=1024,
    #     perturbation_max_size=0.003,
    #     device=device,
    #     N=5,
    #     log_dir_suffix=f"best_perturbed_results_N_5_for_permutations_new_clamp")
    # sld.train(50000, 100, draw_best_device=False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    ##############
    # Batch run:
    ##############
    init_seed = 74
    redos = 4
    for i in range(redos):
        seed = init_seed + i
        for finish_prob in np.arange(0, 1, 0.05):
            sld = SlowLightDeviceWithPerturbations(seed=seed,
                                                   batch_size=256,
                                                   device=device,
                                                   N=10,
                                                   min_L=0.01,
                                                   min_d=0.01,
                                                   perturbation_max_size=0.003,
                                                   gamma=100,
                                                   p_finish_wanted=finish_prob,
                                                   log_dir_suffix=f"perturbed_new_clamp_seed_{seed}__pf"
                                                                  f"_{finish_prob:.2f}")
            sld.train(20000, 100, draw_best_device=False)

        for N in range(1, 20 + 1):
            sld = SlowLightDeviceWithPerturbations(seed=seed,
                                                   batch_size=256,
                                                   device=device,
                                                   N=N,
                                                   min_L=0.01,
                                                   min_d=0.01,
                                                   perturbation_max_size=0.003,
                                                   gamma=100,
                                                   p_finish_wanted=0.1,
                                                   log_dir_suffix=f"perturbed_new_clamp_seed_{seed}__N"
                                                                  f"_{N}")
            sld.train(20000, 100, draw_best_device=False)
