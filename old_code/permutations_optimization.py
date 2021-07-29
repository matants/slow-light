import itertools

import numpy as np
import torch

from analyze_results import extract_best_lengths, extract_final_lengths
from slow_light_device import SlowLightDevice
from utils import permute_array, seed_all


class SlowLightDeviceWithPermutations(SlowLightDevice):
    def __init__(
            self,
            alpha: float = 0.001,
            gamma: float = 100,
            refractive_index: float = 2.,
            N: int = 10,
            batch_size: int = 1,
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
    ):
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

    def permute_L(self, is_with_perturbations: bool = False, num_perturbed_versions: int = 16,
                  perturbation_max_size: float = 0.003):
        init_Lt = self.Lt.detach()
        init_dt = self.dt.detach()
        saved_arrays = {"Lt": init_Lt.cpu().numpy(),
                        "dt": init_dt.cpu().numpy(),
                        }

        if is_with_perturbations:
            seed_all(self.seed)
            perturbations_Lt, perturbations_dt = self.make_perturbations(self.N, num_perturbed_versions,
                                                                         perturbation_max_size)
            saved_arrays["perturbations_Lt"] = np.array([i.cpu().numpy() for i in perturbations_Lt])
            saved_arrays["perturbations_dt"] = np.array([i.cpu().numpy() for i in perturbations_dt])

        np.savez(self.log_dir + "/saved_init_lengths.npz", **saved_arrays)

        for perm in itertools.permutations(range(self.N)):
            new_Lt = permute_array(init_Lt, perm)
            if not is_with_perturbations:
                new_chain, _, _, _, _, _, _, _ = self.initialize_P(new_Lt, init_dt)
                loss = self._build_loss(new_Lt, new_chain).detach().cpu().numpy()
            else:
                loss = self._build_loss_perturbations(new_Lt, init_dt, perturbations_Lt, perturbations_dt)
            with open(self.log_dir + f'/permute_L_{self.N}_results.csv', 'a') as output_csv:  # this will append the
                # result to the file.
                output_csv.writelines(str(perm) + ': ' + str(loss.item()) + '\n')  # write it per line

    def permute_d(self, is_with_perturbations: bool = False, num_perturbed_versions: int = 16,
                  perturbation_max_size: float = 0.003):
        init_Lt = self.Lt.detach()
        init_dt = self.dt.detach()
        saved_arrays = {"Lt": init_Lt.cpu().numpy(),
                        "dt": init_dt.cpu().numpy(),
                        }

        if is_with_perturbations:
            seed_all(self.seed)
            perturbations_Lt, perturbations_dt = self.make_perturbations(self.N, num_perturbed_versions,
                                                                         perturbation_max_size)
            saved_arrays["perturbations_Lt"] = np.array([i.cpu().numpy() for i in perturbations_Lt])
            saved_arrays["perturbations_dt"] = np.array([i.cpu().numpy() for i in perturbations_dt])

        np.savez(self.log_dir + "/saved_init_lengths.npz", **saved_arrays)

        for perm in itertools.permutations(range(self.N + 1)):
            new_dt = permute_array(init_dt, perm)
            if not is_with_perturbations:
                new_chain, _, _, _, _, _, _, _ = self.initialize_P(init_Lt, new_dt)
                loss = self._build_loss(init_Lt, new_chain).detach().cpu().numpy()
            else:
                loss = self._build_loss_perturbations(init_Lt, new_dt, perturbations_Lt, perturbations_dt)
            with open(self.log_dir + f'/permute_d_{self.N}_results.csv', 'a') as output_csv:  # this will append the
                # result to the file.
                output_csv.writelines(str(perm) + ': ' + str(loss.item()) + '\n')  # write it per line

    def make_perturbations(self, N: int, num_perturbed_versions: int = 16, perturbation_max_size: float = 0.003):
        perturbations_Lt = [(torch.rand(N, device=self.device) * 2 - 1) * perturbation_max_size for
                            _ in range(num_perturbed_versions)]
        perturbations_dt = [(torch.rand(N + 1, device=self.device) * 2 - 1) * perturbation_max_size for
                            _ in range(num_perturbed_versions)]
        return perturbations_Lt, perturbations_dt

    def _build_loss_perturbations(self, Lt, dt, perturbations_Lt, perturbations_dt):
        loss = torch.zeros(self.batch_size, device=self.device)
        for i in range(len(perturbations_Lt)):
            new_Lt = Lt + perturbations_Lt[i]
            new_dt = dt + perturbations_dt[i]
            new_chain, _, _, _, _, _, _, _ = self.initialize_P(new_Lt, new_dt)
            loss += self._build_loss(new_Lt, new_chain)
        loss /= len(perturbations_Lt)
        return loss


def init_ascending_Ls(N, d_val=0.05, min_L=0.05):
    d_init = np.ones(N + 1) * d_val
    sum_Ls = 1 - d_val * (N + 1)
    max_L = sum_Ls / N * 2 - min_L
    L_init = np.linspace(min_L, max_L, N)
    assert np.isclose(L_init.sum() + d_init.sum(), 1.)
    return L_init, d_init


def init_ascending_ds(N, L_val=0.05, min_d=0.05):
    L_init = np.ones(N) * L_val
    sum_ds = 1 - L_val * N
    max_d = sum_ds / (N + 1) * 2 - min_d
    d_init = np.linspace(min_d, max_d, N + 1)
    assert np.isclose(L_init.sum() + d_init.sum(), 1.)
    return L_init, d_init


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    ##########################
    # Make permutations sweep data:
    ##########################
    # for N in range(2, 7 + 1):
    #     L_init, d_init = init_ascending_Ls(N, d_val=0.05, min_L=0.01)
    #     print(f"L permutations: L_init = {L_init}, d_init = {d_init}")
    #     sld = SlowLightDeviceWithPermutations(
    #         seed=73,
    #         batch_size=1,
    #         device=device,
    #         is_optimize=True,
    #         N=N,
    #         d_init=d_init,
    #         L_init=L_init,
    #         log_dir_suffix=f"Ls_permutations__N_{N}")
    #     sld.permute_L()
    #
    #     L_init, d_init = init_ascending_ds(N, L_val=0.05, min_d=0.01)
    #     print(f"d permutations: L_init = {L_init}, d_init = {d_init}")
    #     sld = SlowLightDeviceWithPermutations(
    #         seed=73,
    #         batch_size=1,
    #         device=device,
    #         is_optimize=True,
    #         N=N,
    #         d_init=d_init,
    #         L_init=L_init,
    #         log_dir_suffix=f"ds_permutations__N_{N}")
    #     sld.permute_d()

    # Permute "ideal"
    N = 5
    # sld = SlowLightDeviceWithPerturbations(
    #     seed=73,
    #     batch_size=1024,
    #     perturbation_max_size=0.003,
    #     device=device,
    #     N=N,
    #     log_dir_suffix=f"best_perturbed_results_N_5_for_permutations")
    # sld.train(50000, 100, draw_best_device=False)

    L_init, d_init = extract_best_lengths(
        'C:\\Users\\matan\\Documents\\slow_light_markov\\runs'
        '\\run_2021_05_22__19_16__best_perturbed_results_N_5_for_permutations_new_clamp\\results.npz')
    print(f"L permutations: L_init = {L_init}, d_init = {d_init}")
    sld = SlowLightDeviceWithPermutations(
        seed=73,
        batch_size=1,
        device=device,
        is_optimize=True,
        N=N,
        d_init=d_init,
        L_init=L_init,
        log_dir_suffix=f"perturbed_initial_strong_Ls_permutations__N_{N}")
    sld.permute_L(is_with_perturbations=True, num_perturbed_versions=256)
    print(f"d permutations: L_init = {L_init}, d_init = {d_init}")
    sld = SlowLightDeviceWithPermutations(
        seed=73,
        batch_size=1,
        device=device,
        is_optimize=True,
        N=N,
        d_init=d_init,
        L_init=L_init,
        log_dir_suffix=f"perturbed_initial_strong_ds_permutations__N_{N}")
    sld.permute_d(is_with_perturbations=True, num_perturbed_versions=256)

    L_init, d_init = extract_final_lengths(
        'C:\\Users\\matan\\Documents\\slow_light_markov\\runs'
        '\\run_2021_05_22__19_16__best_perturbed_results_N_5_for_permutations_new_clamp\\results.npz')
    print(f"L permutations: L_init = {L_init}, d_init = {d_init}")
    sld = SlowLightDeviceWithPermutations(
        seed=73,
        batch_size=1,
        device=device,
        is_optimize=True,
        N=N,
        d_init=d_init,
        L_init=L_init,
        log_dir_suffix=f"perturbed_initial_semistrong_end_of_sgd_Ls_permutations__N_{N}")
    sld.permute_L(is_with_perturbations=True, num_perturbed_versions=256)
    print(f"d permutations: L_init = {L_init}, d_init = {d_init}")
    sld = SlowLightDeviceWithPermutations(
        seed=73,
        batch_size=1,
        device=device,
        is_optimize=True,
        N=N,
        d_init=d_init,
        L_init=L_init,
        log_dir_suffix=f"perturbed_initial_semistrong_end_of_sgd_ds_permutations__N_{N}")
    sld.permute_d(is_with_perturbations=True, num_perturbed_versions=256)
