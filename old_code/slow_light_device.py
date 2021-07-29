import os
import shutil
import sys
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
# imports for drawing
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter

from absorbing_markov_chain import AbsorbingMarkovChain
from utils import seed_all, project_simplex_2d


class SlowLightDevice:
    def __init__(self,
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
                 ):

        self.alpha = alpha
        self.gamma = gamma
        self.refractive_index = refractive_index
        self.N = N
        self.batch_size = batch_size
        self.weight_of_p_finish_loss = weight_of_p_finish_loss
        self.p_finish_wanted = p_finish_wanted
        self.L = L
        self.min_d = min_d
        self.min_L = min_L
        self.lr = lr
        self.sgd_momentum = sgd_momentum
        self.is_optimize = is_optimize
        self.seed = seed
        self.use_tensorboard = use_tensorboard
        self.device = device

        if self.is_optimize:
            assert self.check_params(), "Negative transition probabilities are possible with the given parameters."

        if self.seed is not None:
            seed_all(self.seed)

        self.Lt, self.dt = self.initialize_lengths(L_init, d_init)
        self.chain, self.Pt, self.num_states, self.init_index, self.finish_index, self.scatter_index, \
        self.return_to_start_index, self.slab_indices = self.initialize_P(self.Lt, self.dt)

        self.optimizer = torch.optim.SGD([self.Lt, self.dt], lr=self.lr, momentum=self.sgd_momentum)

        now = datetime.now().strftime("%Y_%m_%d__%H_%M")
        if log_dir is None:
            self.log_dir = f"./runs/run_{now}"
        if log_dir_suffix is None:
            log_dir_suffix = ""
        else:
            log_dir_suffix = f"__{log_dir_suffix}"
        self.log_dir += log_dir_suffix

        self.tb_logger = None
        if is_optimize:
            os.mkdir(self.log_dir)
            shutil.copy(sys.argv[0], self.log_dir)
            if self.use_tensorboard:
                self.tb_logger = SummaryWriter(log_dir=self.log_dir)

        self.iter_save = []
        self.p_finish_save = []
        self.reward_conditioned_save = []
        self.loss_save = []
        self.p_scatter_save = []
        self.p_reflect_save = []
        self.Lt_save = []
        self.dt_save = []
        self.best_sample_save = []

        self.train_start_time = None

    def check_params(self):
        total_length_check = self.L - (self.N + 1) * self.min_d - self.N * self.min_L

        max_move_prob = np.exp(-self.gamma * self.min_d)
        max_slab_length = self.L - (self.N + 1) * self.min_d - (self.N - 1) * self.min_L
        max_scatter_prob = self.alpha * max_slab_length
        return (total_length_check > 0) and (max_move_prob * 2 + max_scatter_prob <= 1)

    def initialize_lengths(self, L_init: list = None, d_init: list = None):
        if L_init is not None and d_init is not None:
            Lt = torch.tensor(L_init, device=self.device)
            dt = torch.tensor(d_init, device=self.device)
            if self.batch_size == 1 and len(Lt.shape) == 1:
                Lt = torch.unsqueeze(Lt, 0)
            if self.batch_size == 1 and len(dt.shape) == 1:
                dt = torch.unsqueeze(dt, 0)

        elif L_init is None and d_init is None:
            Lt = torch.zeros(self.batch_size, self.N, device=self.device)
            dt = torch.zeros(self.batch_size, self.N + 1, device=self.device)
            for batch in range(self.batch_size):
                Lt_batch, dt_batch = self._initialize_lengths_single()
                Lt[batch, :] = Lt_batch
                dt[batch, :] = dt_batch
        else:
            raise ValueError("Either both L and d are initialised or none of them.")

        Lt.requires_grad = True
        dt.requires_grad = True
        if self.is_optimize:  # don't enforce limitations on SLDs not made for optimizing
            assert Lt.shape[-1] == self.N, "L length isn't N."
            assert dt.shape[-1] == self.N + 1, "L length isn't N+1."
            assert torch.isclose(Lt.sum(-1) + dt.sum(-1),
                                 torch.tensor(1., dtype=Lt.dtype)).all(), "Sum of L and d isn't 1."
            assert (Lt >= self.min_L).all(), "Initialization of L doesn't match minimal value of L."
            assert (dt >= self.min_d).all(), "Initialization of d doesn't match minimal value of d."

        return Lt, dt

    def _initialize_lengths_single(self):
        Lt_i = torch.ones(1, self.N, device=self.device) * self.min_L
        dt_i = torch.ones(1, self.N + 1, device=self.device) * self.min_d
        min_len = self.min_L * self.N + self.min_d * (self.N + 1)
        # Initialize uniformly random lengths
        splits = np.random.random([1, 2 * self.N])
        splits.sort(1)
        splits = np.concatenate([np.zeros([1, 1]), splits, np.ones([1, 1])], 1)
        lengths = np.diff(splits, 1) * (self.L - min_len)
        Lt_i += torch.tensor(lengths[:, :self.N], device=self.device)
        dt_i += torch.tensor(lengths[:, self.N:], device=self.device)
        return Lt_i, dt_i

    def initialize_P(self, Lt: torch.Tensor, dt: torch.Tensor):
        num_states = self.N + 4  # N transition states + init state + death by return to init + death by decay + finish
        init_index = 0
        finish_index = self.N + 1  # in the R matrix this is index 0
        return_to_start_index = self.N + 3  # in the R matrix this is index 2
        scatter_index = self.N + 2  # in the R matrix this is index 1
        slab_indices = range(1, self.N + 1)

        P = torch.zeros([self.batch_size, num_states, num_states], device=self.device, dtype=torch.float64)
        # create transition vectors with length N for standard transition states
        scatter_probs = self.alpha * Lt
        # we put finish_index = self.N + 1 so going forward from last transition leads to finish
        forward_probs = torch.exp(-self.gamma * dt[:, 1:])
        backward_probs = torch.exp(-self.gamma * dt[:, :-1])  # we later deal with the fact that return to start
        # needs to be handled carefully
        staying_probs = torch.ones([self.batch_size, self.N],
                                   device=self.device) - scatter_probs - forward_probs - backward_probs

        # Assign probs to P matrix
        for i in range(1, self.N + 1):
            P[:, i, scatter_index] = scatter_probs[:, i - 1]
            P[:, i, i + 1] = forward_probs[:, i - 1]
            if i == 1:
                P[:, i, return_to_start_index] = backward_probs[:, i - 1]
            else:
                P[:, i, i - 1] = backward_probs[:, i - 1]
            P[:, i, i] = staying_probs[:, i - 1]

        # Assign rest of probs
        P[:, init_index, init_index + 1] = torch.exp(-self.gamma * dt[:, 0])
        P[:, init_index, return_to_start_index] = 1 - torch.exp(-self.gamma * dt[:, 0])
        P[:, finish_index, finish_index] = 1
        P[:, scatter_index, scatter_index] = 1
        P[:, return_to_start_index, return_to_start_index] = 1

        chain = AbsorbingMarkovChain(P, num_absorbing=3, init_calculated_matrices=True)

        return chain, P, num_states, init_index, finish_index, scatter_index, return_to_start_index, slab_indices

    def p_finish(self):
        return self._p_finish(self.chain)

    @staticmethod
    def _p_finish(chain: AbsorbingMarkovChain):
        return chain.probability_of_absorption(0, 0)  # in the R matrix, finishing is index 0

    def p_reflect(self):
        return self._p_reflect(self.chain)

    @staticmethod
    def _p_reflect(chain: AbsorbingMarkovChain):
        return chain.probability_of_absorption(0, 2)  # in the R matrix, reflecting is index 2

    def p_scatter(self):
        return self._p_scatter(self.chain)

    @staticmethod
    def _p_scatter(chain: AbsorbingMarkovChain):
        return chain.probability_of_absorption(0, 1)  # in the R matrix, scattering is index 1

    def reward_conditioned_on_finish(self):
        return self._reward_conditioned_on_finish(self.Lt, self.chain)

    def _reward_conditioned_on_finish(self, Lt: torch.Tensor, chain: AbsorbingMarkovChain):
        return self.refractive_index * torch.sum(Lt * chain.expected_number_of_visits(range(1, self.N + 1), 0,
                                                                                      0), dim=1)

    def build_loss(self):
        return self._build_loss(self.Lt, self.chain)

    def _build_loss(self, Lt: torch.Tensor, chain: AbsorbingMarkovChain):
        time_loss = -torch.log(self._reward_conditioned_on_finish(Lt, chain))
        finish_loss = torch.nn.ReLU()(-(self._p_finish(chain) - self.p_finish_wanted))
        # print(f"time_loss={time_loss.mean():.4f}, finish_loss={finish_loss.mean():.4f}")
        # if (finish_loss > 0).any():
        #     print(self.p_finish())
        return time_loss + self.weight_of_p_finish_loss * finish_loss

    def step(self):
        loss = self.build_loss()
        self.optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        self.optimizer.step()
        # clamps to constraints
        with torch.no_grad():  # can't use in-place operations on a leaf Variable that requires grad, but in-place is
            # required to keep optimizer working on the right parameters
            self._clamp_inplace(self.Lt, self.dt)
        self.Lt.requires_grad = True
        self.dt.requires_grad = True
        self.chain, self.Pt, _, _, _, _, _, _ = self.initialize_P(self.Lt, self.dt)

    def _clamp_not_inplace(self, Lt: torch.Tensor, dt: torch.Tensor):
        concatenated = torch.cat([Lt, dt], dim=1)
        lims = torch.tensor(self.N * [self.min_L] + (self.N + 1) * [self.min_d], device=concatenated.device)
        simplex_sum = self.L - sum(lims)
        to_project = concatenated - lims
        projected_to_simplex = project_simplex_2d(to_project, simplex_sum)
        new_concatenated = projected_to_simplex + lims
        new_Lt = new_concatenated[:, :self.N]
        new_dt = new_concatenated[:, self.N:]
        return new_Lt, new_dt

    def _clamp_inplace(self, Lt: torch.Tensor, dt: torch.Tensor):
        new_Lt, new_dt = self._clamp_not_inplace(Lt, dt)
        Lt.copy_(new_Lt)
        dt.copy_(new_dt)

    def train(self, num_of_iters: int = 100, log_period: int = 1, draw_best_device: bool = False):
        assert self.is_optimize, "Not supported in non-optimizing mode."
        self.train_start_time = time.time()
        self.log(0, draw_best_device=draw_best_device)
        for iter_ in range(num_of_iters):
            self.step()
            if (iter_ + 1) % log_period == 0:
                self.log(iter_ + 1, draw_best_device=draw_best_device)

    def log(self, iter_: int, save_arrays: bool = True, draw_best_device: bool = False):
        assert self.is_optimize, "Not supported in non-optimizing mode."
        train_time = time.time() - self.train_start_time
        p_finish = self.p_finish().detach().cpu().numpy()
        reward_conditioned = self.reward_conditioned_on_finish().detach().cpu().numpy()
        loss = self.build_loss().detach().cpu().numpy()
        p_scatter = self.p_scatter().detach().cpu().numpy()
        p_reflect = self.p_reflect().detach().cpu().numpy()

        best_sample = self.find_best_sample()
        if best_sample == -1:
            warnings.warn(f"Iteration {iter_}, no sample has the necessary p_finish.")

        self.iter_save.append(iter_)
        self.p_finish_save.append(p_finish)
        self.reward_conditioned_save.append(reward_conditioned)
        self.loss_save.append(loss)
        self.p_scatter_save.append(p_scatter)
        self.p_reflect_save.append(p_reflect)
        self.Lt_save.append(self.Lt.detach().cpu().numpy())
        self.dt_save.append(self.dt.detach().cpu().numpy())
        self.best_sample_save.append(best_sample)

        if save_arrays:
            saved_arrays = {"iter": np.array(self.iter_save),
                            "p_finish": np.array(self.p_finish_save),
                            "reward_conditioned": np.array(self.reward_conditioned_save),
                            "loss": np.array(self.loss_save),
                            "p_scatter": np.array(self.p_scatter_save),
                            "p_reflect": np.array(self.p_reflect_save),
                            "Lt": np.array(self.Lt_save),
                            "dt": np.array(self.dt_save),
                            "best_sample": np.array(self.best_sample_save),
                            }
            np.savez(self.log_dir + "/results.npz", **saved_arrays)

        if self.use_tensorboard:
            self.tb_logger.add_scalar('Mean statistics/p_finish', p_finish.mean(), iter_)
            self.tb_logger.add_scalar('Mean statistics/reward_conditioned', reward_conditioned.mean(), iter_)
            self.tb_logger.add_scalar('Mean statistics/loss', loss.mean(), iter_)
            self.tb_logger.add_scalar('Mean statistics/p_scatter', p_scatter.mean(), iter_)
            self.tb_logger.add_scalar('Mean statistics/p_reflect', p_reflect.mean(), iter_)

            self.tb_logger.add_scalar('Best sample/sample_index', best_sample, iter_)
            self.tb_logger.add_scalar('Best sample/p_finish', p_finish[best_sample], iter_)
            self.tb_logger.add_scalar('Best sample/reward_conditioned', reward_conditioned[best_sample], iter_)
            self.tb_logger.add_scalar('Best sample/loss', loss[best_sample], iter_)
            self.tb_logger.add_scalar('Best sample/p_scatter', p_scatter[best_sample], iter_)
            self.tb_logger.add_scalar('Best sample/p_reflect', p_reflect[best_sample], iter_)

            self.tb_logger.add_scalar('misc/train_time', train_time, iter_)

        print(f"Iteration:\t\t\t\t"
              f"{iter_}")

        print(f"Mean statistics: \t\t"
              f"finish probability = {p_finish.mean():.4f}, "
              f"reward conditioned = {reward_conditioned.mean():.4f}, "
              f"loss = {loss.mean():.4f}, "
              f"")

        print(f"Best sample statistics:\t"
              f"finish probabality = {p_finish[best_sample]:.4f}, "
              f"reward conditioned = {reward_conditioned[best_sample]:.4f}, "
              f"loss = {loss[best_sample]:.4f}, "
              f"")

        print(f"Training time:\t\t\t"
              f"{train_time:.2f} seconds")

        print("-----")

        if draw_best_device:
            self.draw(self.Lt[best_sample], self.dt[best_sample])

    def find_best_sample(self):
        loss = self.build_loss()
        return loss.argmin().item()

    def score_simulation_single(self, batch_index: int = 0):
        visit_count, absorbed_state = self.chain.simulate_until_absorbed(starting_state=self.init_index,
                                                                         batch_index=batch_index)
        if absorbed_state != self.finish_index:
            return -np.inf
        L = self.Lt[batch_index].detach().cpu().numpy()
        rewards_collected_list = L * visit_count[self.slab_indices]
        return self.refractive_index * rewards_collected_list.sum()

    def score_simulations_accumulated(self, num_of_simulations: int):
        scores = np.empty(num_of_simulations)
        for i in range(num_of_simulations):
            single_score = self.score_simulation_single()
            scores[i] = single_score
        experimental_rew = np.ma.masked_invalid(scores).mean()
        experimental_pf = (scores != -np.inf).mean()
        return experimental_rew, experimental_pf

    @staticmethod
    def draw(Lt, dt):
        assert len(Lt.shape) == 1, "Insert single device parameters"
        assert len(dt.shape) == 1, "Insert single device parameters"
        assert dt.shape[0] == Lt.shape[0] + 1, "Lengths of Lt and dt arrays don't match"
        if torch.is_tensor(Lt):
            Lt = Lt.detach().cpu().numpy()
        if torch.is_tensor(dt):
            dt = dt.detach().cpu().numpy()
        im = Image.new('RGB', (1000, 100), (128, 128, 128))
        draw = ImageDraw.Draw(im)
        N = Lt.shape[0]
        fill_L = (255, 128, 0)
        fill_d = (255, 255, 0)
        draw.rectangle((0, 0, int(np.round(dt[0] * 1000)), 100), fill_d)
        x = int(np.round(dt[0] * 1000))
        for i in range(N):
            draw.rectangle((x, 0, x + int(np.round(Lt[i] * 1000)), 100), fill_L)
            x += int(np.round(Lt[i] * 1000))
            draw.rectangle((x, 0, x + int(np.round(dt[i + 1] * 1000)), 100), fill_d)
            x += int(np.round(dt[i + 1] * 1000))
        plt.ion()
        plt.imshow(im, extent=[0, 1, 0, 0.1])
        plt.yticks([])
        plt.show()
        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    ##############
    # Batch run:
    ##############
    # init_seed = 73
    # redos = 5
    # for i in range(redos):
    #     seed = init_seed + i
    #     for finish_prob in np.arange(0, 0.36, 0.01):
    #         sld = SlowLightDevice(seed=seed,
    #                               batch_size=256,
    #                               device=device,
    #                               N=10,
    #                               min_L=0.01,
    #                               min_d=0.01,
    #                               gamma=100,
    #                               p_finish_wanted=finish_prob,
    #                               log_dir_suffix=f"new_clamp_seed_{seed}__more_pf"
    #                                              f"_{finish_prob:.2f}")
    #         sld.train(20000, 100, draw_best_device=False)
    #
    # for N in range(1, 30 + 1):
    #     sld = SlowLightDevice(seed=seed,
    #                           batch_size=256,
    #                           device=device,
    #                           N=N,
    #                           min_L=0.01,
    #                           min_d=0.01,
    #                           gamma=100,
    #                           p_finish_wanted=0.1,
    #                           log_dir_suffix=f"new_clamp_seed_{seed}__N"
    #                                          f"_{N}")
    #     sld.train(20000, 100, draw_best_device=False)

    # for alpha in np.logspace(-4, -1, 31):
    #     sld = SlowLightDevice(seed=seed,
    #                           batch_size=256,
    #                           device=device,
    #                           N=10,
    #                           min_L=0.01,
    #                           min_d=0.01,
    #                           gamma=100,
    #                           p_finish_wanted=0.1,
    #                           alpha=alpha,
    #                           log_dir_suffix=f"new_clamp_seed_{seed}__alpha"
    #                                          f"_{alpha}")
    #     sld.train(20000, 100, draw_best_device=False)

    ##############
    # Single run:
    ##############
    sld = SlowLightDevice(seed=73, batch_size=1, device=device, N=8,
                          log_dir_suffix=f"added_actual_return_to_start_now_what_1st_try")
    sld.train(20000, 100, draw_best_device=True)
