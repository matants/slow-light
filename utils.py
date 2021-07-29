import random

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


def permute_array(array, permutation_order):
    """
    Permutes array according to order given. works on numpy and torch.
    :param array:
    :param permutation_order:
    :return:
    """
    if type(permutation_order) is tuple:
        permutation_order = list(permutation_order)
    dims = len(array.shape)
    if dims == 1:
        assert array.shape[0] == len(permutation_order), "Permutation length doesn't match input array"
        return array[permutation_order]
    elif dims == 2:
        assert array.shape[1] == len(permutation_order), "Permutation length doesn't match input array"
        return array[:, permutation_order]
    else:
        raise ValueError("Only 1d and 2d arrays supported.")


def project_simplex_2d(v, z):
    """
    Implements the algorithm in Figure 1 of
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    This algorithm project vectors v onto the simplex w >= 0, \sum w_i = z.
     - Assuming that all vectors are arranged in rows of v.
    Taken from: https://github.com/smatmo/ProjectionOntoSimplex/
    :param v: NxD torch tensor; Duchi et al. algorithm is applied to each row in vecotrized form
    :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
    :return: w: result of the projection
    """
    with torch.no_grad():
        shape = v.shape
        if shape[1] == 1:
            w = v.clone().detach()
            w[:] = z
            return w

        mu = torch.sort(v, dim=1, descending=True)[0]
        cum_sum = torch.cumsum(mu, dim=1)
        j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
        rho = torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1
        max_nn = cum_sum[torch.arange(shape[0]), rho[:, 0]]
        theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
        w = torch.clamp(v - theta, min=0.0)
        return w


if __name__ == '__main__':
    test_tensor = torch.tensor([[0.2, 0.2, 1.1], [-0.1, 1.5, 0.8], [-0.1, -0.2, -0.3]])
    projected = project_simplex_2d(test_tensor, 1)
    db = 1
