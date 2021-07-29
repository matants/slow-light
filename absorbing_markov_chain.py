import numpy as np
import torch


class AbsorbingMarkovChain:
    def __init__(self, Pt: torch.Tensor, num_absorbing: int, init_calculated_matrices: bool = True):
        self.device = Pt.device
        if len(Pt.shape) == 2:
            self.Pt = torch.unsqueeze(Pt, 0)
        elif len(Pt.shape) == 3:
            self.Pt = Pt
        else:
            raise ValueError("Pt dimensions aren't valid.")
        assert self.Pt.shape[1] == self.Pt.shape[2], "Transition matrix isn't square"
        assert (self.Pt >= 0).all(), "There are negative transition probabilities."
        assert torch.isclose(self.Pt.sum(dim=2), torch.tensor(1., dtype=self.Pt.dtype)).all(), "Not all rows sum to 1."
        self.num_states = self.Pt.shape[-1]
        self.num_absorbing = num_absorbing
        self.num_transient = self.num_states - self.num_absorbing

        self.Qt = self.Pt[:, :self.num_transient, :self.num_transient]
        self.Rt = self.Pt[:, :self.num_transient, self.num_transient:]

        self.Nt = None
        self.Bt = None

        if init_calculated_matrices:
            self.calc_B(True)

    def calc_N(self):
        try:
            self.Nt = torch.inverse(torch.unsqueeze(torch.eye(self.num_transient, device=self.device), 0) - self.Qt)
            # self.Nt = torch.pinverse(torch.unsqueeze(torch.eye(self.num_transient, device=self.device), 0) - self.Qt)
        except Exception as err:
            raise err
        return self.Nt

    def calc_B(self, update_N: bool = True):
        if update_N:
            self.calc_N()
        self.Bt = torch.matmul(self.Nt, self.Rt)
        return self.Bt

    def expected_number_of_visits(self, state_to_count, initial_state, absorbing_state,
                                  correction_epsilon: float = 1e-8):
        if self.Nt is None or self.Bt is None:
            raise RuntimeError("Calculate N and B matrices first.")
        return (self.Bt[:, state_to_count, absorbing_state] / (torch.unsqueeze(self.Bt[:, initial_state,
                                                                               absorbing_state],
                                                                               1) + correction_epsilon)) * self.Nt[:,
                                                                                                           initial_state,
                                                                                                           state_to_count]

    def probability_of_absorption(self, initial_state, absorbing_state):
        if self.Nt is None or self.Bt is None:
            raise RuntimeError("Calculate N and B matrices first.")
        return self.Bt[:, initial_state, absorbing_state]

    def simulate_until_absorbed(self, starting_state: int, batch_index: int):
        visit_counts = np.zeros(self.num_states)
        visit_counts[starting_state] += 1
        P = self.Pt[batch_index].detach().cpu().numpy()
        while True:
            probs = P[starting_state]
            probs /= probs.sum()  # normalize to handle numpy issue with sums close to 1 but not equal
            next_state = np.random.choice(self.num_states, p=probs)
            visit_counts[next_state] += 1
            if next_state >= self.num_transient:  # Absorbed
                return visit_counts, next_state  # returning visit counts and absorbing state
            starting_state = next_state


if __name__ == '__main__':
    P = torch.tensor([[[0.15, 0.1, 0.55, 0.2], [0.4, 0.05, 0.25, 0.3], [0, 0, 1, 0], [0, 0, 0, 1]],
                      [[0, 0.25, 0.55, 0.2], [0.4, 0.05, 0.25, 0.3], [0, 0, 1, 0], [0, 0, 0, 1]]])
    chain = AbsorbingMarkovChain(P, 2)
