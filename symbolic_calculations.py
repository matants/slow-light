import sympy as sp

S_INIT = 0
S_FORWARD = 1
S_BACKWARD = 2
S_REFLECT = 3
S_TRANSMIT = 4
NUM_TRANSIENT = 3

sp.init_printing(use_unicode=True)

d0, d1, L1, gamma, Pf, L = sp.symbols(
    'd0 d1 L1 gamma Pf L')

# d1 = d0
L1 = L - d0 - d1

P = sp.Matrix([[0, sp.exp(-gamma * d0), 0, 1 - sp.exp(-gamma * d0), 0],
               [0, 0, 1 - sp.exp(-gamma * d1), 0, sp.exp(-gamma * d1)],
               [0, 1 - sp.exp(-gamma * d0), 0, sp.exp(-gamma * d0), 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]])

Q = P[:NUM_TRANSIENT, :NUM_TRANSIENT]
R = P[:NUM_TRANSIENT, NUM_TRANSIENT:]

N = (sp.eye(NUM_TRANSIENT) - Q).inv()
B = N * R
B_Pf = sp.simplify(B[S_INIT, S_TRANSMIT - NUM_TRANSIENT])
epsilon_sum_for_slab = sp.simplify(((1 / B_Pf) * N[S_INIT, S_FORWARD:S_BACKWARD + 1] *
                                    B[S_FORWARD:S_BACKWARD + 1, S_TRANSMIT - NUM_TRANSIENT])[0, 0])

reward = L1 * epsilon_sum_for_slab
