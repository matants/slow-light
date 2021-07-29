import sympy as sp

S_INIT = 0
S_1_FORWARD = 1
S_1_BACKWARD = 2
S_2_FORWARD = 3
S_2_BACKWARD = 4
S_REFLECT = 5
S_TRANSMIT = 6
NUM_SLABS = 2
NUM_STATES = NUM_SLABS * 2 + 3
NUM_TRANSIENT = NUM_STATES - 2

sp.init_printing(use_unicode=True)

d0, d1, d2, L1, L2, gamma, Pf, L = sp.symbols(
    'd0 d1 d2 L1 L2 gamma Pf L')
#
# d1 = d0
# L1 = L - d0 - d1

P = sp.Matrix([[0] * NUM_STATES] * NUM_STATES)
P[S_INIT, S_1_FORWARD] = sp.exp(-gamma * d0)
P[S_INIT, S_REFLECT] = 1 - sp.exp(-gamma * d0)
P[S_1_FORWARD, S_2_FORWARD] = sp.exp(-gamma * d1)
P[S_1_FORWARD, S_1_BACKWARD] = 1 - sp.exp(-gamma * d1)
P[S_2_FORWARD, S_TRANSMIT] = sp.exp(-gamma * d2)
P[S_2_FORWARD, S_2_BACKWARD] = 1 - sp.exp(-gamma * d2)
P[S_1_BACKWARD, S_REFLECT] = sp.exp(-gamma * d0)
P[S_1_BACKWARD, S_1_FORWARD] = 1 - sp.exp(-gamma * d0)
P[S_2_BACKWARD, S_1_BACKWARD] = sp.exp(-gamma * d1)
P[S_2_BACKWARD, S_2_FORWARD] = 1 - sp.exp(-gamma * d1)
P[S_REFLECT, S_REFLECT] = 1
P[S_TRANSMIT, S_TRANSMIT] = 1

Q = P[:NUM_TRANSIENT, :NUM_TRANSIENT]
R = P[:NUM_TRANSIENT, NUM_TRANSIENT:]

N = sp.simplify((sp.eye(NUM_TRANSIENT) - Q).inv())
B = sp.simplify(N * R)
B_Pf = sp.simplify(B[S_INIT, S_TRANSMIT - NUM_TRANSIENT])
epsilon_sum_for_slab1 = sp.simplify(((1 / B_Pf) * N[S_INIT, S_1_FORWARD:S_1_BACKWARD + 1] *
                                     B[S_1_FORWARD:S_1_BACKWARD + 1, S_TRANSMIT - NUM_TRANSIENT])[0, 0])
epsilon_sum_for_slab2 = sp.simplify(((1 / B_Pf) * N[S_INIT, S_2_FORWARD:S_2_BACKWARD + 1] *
                                     B[S_2_FORWARD:S_2_BACKWARD + 1, S_TRANSMIT - NUM_TRANSIENT])[0, 0])

reward = L1 * epsilon_sum_for_slab1 + L2 * epsilon_sum_for_slab2
