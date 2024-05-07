"""
Name:(Kyle Webb)
Date:(4/6/24)
Assignment:(Assignment #11)
Due Date:(4/7/24)
About this project:(Markov Decision Process to decide whether to fish salmon this year)
Assumptions:(None)
All work below was performed by (Kyle Webb)
"""

import mdptoolbox
import numpy as np

# Transitions
P = np.array([
    # P[1] = Re-breed only can re-breed when empty
    [[0, 1, 0, 0],  # 0 = empty
     [0, 1, 0, 0],  # 1 = low
     [0, 1, 0, 0],  # 2 = medium
     [0, 1, 0, 0]],  # 3 = high
    # P[1] = Not to Fish
    [[1, 0, 0, 0],  # 0 = empty : unable - not to fish
     [0, 0.3, 0.7, 0],  # 1 = low
     [0, 0, 0.25, 0.75],  # 2 = medium
     [0, 0, 0.05, 0.95]],  # 3 = high
    # P[2] = Fish
    [[1, 0, 0, 0],  # 0 = empty : unable - fish
     [0.75, 0.25, 0, 0],  # 1 = low
     [0, 0.75, 0.25, 0],  # 2 = medium
     [0, 0, 0.6, 0.4]]  # 3 = high
])

# using : " assume the rewards of fishing at state low, medium and high are $5K, $50K and $100k respectively "
# [0] = Re-Breed
# [1] = Not to Fish
# [2] = Fish
R = np.array([[-200000, 0, 0],  # empty
              [0, 0, 5000],  # low
              [0, 0, 50000],  # medium
              [0, 0, 100000]  # high
              ])

print("P=", P)
print("R=", R)

Discount = 0.9
NumPeriods = 10

##########################
print("Policy Iteration")
pi = mdptoolbox.mdp.PolicyIteration(P, R, Discount)
pi.setVerbose()
pi.run()
print("optimal value function=", pi.V)
print("optimal policy=", pi.policy)
##########################
##########################
print("Value Iteration")
vi = mdptoolbox.mdp.ValueIteration(P, R, Discount, NumPeriods)
vi.setVerbose()
vi.run()
print("optimal value function=", vi.V)
print("optimal policy=", vi.policy)
##########################
