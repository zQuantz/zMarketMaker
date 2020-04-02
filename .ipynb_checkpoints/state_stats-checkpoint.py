from dp import ACTIONS, transition_and_cost, get_initial_state
import pandas as pd
import numpy as np
import sys, os
import pickle

LENGTH = 100
paths = np.load("data/sample_paths.npy")[:, :100]

state_at_step = {
    i : [] for i in range(LENGTH)
}

for i, path in enumerate(paths):
    print(f"Progress: {(i + 1) / len(paths) * 100}%")
    state = get_initial_state()
    state_at_step[0].append(tuple(state))
    for i, v in enumerate(path):
        if i+1 == LENGTH: break
        for action in ACTIONS:
            next_state, cost = transition_and_cost(state.copy(), action, v)
            state_at_step[i+1].append(tuple(next_state))
        state = next_state
    
for key in state_at_step.copy().keys():
    state_at_step[key] = set(state_at_step[key])

with open('data/states.pickle', 'wb') as file:
    pickle.dump(state_at_step, file)