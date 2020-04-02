from dp import transition_and_cost, get_initial_state, get_possible_actions
from const import TICK_LIMIT, TICKS
import pandas as pd
import numpy as np
import sys, os
import pickle

K = 1
LENGTH = 20
paths = np.load("data/sample_paths.npy")[:, :20]

state_at_step = {
    i : set() for i in range(LENGTH)
}

for k in range(K):
    for i, path in enumerate(paths):
        print(f"Progress: {(i + 1 + k * len(paths)) / (len(paths) * K) * 100}%")
        state = get_initial_state()
        state_at_step[0].add(tuple(state))
        for i, v in enumerate(path):
            if i+1 == LENGTH: break
            actions = get_possible_actions(state)
            for action in actions:
                next_state, cost = transition_and_cost(state.copy(), action, v)
                next_state = tuple(next_state)
                state_at_step[i+1].add(next_state)
                for tick in TICKS:
                    next_state, cost = transition_and_cost(state.copy(), action, tick)
                    next_state = tuple(next_state)
                    state_at_step[i+1].add(next_state)
            idx = np.random.randint(0, len(actions))
            next_state, cost = transition_and_cost(state.copy(), actions[idx], v)
            state = next_state

with open('data/states.pickle', 'wb') as file:
    pickle.dump(state_at_step, file)