from dp import transition_and_cost, get_initial_state, get_possible_actions
from const import TICK_LIMIT, TICKS
import pandas as pd
import numpy as np
import sys, os
import joblib
import time

K = 1
LENGTH = 201
NUM_PATHS = 10_000
paths = np.load("data/sample_paths.npy")[:NUM_PATHS, :LENGTH]

state_at_step = {
    i : {} for i in range(LENGTH)
}
start = time.time()
for i, path in enumerate(paths):
    
    print(f"Progress: {(i + 1 ) / len(paths) * 100}%")
    state = get_initial_state()

    for i, v in enumerate(path):
        
        if i+1 == LENGTH: break
        actions = get_possible_actions(state)
        
        for action in actions:
        
            next_state, cost = transition_and_cost(state.copy(), action, v)
            next_state = tuple(next_state)
            
            try:
                state_at_step[i+1][next_state] += 1
            except:
                state_at_step[i+1][next_state] = 1
        
        idx = np.random.randint(0, len(actions))
        next_state, cost = transition_and_cost(state.copy(), actions[idx], v)
        state = next_state
end = time.time()

print("Total Computation Time", end - start)
with open(f'states/states_{NUM_PATHS}.pkl', 'wb') as file:
    joblib.dump(state_at_step, file)