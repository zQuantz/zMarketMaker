from dp import get_initial_state, get_possible_actions, transition_and_cost, terminal_cost
from dp import MAX_POSITION, MAX_UNREALIZED
from const import TICK_LIMIT, TICKS
from itertools import product
import pandas as pd
import numpy as np
import sys, os
import pickle
import gc

###################################################################################################

coocc = pd.read_csv("data/cooccurrence_matrix.csv", index_col=0)
ticks = coocc.columns.astype(int).tolist()

coocc = coocc.values
coocc = (coocc.T / coocc.sum(axis=1)).T

###################################################################################################

def state_generator(max_k):
    
    positions = {
        i : [] for i in range(max_k + 1)
    }
    unrealized_pnls = positions.copy()
    ticks = positions.copy()
    
    ## Base Case
    ticks[0] = [0]
    positions[0] = [0]
    unrealized_pnls[0] = [0]
    
    ## Second Base Case
    ticks[1] = TICKS
    positions[1] = [-1, 0, 1]
    unrealized_pnls[1] = [0]*len(TICKS)
    
    for i in range(2, max_k + 1):
        
        ticks[i] = TICKS
        
        k = min(i, MAX_POSITION)
        positions[i] = [position - k for position in range(0, k * 2 + 1)]
        unrealized_pnls[i] = [i - MAX_UNREALIZED for i in range(0, 2 * MAX_UNREALIZED + 1)]
        
    states = {i : [] for i in range(len(positions))}
    for i in range(len(states)):
        states[i] = list(product(set(positions[i]), set(unrealized_pnls[i]), set(ticks[i])))
        states[i] = [list(state) for state in states[i]]
        
    return states

def solve():

	K = 200
	states = state_generator(K)

	### N-Step
	J_N, U_N = {}, {}
	for state in states[K]:
		J_N[tuple(state)] = terminal_cost(state)

	J, U = J_N, U_N
	with open(f'dicts/U_{K}.pickle', 'wb') as file:
			pickle.dump(U_N, file)

	with open(f'dicts/J_{K}.pickle', 'wb') as file:
		pickle.dump(J_N, file)

	## K-Steps
	while(K > 0):

		K -= 1
		print("Starting Stage", K)

		J_K, U_K = {}, {}
		for j, state in enumerate(states[K]):

			print("State", j)

			actions = get_possible_actions(state)
			costs = [-10000]*len(actions)

			for i, action in enumerate(actions):

				print("Action", i)

				avg_cost = 0
				
				for tick in ticks:

					new_state, cost = transition_and_cost(state.copy(), action, tick)
					p = coocc[state[-1] + TICK_LIMIT, tick + TICK_LIMIT]
					print("Next Tick Probability", p)
					print("Next Cost", cost)

					ns = tuple(new_state)
					print("Next State Cost-to-go", J[ns])
					cost += J[ns]
					avg_cost += p * cost

				print("State/Action Average Cost", avg_cost)
				costs[i] = avg_cost

			idx = np.argmax(costs)
			state = tuple(state)
			J_K[state] = costs[idx]
			U_K[state] = actions[idx]

		with open(f'dicts/U_{K}.pickle', 'wb') as file:
			pickle.dump(U_K, file)

		with open(f'dicts/J_{K}.pickle', 'wb') as file:
			pickle.dump(J_K, file)

		J = J_K
		U = U_K

		gc.collect()

###################################################################################################

if __name__ == '__main__':

	solve()