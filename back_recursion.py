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

def state_generator(k):
	
	unrealized = [0]
	position = [0]
	move = [0]

	for i in range(1, k+1):
		move.append(-TICK_LIMIT)
		position.append(min(i+1, MAX_POSITION))
		unrealized.append(max(-MAX_UNREALIZED, unrealized[-1] + position[i-1] * move[-1]))

	max_unrl = max(abs(np.array(unrealized)))
	possible_unrealized = np.arange(-max_unrl, max_unrl+1)
	possible_position = np.arange(-MAX_POSITION, MAX_POSITION+1)
	ask_prices = [i for i in range(0, TICK_LIMIT+2)]
	bid_prices = [-i for i in range(0, TICK_LIMIT+2)]
	
	return product(possible_position, possible_unrealized, bid_prices, [1], ask_prices, [1], TICKS)

def solve():

	K = 50

	J_N = {}
	U_N = {}
	for state in state_generator(K):
		J_N[state] = terminal_cost(state)

	J = J_N
	U = U_N

	while(K >= 0):

		print("Starting Stage", K)

		J_K, U_K = {}, {}
		for j, state in enumerate(state_generator(K-1)):

			print("State", j)

			actions = get_possible_actions(state)
			costs = [-10000]*len(actions)

			for i, action in enumerate(actions):

				print("Action", i)

				avg_cost = 0
				
				for tick in ticks:

					new_state, cost = transition_and_cost(list(state), action, tick)
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
			J_K[state] = costs[idx]
			U_K[state] = actions[idx]

		with open(f'dicts/U_{K}.pickle', 'wb') as file:
			pickle.dump(U_K, file)

		with open(f'dicts/J_{K}.pickle', 'wb') as file:
			pickle.dump(J_K, file)

		K -= 1

		J = J_K
		U = U_K

		gc.collect()

###################################################################################################

if __name__ == '__main__':

	solve()