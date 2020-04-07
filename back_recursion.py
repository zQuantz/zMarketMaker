from dp import get_possible_actions, transition_and_cost, terminal_cost
from dp import MAX_POSITION, MAX_UNREALIZED
from const import TICK_LIMIT, TICKS
from argparse import ArgumentParser
from itertools import product
import pandas as pd
import numpy as np
import sys, os
import joblib
import time
import gc

###################################################################################################

argparser = ArgumentParser()
argparser.add_argument("K")
args = argparser.parse_args()

coocc = pd.read_csv("data/cooccurrence_matrix.csv", index_col=0).values
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

###################################################################################################

def solve(K_):

	try:
		os.mkdir(f"back_recursion/{K_}")
	except Exception as e:
		print(e)

	K = K_
	start = time.time()
	states = state_generator(K)

	###################################################################################################

	### N-Step
	MEMORY, J_N, U_N = {}, {}, {}
	for state in states[K]:
		J_N[tuple(state)] = terminal_cost(state)

	J_K_1, U_K_1 = J_N, U_N
	MEMORY[K] = {
		"J" : J_K_1,
		"U" : U_K_1
	}

	## K-Steps
	while(K > 0):

		K -= 1
		print("Starting Stage", K)

		J_K, U_K = {}, {}
		for j, state in enumerate(states[K]):

			actions = get_possible_actions(state)
			costs = [-10000]*len(actions)

			for i, action in enumerate(actions):

				avg_cost = 0
					
				for tick in TICKS:

					new_state, cost = transition_and_cost(state.copy(), action, tick)
					p = coocc[state[-1] + TICK_LIMIT, tick + TICK_LIMIT]

					ns = tuple(new_state)
					cost += J_K_1[ns]
					avg_cost += p * cost

				costs[i] = avg_cost

			idx = np.argmax(costs)
			state = tuple(state)
			J_K[state] = costs[idx]
			U_K[state] = actions[idx]


		J_K_1 = J_K
		U_K_1 = U_K
		MEMORY[K] = {
			"J" : J_K_1,
			"U" : U_K_1
		}

		gc.collect()

	with open(f"back_recursion/{args.K}/back_recursion.pkl", "wb") as file:
		joblib.dump(MEMORY, file)

	###################################################################################################

	end = time.time()

	try:

		with open("timers/timer_dict.pkl", "rb") as file:
			timer_dict = joblib.load(file)
			key = timer_dict.get("back_recursion", None)
			if not key:
				timer_dict["back_recursion"] = {}
			timer_dict["back_recursion"][K_] = end - start

		with open("timers/timer_dict.pkl", "wb") as file:
			joblib.dump(timer_dict, file)

	except Exception as e:

		print(e)
		with open("timers/timer_dict.pkl", "wb") as file:
			timer_dict = {"back_recursion" : {}}
			timer_dict["back_recursion"][K_] = end - start
			joblib.dump(timer_dict, file)

###################################################################################################

if __name__ == '__main__':

	solve(int(args.K))