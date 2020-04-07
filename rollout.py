from dp import transition_and_cost, get_initial_state, get_possible_actions
from argparse import ArgumentParser
from const import TICKS, TICK_LIMIT
import pandas as pd
import numpy as np
import sys, os
import pickle
import joblib
import time

###################################################################################################

coocc = pd.read_csv('data/cooccurrence_matrix.csv', index_col = 0)
coocc = (coocc.T / coocc.sum(axis=1)).T.values

paths = np.load('data/sample_paths.npy')

###################################################################################################
## Get some set of actions sorted by prority to reduce the number of calculations during the rollout.

actions = get_possible_actions(get_initial_state())
score = [
    abs(action[0]) + abs(action[2])
    for action in actions
]
idc = np.argsort(score)
SORTED_ACTIONS = [actions[idx] for idx in idc]

A_PROBS = [5] * 20 + [2] * 4
A_PROBS = np.array(A_PROBS) / sum(A_PROBS)

A_RANGE = np.arange(24)

###################################################################################################

def get_action_subset(state):
	actions = get_possible_actions(state)
	if len(actions) == 5:
		return actions
	idc = np.random.choice(A_RANGE, size=10, p=A_PROBS)
	return [SORTED_ACTIONS[idx] for idx in idc]

def deeper(state, total_cost, weight, k):

	if k == 2:
		global costs
		costs.append(total_cost * weight)
	else:
		actions = get_action_subset(state)
		ps = coocc[state[-1] + TICK_LIMIT, :]
		for action in actions:
			for tick in np.random.choice(TICKS, p=ps, size=4):
				next_state, cost = transition_and_cost(state.copy(), action, tick)
				deeper(next_state, total_cost + cost, ps[tick + TICK_LIMIT] * weight, k+1)

def rollout(K_, path_name):

	try:
		os.mkdir(f"rollout/{K_}/")
	except Exception as e:
		print(e)

	path = np.load(f"paths/{K_}/{path_name}.npy")
	###################################################################################################

	start = time.time()

	K = K_
	state = get_initial_state()
	states, policy, rewards = [], [], []

	## Only 1 step ahead
	for k in range(K):

		print("Stage", k)

		actions = get_possible_actions(state)
		cost_to_gos = []

		for action in actions:
			avg_cost = 0
			for tick in TICKS:	
				next_state, cost = transition_and_cost(state.copy(), action, tick)
				p = coocc[state[-1] + TICK_LIMIT, tick + TICK_LIMIT]
				## Approximate Cost To Go Function
				global costs 
				costs = []
				deeper(next_state, 0, 1, 0)
				avg_cost += (np.mean(costs) + cost) * p
			cost_to_gos.append(avg_cost)

		idx = np.argmax(cost_to_gos)
		best_action = actions[idx]
		next_state, reward = transition_and_cost(state.copy(), best_action, path[k])

		states.append([state, next_state])
		policy.append(best_action)
		rewards.append(reward)

		state = next_state

	end = time.time()

	###################################################################################################

	objs = {
		"states" : states,
		"policy" : policy,
		"rewards" : rewards 
	}
	with open(f'rollout/{K_}/{path_name}_policy.pkl', 'wb') as file:
		joblib.dump(objs, file)

	try:

		with open("timers/timer_dict.pkl", "rb") as file:
			timer_dict = joblib.load(file)
			key = timer_dict.get("rollout", None)
			if not key:
				timer_dict["rollout"] = {}
			timer_dict["rollout"][K_] = end - start

		with open("timers/timer_dict.pkl", "wb") as file:
			joblib.dump(timer_dict, file)

	except Exception as e:

		print(e)
		with open("timers/timer_dict.pkl", "wb") as file:
			timer_dict = {"rollout" : {}}
			timer_dict["rollout"][K_] = end - start
			joblib.dump(timer_dict, file)

###################################################################################################

if __name__ == '__main__':

	argparser = ArgumentParser()
	argparser.add_argument("K")
	argparser.add_argument("path_name")
	args = argparser.parse_args()

	rollout(int(args.K), args.path_name)