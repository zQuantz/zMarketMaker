from dp import transition_and_cost, get_initial_state, get_possible_actions
from const import TICKS, TICK_LIMIT
import pandas as pd
import numpy as np
import sys, os
import pickle

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

def deeper(state, weighted_cost, k):

	if k == 3:
		global costs
		costs.append(weighted_cost)
	else:
		actions = get_action_subset(state)
		ps = coocc[state[-1] + TICK_LIMIT, :]
		for action in actions:
			for tick in np.random.choice(TICKS, p=ps, size=4):
				next_state, cost = transition_and_cost(state.copy(), action, tick)
				deeper(next_state, ps[tick + TICK_LIMIT] * cost, k+1)

def rollout():

	K = 200
	path_idx = 0
	path = paths[path_idx, :K]

	state = get_initial_state()

	states, policy, rewards = [], [], []

	## Only 1 step ahead
	for k in range(K):

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
				deeper(next_state, 0, 0)
				avg_cost += (np.mean(costs) + cost) * p
			cost_to_gos.append(avg_cost)

		idx = np.argmax(cost_to_gos)
		best_action = actions[idx]
		next_state, reward = transition_and_cost(state.copy(), best_action, path[k])

		states.append([state, next_state])
		policy.append(best_action)
		rewards.append(reward)

		state = next_state

		print(k)
		print(state, next_state, best_action, reward)
		print()

	objs = {
		"states" : states,
		"policy" : policy,
		"rewards" : rewards 
	}
	with open(f'rollout/{path_idx}_{k+1}.pickle', 'wb') as file:
		pickle.dump(objs, file)

###################################################################################################

if __name__ == '__main__':

	rollout()