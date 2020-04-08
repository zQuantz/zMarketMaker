from dp import get_initial_state, get_possible_actions, transition_and_cost
from const import TICKS, TICK_LIMIT
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, os
import joblib

###################################################################################################

COLS = [
		'Path Name',
		'Num. Trades',
		'Closing Trades',
		'Continuation Trades',
		'Max Runup',
		'Max Drawdown',
		'Most Held Position',
		'Most Held Position by %',
		'Least Held Position',
		'Least Held Position by %',
		'Total Revenue',
		'Avg Revenue per Period'
	]

coocc = pd.read_csv('data/cooccurrence_matrix.csv', index_col = 0)
coocc = (coocc.T / coocc.sum(axis=1)).T.values

###################################################################################################

def get_paths(K):
	return {
		"uptrend" : np.load(f"paths/{K}/uptrend.npy"),
		"downtrend" : np.load(f"paths/{K}/downtrend.npy"),
		"no_change" : np.load(f"paths/{K}/no_change.npy"),
		"high_vol" : np.load(f"paths/{K}/high_vol.npy"),
		"low_vol" : np.load(f"paths/{K}/low_vol.npy"),
		"avg_vol" : np.load(f"paths/{K}/avg_vol.npy"),
	}

def get_random_action(k, state):
	actions = get_possible_actions(state)
	return actions[np.random.randint(0, len(actions))]

def get_linear_parametrized_action(model, state, k):

	costs = []
	actions = get_possible_actions(state)
	for action in actions:
		
		avg_cost = 0
		
		for tick in TICKS:
			
			p = coocc[state[-1] + TICK_LIMIT, tick + TICK_LIMIT]
			next_state, cost = transition_and_cost(state.copy(), action, tick)
			X = np.array([next_state])
			pred = model[k+1].predict(X)[0]
			avg_cost += p * (cost + pred)
		
		costs.append(avg_cost)
	
	idx = np.argmax(costs)
	return actions[idx]

###################################################################################################

def calculate_stats(states, costs, path_name):
		
	states = np.array(states)
	
	max_runup = states[:, 1].max()
	max_drawdown = states[:, 1].min()

	positions = pd.Series(states[:, 0]).value_counts()
	positions = positions / sum(positions)

	most_held_position = positions.index[0]
	most_held_pct = positions.values[0]

	least_held_position = positions.index[-1]
	least_held_pct = positions.values[-1]
	
	trades = states[:, 0]

	closers = 0
	adders = 0
	num_trades = 0

	for i in range(1, len(trades)):
		t = trades[i]
		t1 = trades[i-1]
		if t != t1:
			d = t-t1
			if np.sign(d) == np.sign(t1):
				adders += 1
			elif np.sign(d) != np.sign(t1):
				closers += 1
			num_trades += 1
			
	total_revenue = sum(costs)
	avg_rev_per_period = np.mean(costs)
	
	return [
		path_name,
		num_trades,
		closers,
		adders,
		max_runup,
		max_drawdown,
		most_held_position,
		most_held_pct,
		least_held_position,
		least_held_pct,
		total_revenue,
		avg_rev_per_period,
	]

###################################################################################################

def back_recursion():

	try:
		os.mkdir("back_recursion/results/")
	except Exception as e:
		print(e)

	for K in [50, 100, 200]:

		stats = []
		paths = get_paths(K)
		with open(f"back_recursion/{K}/back_recursion.pkl", "rb") as file:
			model = joblib.load(file)

		for path_name in paths:

			state = get_initial_state()
			states, costs = [state], [0]
		
			for i, tick in enumerate(paths[path_name]):
				next_state, cost = transition_and_cost(state.copy(), model[i]['U'][tuple(state)], tick)
				states.append(next_state)
				costs.append(cost)
				state = next_state

			stats.append(calculate_stats(states, costs, path_name))

		df = pd.DataFrame(stats, columns = COLS)
		df.to_csv(f"back_recursion/results/{K}.csv", index=False)

def approximation():

	try:
		os.mkdir("approximation/results/")
	except Exception as e:
		print(e)

	for K in [50, 100, 200]:

		stats = []
		paths = get_paths(K)
		with open(f"approximation/{K}/linear_models.pkl", "rb") as file:
			model = joblib.load(file)

		for path_name in paths:

			state = get_initial_state()
			states, costs = [state], [0]
		
			for i, tick in enumerate(paths[path_name]):
				action = get_linear_parametrized_action(model, state.copy(), i)
				next_state, cost = transition_and_cost(state.copy(), action, tick)
				states.append(next_state)
				costs.append(cost)
				state = next_state

			stats.append(calculate_stats(states, costs, path_name))

		df = pd.DataFrame(stats, columns = COLS)
		df.to_csv(f"approximation/results/{K}.csv", index=False)

def rollout():

	try:
		os.mkdir("rollout/results/")
	except Exception as e:
		print(e)

	for K in [50, 100, 200]:

		stats = []
		paths = get_paths(K)

		for path_name in paths:

			with open(f"rollout/{K}/{path_name}_policy.pkl", "rb") as file:
				policy = joblib.load(file)

			states = [state[0] for state in policy['states']]
			costs = policy['rewards']

			stats.append(calculate_stats(states, costs, path_name))

		df = pd.DataFrame(stats, columns = COLS)
		df.to_csv(f"rollout/results/{K}.csv", index=False)

if __name__ == '__main__':

	print("Generating Back Recursion Report")
	back_recursion()

	print("Generating Approximation Report")
	approximation()

	print("Generating Rollout Report")
	rollout()
