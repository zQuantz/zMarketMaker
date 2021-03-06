from dp import get_initial_state, get_possible_actions, terminal_cost, transition_and_cost
from sklearn.linear_model import LinearRegression
from const import TICK_LIMIT, TICKS
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import sys, os
import joblib
import time

###################################################################################################

argparser = ArgumentParser()
argparser.add_argument("K")
args = argparser.parse_args()

coocc = pd.read_csv('data/cooccurrence_matrix.csv', index_col = 0)
coocc = (coocc.T / coocc.sum(axis=1)).T.values

with open(f'states/states_10000.pkl', 'rb') as file:
	states = joblib.load(file)
states[0] = {tuple(get_initial_state()) : 10_000}
assert int(args.K) <= len(states)

###################################################################################################

def approx(K_):

	try:
		os.mkdir(f"approximation/{K_}")
	except Exception as e:
		print(e)

	start = time.time()
	np.random.seed(72)
	models = {}
	K = K_

	X, y = [], []
	for state, count in states[K].items():
		
		cost = terminal_cost(state)
		
		state = list(state)
		count = int(count / 4)
		
		X.extend([state]*count)
		y.extend([cost]*count)

	X, y = np.array(X), np.array(y)

	model = LinearRegression().fit(X, y)
	models[K] = model
	print(f"Stage {K} Model Fitted")

	K-=1
	while(K >= 0):

		X = []
		y = []

		for i, (state, count) in enumerate(states[K].items()):
 
			state = list(state)
			count = int(count / 4)

			actions = get_possible_actions(state)
			costs = []

			for action in actions:

				tick_costs = []
				next_states = []
				ps = []

				for tick in TICKS:

					p = coocc[state[-1] + TICK_LIMIT, tick + TICK_LIMIT]
					next_state, cost = transition_and_cost(state.copy(), action, tick)
					
					ps.append(p)
					tick_costs.append(cost)
					next_states.append(next_state)

				tick_costs = np.array(tick_costs)
				ps = np.array(ps)
				
				next_states = np.array(next_states)
				J_k_1 = model.predict(next_states).reshape(-1)
				costs.append(((tick_costs + J_k_1) * ps).sum())

			idx = np.argmax(costs)
			X.extend([state] * count)
			y.extend([costs[idx]] * count)

		X = np.array(X)
		y = np.array(y)

		model = LinearRegression().fit(X, y)
		models[K] = model
		print(f"Stage {K} Model Fitted")
		K-=1


	end = time.time()

	with open(f'approximation/{K_}/linear_models.pkl', 'wb') as file:
		joblib.dump(models, file)

	###################################################################################################

	try:

		with open("timers/timer_dict.pkl", "rb") as file:
			timer_dict = joblib.load(file)
			key = timer_dict.get("approximation", None)
			if not key:
				timer_dict["approximation"] = {}
			timer_dict["approximation"][K_] = end - start

		with open("timers/timer_dict.pkl", "wb") as file:
			joblib.dump(timer_dict, file)

	except Exception as e:

		print(e)
		with open("timers/timer_dict.pkl", "wb") as file:
			timer_dict = {"approximation" : {}}
			timer_dict["approximation"][K_] = end - start
			joblib.dump(timer_dict, file)

if __name__ == '__main__':

	approx(int(args.K))