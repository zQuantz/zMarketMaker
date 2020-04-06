from dp import get_possible_actions, terminal_cost, transition_and_cost
from sklearn.linear_model import LinearRegression
from const import TICK_LIMIT, TICKS
import pandas as pd
import numpy as np
import sys, os
import pickle
import joblib

###################################################################################################

K = 200

coocc = pd.read_csv('data/cooccurrence_matrix.csv', index_col = 0)
coocc = (coocc.T / coocc.sum(axis=1)).T.values

with open(f'states/states_{K}_10000.pickle', 'rb') as file:
	states = pickle.load(file)

###################################################################################################

def approx():

	np.random.seed(72)
	models = {}
	K = 200

	K -= 1
	X, y = [], []
	for state, count in states[K].items():
		
		state, _ = state
		cost = 0#terminal_cost(state)
		
		state = list(state)
		count = int(count / 4)
		
		X.extend([state]*count)
		y.extend([cost]*count)

	X, y = np.array(X), np.array(y)

	model = LinearRegression()
	models[K] = model.fit(X, y)
	print(f"Stage {K} Model Fitted")

	K -= 1
	while(K > 0):

		X = []
		y = []

		for i, (state, count) in enumerate(states[K].items()):

			if K == 0:
				print(state)

			state, _ = state 
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

		model = LinearRegression()
		models[K] = model.fit(X, y)
		print(f"Stage {K} Model Fitted")

		with open('linear_models_2/simple_linear_models.pkl', 'wb') as file:
			joblib.dump(models, file)

		K-=1

if __name__ == '__main__':

	approx()