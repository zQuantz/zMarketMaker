from dp import get_possible_actions, terminal_cost, transition_and_cost
from keras.layers import Dense, Activation, Flatten, Input
from keras.models import Sequential, Model
from const import TICK_LIMIT, TICKS
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import sys, os
import pickle

###################################################################################################

K = 50

coocc = pd.read_csv('data/cooccurrence_matrix.csv', index_col = 0)
coocc = (coocc.T / coocc.sum(axis=1)).T.values

with open(f'states/states_{K}_10000.pickle', 'rb') as file:
	states = pickle.load(file)

###################################################################################################

def network(input_shape):

	model = Sequential()
	model.add(Dense(9, input_shape=input_shape))
	model.add(Activation('linear'))
	model.add(Dense(27))
	model.add(Activation('linear'))
	model.add(Dense(54))
	model.add(Activation('linear'))
	model.add(Dense(27))
	model.add(Activation('linear'))
	model.add(Dense(3))
	model.add(Activation('linear'))
	model.add(Dense(1))
	model.add(Activation('linear'))
	print(model.summary())

	return model

def approx():

	np.random.seed(72)
	model = network((3,))
	adam = Adam(learning_rate=0.0001)
	model.compile(loss="mean_squared_error", optimizer=adam)

	K = 6 - 1
	X, y = [], []
	for state, count in states[K].items():
		
		state, _ = state
		cost = terminal_cost(state)
		
		state = list(state)
		count = int(count / 4)
		
		X.extend([state]*count)
		y.extend([cost]*count)

	X, y = np.array(X), np.array(y)
	loss = model.train_on_batch(X, y)
	print("Terminal State Loss", loss)

	while(K > 0):

		X = []
		y = []

		for i, (state, count) in enumerate(states[K].items()):

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

		loss = model.train_on_batch(X, y)
		print(f"Stage {K} Loss", loss)
		model.save(f"keras_models_3/{K}")

		K-=1

if __name__ == '__main__':

	approx()