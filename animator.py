from dp import get_initial_state, get_possible_actions, transition_and_cost
from const import TICK_LIMIT, TICKS
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
np.random.seed(3)

###################################################################################################

PATHS = np.load('data/sample_paths.npy')
LENGTH = 200
PATHS = PATHS[:, :LENGTH]

nn = load_model("keras_models_3/1")

coocc = pd.read_csv('data/cooccurrence_matrix.csv', index_col = 0)
coocc = (coocc.T / coocc.sum(axis=1)).T.values

###################################################################################################

def get_exact_method_action(k, state):
	with open(f'exact_method_200/U_{k}.pickle', 'rb') as file:
		U = pickle.load(file)
	return U[tuple(state)]

def get_random_action(k, state):
	actions = get_possible_actions(state)
	return actions[np.random.randint(0, len(actions))]

def get_parametrized_action(k, state):

	costs = []
	actions = get_possible_actions(state)
	for action in actions:
		
		avg_cost = 0
		
		for tick in TICKS:
			
			p = coocc[state[-1] + TICK_LIMIT, tick + TICK_LIMIT]
			next_state, cost = transition_and_cost(state.copy(), action, tick)
			avg_cost += p * (cost + nn.predict(np.array([next_state]))[0][0])
		
		costs.append(avg_cost)
	
	idx = np.argmax(costs)
	return actions[idx]

def rollout_policy():

	with open('rollout/0_200.pickle', 'rb') as file:
		obj = pickle.load(file)

	bid_prices = []
	bid_volumes = []

	ask_prices = []
	ask_volumes = []
	 
	prices = []
	cprices = []

	unrealized_pnls = []
	realized_pnls = []
	net_position = []

	path = PATHS[12, :200]

	for i, (state, action, cost, jump) in enumerate(zip(*obj.values(), path)):

		print("Step", i)

		state, next_state = state

		## Log the action pre-jump
		prices.append(0)
		cprices = np.cumsum(prices)
			
		bid_prices.append(action[0] + cprices[-1])
		bid_volumes.append(action[1])

		ask_prices.append(action[2] + cprices[-1])
		ask_volumes.append(action[3])
		
		unrealized_pnls.append(state[1])
		realized_pnls.append(0)
		net_position.append(state[0])
		
		state = next_state
		
		prices.append(jump)
		cprices = np.cumsum(prices)
			
		bid_prices.append(action[0] + cprices[-1] - jump)
		bid_volumes.append(action[1])

		ask_prices.append(action[2] + cprices[-1] - jump)
		ask_volumes.append(action[3])

		unrealized_pnls.append(state[1])
		realized_pnls.append(cost)
		net_position.append(state[0])
		
	realized_pnls = np.cumsum(realized_pnls)
	cprices = np.cumsum(prices)

	return bid_prices, bid_volumes, ask_prices, ask_volumes, unrealized_pnls, realized_pnls, net_position, cprices

def compute_policy(get_action):

	bid_prices = []
	bid_volumes = []

	ask_prices = []
	ask_volumes = []
	 
	prices = []
	cprices = []

	unrealized_pnls = []
	realized_pnls = []
	net_position = []

	state = get_initial_state()

	jumps = PATHS[0]

	for i in range(LENGTH):
			
		print("Step", i)

		action = get_action(i, state)
		
		## Log the action pre-jump
		prices.append(0)
		cprices = np.cumsum(prices)
			
		bid_prices.append(action[0] + cprices[-1])
		bid_volumes.append(action[1])

		ask_prices.append(action[2] + cprices[-1])
		ask_volumes.append(action[3])
		
		unrealized_pnls.append(state[1])
		realized_pnls.append(0)
		net_position.append(state[0])
		
		jump = jumps[i]
		state, cost = transition_and_cost(state, action, jump)
		
		## Log the result post-action
		prices.append(jump)
		cprices = np.cumsum(prices)
			
		bid_prices.append(action[0] + cprices[-1] - jump)
		bid_volumes.append(action[1])

		ask_prices.append(action[2] + cprices[-1] - jump)
		ask_volumes.append(action[3])

		unrealized_pnls.append(state[1])
		realized_pnls.append(cost)
		net_position.append(state[0])
		
	realized_pnls = np.cumsum(realized_pnls)
	cprices = np.cumsum(prices)

	return bid_prices, bid_volumes, ask_prices, ask_volumes, unrealized_pnls, realized_pnls, net_position, cprices

def z_animate():

	print("Animating")
	arrs = compute_policy(get_parametrized_action)
	# arrs = rollout_policy()
	bid_prices, bid_volumes, ask_prices, ask_volumes, unrealized_pnls, realized_pnls, net_position, cprices = arrs

	for idx in range(1, len(cprices)):

		print("Plot", idx)

		################################################################################################################################
		min_ = min(bid_prices[:idx]) - 3
		max_ = max(ask_prices[:idx]) + 3

		f, ax = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={"width_ratios" : [3, 1]})
		ax[0, 0].plot(cprices[:idx], color="black", label="price")
		ax[0, 0].axhline(ask_prices[idx-1], color="r", label="ask")
		ax[0, 0].axhline(bid_prices[idx-1], color="g", label="bid")
		ax[0, 0].legend()
		ax[0, 0].set_ylim(min_, max_)

		modifier = "Action Step" if idx % 2 == 0 else "Jump Step"
		ax[0, 0].set_title(f"Market Maker Actions & Price - {modifier}")

		################################################################################################################################
		min_ = min(net_position[:idx]) - 2
		max_ = max(net_position[:idx]) + 2
		net_pos = net_position[idx-1]

		color = "r" if net_pos <= 0 else "g"
		label = "Net Long" if net_pos > 0 else "Net Short" if net_pos < 0 else ""
		ax[0, 1].bar([0, 1, 2, 3, 4, 5, 6], [0, 0, 0, net_pos, 0, 0, 0], color=color, label=label)
		ax[0, 1].set_title("Net Open Position")
		ax[0, 1].set_ylim(min_, max_)

		################################################################################################################################
		rpnl = realized_pnls[:idx]
		min_ = min(rpnl) - 5
		max_ = max(rpnl) + 5

		idc = np.arange(len(rpnl))
		ax[1, 0].bar(idc[rpnl>=0], rpnl[rpnl>=0], color="g")
		ax[1, 0].bar(idc[rpnl<0], rpnl[rpnl<0], color="r")
		ax[1, 0].set_title("Realized Profit & Loss")
		ax[1, 0].set_ylim(min_, max_)

		################################################################################################################################
		upnl = unrealized_pnls[:idx]
		min_ = min(upnl) - 5
		max_ = max(upnl) + 5
		upnl = upnl[-1]

		color = "r" if upnl <= 0 else "g"
		ax[1, 1].bar([0, 1, 2, 3, 4, 5, 6], [0, 0, 0, upnl, 0, 0, 0], color=color)
		ax[1, 1].set_title("Unrealized Profit & Loss")
		ax[1, 1].set_ylim(min_, max_)

		f.savefig(f"plots/{idx}.png")
		plt.close()

		################################################################################################################################

if __name__ == '__main__':

	z_animate()