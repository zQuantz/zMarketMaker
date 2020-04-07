from dp import get_initial_state, get_possible_actions, transition_and_cost, terminal_cost
from const import TICK_LIMIT, TICKS
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, os
import pickle
import joblib
import imageio
np.random.seed(3)

###################################################################################################

argparser = ArgumentParser()
argparser.add_argument("K")
argparser.add_argument("path_name")
argparser.add_argument("method")
args = argparser.parse_args()

with open(f'approximation/{args.K}/linear_models.pkl', 'rb') as file:
	linear_models = joblib.load(file)

try:
	with open(f'rollout/{args.K}/{args.path_name}_policy.pkl', 'rb') as file:
		rollout = joblib.load(file)
except Exception as e:
	print(e)

with open(f'back_recursion/{args.K}/back_recursion.pkl', 'rb') as file:
	br_memory = joblib.load(file)

coocc = pd.read_csv('data/cooccurrence_matrix.csv', index_col = 0)
coocc = (coocc.T / coocc.sum(axis=1)).T.values

###################################################################################################

def get_back_recursion_action(k, state, K):
	return br_memory[k]['U'][tuple(state)]

def get_random_action(k, state):
	actions = get_possible_actions(state)
	return actions[np.random.randint(0, len(actions))]

def get_linear_parametrized_action(k, state, K):

	costs = []
	actions = get_possible_actions(state)
	for action in actions:
		
		avg_cost = 0
		
		for tick in TICKS:
			
			p = coocc[state[-1] + TICK_LIMIT, tick + TICK_LIMIT]
			next_state, cost = transition_and_cost(state.copy(), action, tick)
			X = np.array([next_state])
			pred = linear_models[k+1].predict(X)[0]
			avg_cost += p * (cost + pred)
		
		costs.append(avg_cost)
	
	idx = np.argmax(costs)
	return actions[idx]

###################################################################################################

def rollout_policy(K, path_name):

	bid_prices = []
	bid_volumes = []

	ask_prices = []
	ask_volumes = []
	 
	prices = []
	cprices = []

	unrealized_pnls = []
	realized_pnls = []
	net_position = []

	path = np.load(f"paths/{K}/{path_name}.npy")
	state = get_initial_state()
	for i, (state, action, cost, jump) in enumerate(zip(*rollout.values(), path)):

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

def compute_policy(get_action, K, path_name):

	bid_prices = []
	bid_volumes = []

	ask_prices = []
	ask_volumes = []
	 
	prices = []
	cprices = []

	unrealized_pnls = []
	realized_pnls = []
	net_position = []

	path = np.load(f"paths/{K}/{path_name}.npy")
	state = get_initial_state()
	for i in range(K):
			
		print("Step", i)

		action = get_action(i, state, K)
		
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
		
		tick = path[i]
		state, cost = transition_and_cost(state, action, tick)
		
		## Log the result post-action
		prices.append(tick)
		cprices = np.cumsum(prices)
			
		bid_prices.append(action[0] + cprices[-1] - tick)
		bid_volumes.append(action[1])

		ask_prices.append(action[2] + cprices[-1] - tick)
		ask_volumes.append(action[3])

		unrealized_pnls.append(state[1])
		realized_pnls.append(cost)
		net_position.append(state[0])
		
	realized_pnls = np.cumsum(realized_pnls)
	cprices = np.cumsum(prices)

	return bid_prices, bid_volumes, ask_prices, ask_volumes, unrealized_pnls, realized_pnls, net_position, cprices

def z_animate(K, path_name, method):

	if method == "approx":
		arrs = compute_policy(get_back_recursion_action, K, path_name)
	elif method == "back_recursion":
		arrs = compute_policy(get_back_recursion_action, K, path_name)
	elif method == "rollout":
		arrs = rollout_policy(K, path_name)

	bid_prices, bid_volumes, ask_prices, ask_volumes, unrealized_pnls, realized_pnls, net_position, cprices = arrs
	
	print("Animating")
	for idx in range(1, len(cprices)):

		print("Plot", idx)

		################################################################################################################################
		min_ = min(bid_prices[:idx]) - 3
		max_ = max(ask_prices[:idx]) + 3

		f, ax = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={"width_ratios" : [3, 1]})
		ax[0, 0].plot(cprices[:idx], color="black", label="price")
		ax[0, 0].axhline(ask_prices[idx-1], color="r", label="ask")
		ax[0, 0].axhline(bid_prices[idx-1], color="g", label="bid")
		ax[0, 0].set_ylim(min_, max_)
		ax[0, 0].legend()

		for i in range(1, len(net_position[:idx])):
			c = net_position[i]
			p = net_position[i-1]
			if c != p:
				diff = c - p
				ax[0, 0].annotate("x", (i, cprices[i]), color="g" if diff > 0 else "r")

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

	## Delete old images
	os.system("rm plots/*")

	## Get new images
	z_animate(int(args.K), args.path_name, args.method)

	## Stitch together
	writer = imageio.get_writer(f'policies/{args.K}_{args.path_name}_{args.method}.mp4', fps=2)
	num_files = len(os.listdir("plots/"))
	for i in range(1, 1+num_files):
		writer.append_data(imageio.imread(f"plots/{i}.png"))
	writer.close()