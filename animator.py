from dp import get_initial_state, get_possible_actions, transition_and_cost
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(72)

###################################################################################################

LENGTH = 200

###################################################################################################

def compute_random_policy():

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

	jumps = pd.read_csv("data/ibm_t.csv").change.values
	jumps = jumps[2000:2000+LENGTH]

	for i in range(len(jumps[:LENGTH])):
			
		print("Step", i)

		actions = get_possible_actions(state)
		action = actions[np.random.randint(0, len(actions))]
		
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
			
		bid_prices.append(state[2] + cprices[-1])
		bid_volumes.append(state[3])

		ask_prices.append(state[4] + cprices[-1])
		ask_volumes.append(state[5])

		unrealized_pnls.append(state[1])
		realized_pnls.append(cost)
		net_position.append(state[0])
		
	realized_pnls = np.cumsum(realized_pnls)
	cprices = np.cumsum(prices)

	return bid_prices, bid_volumes, ask_prices, ask_volumes, unrealized_pnls, realized_pnls, net_position, cprices

def z_animate():

	print("Animating")

	bid_prices, bid_volumes, ask_prices, ask_volumes, unrealized_pnls, realized_pnls, net_position, cprices = compute_random_policy()

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