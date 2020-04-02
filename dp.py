from itertools import product
from const import TICK_LIMIT
import pandas as pd
import numpy as np
import sys, os

###################################################################################################

MAX_POSITION = 3
MAX_UNREALIZED = 30

MAX_VOLUME = 1
MIN_VOLUME = 1
VOLUME = np.arange(MIN_VOLUME, MAX_VOLUME + 1)

MAX_PRICE = TICK_LIMIT
MIN_PRICE = 0
PRICE = np.arange(MIN_PRICE, MAX_PRICE + 1)

###################################################################################################

bid_price_actions = -PRICE
bid_volume_actions = VOLUME

ask_price_actions = PRICE
ask_volume_actions = VOLUME

actions = list(product(bid_price_actions, bid_volume_actions, ask_price_actions, ask_volume_actions))
ACTIONS = [action for action in actions if not (action[0] == 0 and action[2] == 0)]

BID_BAN = [(-TICK_LIMIT-1, 1, action[2], 1) for action in ACTIONS]
BID_BAN = list(set(BID_BAN))

ASK_BAN = [(action[0], 1, TICK_LIMIT + 1, 1) for action in ACTIONS]
ASK_BAN = list(set(ASK_BAN))

###################################################################################################

# def print(*args):
# 	pass

def get_initial_state():
	return [
		0, ## Inventory of stock
		0, ## Unrealized PnL
		0 ## Previous Change
	]

def get_possible_actions(state):

	if abs(state[0]) < MAX_POSITION:
		return ACTIONS
	elif state[0] > 0:
		return BID_BAN
	else:
		return ASK_BAN

def terminal_cost(state): ## Realize the unrealized profit at the end of the period
	return state[1]

def transition_and_cost(state, action, randomness):
	
	print("State in:", state)
	print("Random Jump:", randomness)
	print("Ask Price:", action[2])
	print("Bid Price:", action[0])
	
	## Unrealized PnL Step
	cost = 0
	state[1] += state[0] * randomness # Keep track of unrealized PnL based on net long or short position.

	## Apply the limit to the unrealized PnL.
	sign = np.sign(state[1]) 
	state[1] = min(MAX_UNREALIZED, abs(state[1])) * np.sign(state[1])
	print("New Unrealized PnL", state[1])

	## Randomness Step
	## Jump in Price => Leads to a possible sale.
	if randomness >= 0:

		if action[2] <= randomness: ## Trade has occurred

			print("Ask order filled.")

			if state[0] <= 0: ## Adding to position
				print("Adding to Short Position.")
				state[0] -= 1
			elif state[0] > 0: ## Closing a Long Position
				print("Closing 1 Long Position")
				p = 1 / abs(state[0])
				cost = p * state[1]
				state[1] = (1 - p) * state[1]
				state[0] -= 1
				print("Realized", cost)

	if randomness <= 0:

		if action[0] >= randomness:

			print("Bid Order filled.")

			if state[0] >= 0:
				print("Adding to Long Position.")
				state[0] += 1
			elif state[0] < 0:
				print("Closing 1 Short Position.")
				p = 1 / abs(state[0])
				cost = p * state[1]
				state[1] = (1 - p) * state[1]
				state[0] += 1
				print("Realized", cost)

	print("")
	state[1] = int(state[1]) ## Reduce the number of states
	state[-1] = randomness ## Log the jump
	
	return state, cost

###################################################################################################
