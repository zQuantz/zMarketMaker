from itertools import product
from const import TICK_LIMIT
import pandas as pd
import numpy as np
import sys, os

###################################################################################################

MAX_POSITION = 3
MAX_UNREALIZED = 20

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
        -MAX_PRICE, ## Bid price
        1, ## Bid volume
        MAX_PRICE, ## Ask price
        1,  ## Ask volume
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
    ## Action Step
    state[2] = action[0] ## Assign new bid order price
    state[3] = action[1] ## Assign new bid order quantity
    state[4] = action[2] ## Assign new ask order price
    state[5] = action[3] ## Assign new ask order quantity
    
    ## Unrealized PnL Step
    cost = 0
    state[1] += state[0] * randomness # Keep track of unrealized PnL based on net long or short position.

    ## Apply the limit to the unrealized PnL.
    sign = np.sign(state[1]) 
    state[1] = min(MAX_UNREALIZED, abs(state[1])) * np.sign(state[1])
    print("New Unrealized PnL", state[1])
    
    ## Randomness Step
    ## Jump in price => Sale by the market maker to market participant who bought at a higher price.
    if randomness >= 0:
        print("Positive Jump", randomness)
        print("Distance", state[4], "Jump", randomness)
        if state[4] == MAX_PRICE + 1:
        	print("Ask Ban. Setting Order to Max Again")
        elif state[4] > randomness: ## Order Not Filled
            print("Ask Order Not Filled. Adjusting to new Distance.", state[4] - randomness)
            state[4] = state[4] - randomness # Order not filled adjust the state
        elif state[4] <= randomness: ## Order Filled
            print("Ask Order Filled. Setting new Order w/ Max Distance.", MAX_PRICE)
            state[4] = MAX_PRICE # Order filled, set the maximum order limit from the current known price.
            
            if state[0] <= 0:
                state[0] -= 1
                print("Adding to Short Position", state[0])
            else:
                print("Closing 1 Long Unit")
                p = (1 / abs(state[0]))
                print("Proportion of Position Closed", p)
                cost = p * state[1]
                print("Proportion Of Position Maintained", 1 - p, "Cost Incurred", cost)
                
                state[1] = (1 - p) * state[1]
                print("New Unrealized PnL", state[1])
                state[0] -= 1
                print("New Net Position")
        
        ## Fixing the bid distance if it falls outside of the bounds
        if state[2] - randomness < -MAX_PRICE and state[2] != -MAX_PRICE - 1:
            print("Bid Distance", state[2] - randomness)
            state[2] = -MAX_PRICE
            print("Bid Distance Now", state[2])
    
    ## Drop in price => Purchase by the market maker to market participant who 
    if randomness <= 0:
        print("Negative Jump", randomness)
        if state[2] == -MAX_PRICE - 1:
        	print("Bid Ban. Setting Order to Max Again")
        elif state[2] < randomness:
            print("Bid Order Not Filled. Adjusting to new Distance.", state[2] - randomness)
            state[2] = state[2] - randomness ## Order not filled. Adjust new distance from price.
        elif state[2] >= randomness:
            print("Bid Order Filled. Setting new Order w/ Max Distance.", -MAX_PRICE)
            state[2] = -MAX_PRICE ## Order filled. Setting maximum distance from price.
            
            if state[0] >= 0:
                state[0] += 1
            else:
                print("Closing 1 Short Unit")
                p = (1 / abs(state[0]))
                print("Proportion of Position Closed", p)
                cost = p * state[1]
                print("Proportion Of Position Maintained", 1 - p, "Cost Incurred", cost)
                
                state[1] = (1 - p) * state[1]
                print("New Unrealized PnL", state[1])
                state[0] += 1
                print("New Net Position", state[0])
            
        ## Fixing the ask distance if it falls outside of the bounds
        if state[4] - randomness > MAX_PRICE and state[4] != MAX_PRICE + 1:
            print("Ask Distance", state[4] - randomness)        
            state[4] = MAX_PRICE
            print("Ask Distance Now", state[4])

    state[1] = int(state[1]) ## Reduce the number of states
    state[-1] = randomness ## Log the jump
    
    return state, cost

###################################################################################################
