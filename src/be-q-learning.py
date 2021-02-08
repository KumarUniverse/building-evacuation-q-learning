# Building evacuation Q-learning algorithm.
# By Akash Kumar
# Tutorial: http://mnemstudio.org/path-finding-q-learning-tutorial.htm

import random
import sys

alpha = 0.1 # learning rate
gamma = 0.8 # discount factor

# Building graph represented using an adjacency list, where the nodes
# are the rooms and the edges are the connections between rooms.
# The last room represents the outside world.
building = [
    [4],     # room 0
    [3,5],   # room 1
    [3],     # room 2
    [1,2,4], # room 3
    [0,3,5], # room 4
    [1,4,5]  # room 5 (outside)
]

# Reward table.
reward = [
    # action
    [None,None,None,None,-10,None],  # state 0
    [None,None,None,-10,None,100],   # state 1
    [None,None,None,-10,None,None],  # state 2
    [None,-10,-10,None,-10,None],    # state 3
    [-10,None,None,-10,None,100],    # state 4
    [None,-10,None,None,-10,100]     # state 5
]

# Q table.
# Before learning, the agent knows nothing about the environment.
# So, all Q values are initialized to 0.
qtable = [
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0]
]

# A reward function R(s,a) that gives the agent a reward for taking
# action a in state s.
def r(s,a):
    return reward[s][a]

# Q function.
# Used to calculate Q (quality) values for state-action pairs using
# the Bellman equation and returns the new Q value.
def q(s,a):
    # The Q value of this state is based on the max Q value of the next state.
    next_state_max_q = max([qtable[a][adj_room] for adj_room in building[a]])
    qtable[s][a] = (qtable[s][a]
        + alpha * (r(s,a) + gamma * next_state_max_q - qtable[s][a]))

    return qtable[s][a]

# Used to set all Q values in the Q table to 0.
def reset_q_table():
    for i in range(len(qtable)):
        for j in range(len(qtable[0])):
            qtable[i][j] = 0

# Used to represent the Q table as a readable string.
def get_qtable_str():
    output = "[\n"
    for row in qtable:
        output += str(row) + ",\n"
    output += "]\n"

    return output

# Returns the index of the maximum element in a number list.
def get_max_index(l):
    max_i = -1
    max_num = -sys.maxsize
    for i, num in enumerate(l):
        if num > max_num:
            max_num = num
            max_i = i

    return max_i

# Used to make the agent learn about its environment for a given number of
# simulations and learn the values of the Q table.
def qlearn(num_simulations):
    total_move_count = 0
    for _ in range(num_simulations):
        start_state = random.randrange(len(building))
        end_state = 5
        while start_state != end_state:
            next_state = random.choice(building[start_state])
            q(start_state,next_state)
            start_state = next_state
            total_move_count += 1

    avg_move_count = total_move_count / num_simulations

    return (f"Completed {num_simulations} Q-learning simulations with an "
        + f"average move count of {avg_move_count}")

# Used to make the agent evacuate the building as fast as possible,
# after having learnt about its environment.
def evacuate(num_simulations):
    total_move_count = 0
    for _ in range(num_simulations):
        start_state = random.randrange(len(building))
        end_state = 5
        while start_state != end_state:
            start_state = get_max_index(qtable[start_state])
            total_move_count += 1

    avg_move_count = total_move_count / num_simulations

    return (f"Completed {num_simulations} evacuations with an average "
        + f"move count of {avg_move_count}")

if __name__ == "__main__":
    num_simulations = 100
    print(qlearn(num_simulations))
    print(evacuate(num_simulations))
    print(get_qtable_str() + "\n")
    reset_q_table()

    num_simulations = 500
    print(qlearn(num_simulations))
    print(evacuate(num_simulations) + "\n")
    reset_q_table()

    num_simulations = 1000
    print(qlearn(num_simulations))
    print(evacuate(num_simulations) + "\n")
    reset_q_table()
