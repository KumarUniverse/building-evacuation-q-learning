# By Akash Kumar and Dr. Burns

import copy
import time
import random

class Maze():
    """A pathfinding problem."""

    possible_directions = ['N', 'S', 'E', 'W']
    dirs_to_moves = {'N':(-1,0), 'S':(1,0), 'E':(0,1), 'W':(0,-1)}
    moves_to_dirs = {value:key for key, value in dirs_to_moves.items()}

    def __init__(self, grid, location):
        """Instances differ by their current agent locations."""
        self.grid = grid
        self.location = location  # Tuple containing (x, y) coordinates.

    def display(self):
        """Print the maze, marking the current agent location."""
        for r in range(len(self.grid)):
            for c in range(len(self.grid[r])):
                if (r, c) == self.location:
                    print('\033[96m*\x1b[0m', end=' ')  # print a blue *
                else:
                    print(self.grid[r][c], end=' ')     # prints a space or wall
            print()
        print()

    def moves(self):
        """Return a list of possible moves given the current agent location."""
        move_list = []
        for direction in Maze.possible_directions:
            move = Maze.dirs_to_moves[direction]
            if (0 <= (self.location[0]+move[0]) < len(self.grid) and
                    0 <= (self.location[1]+move[1]) < len(self.grid[0]) and
                    self.grid[self.location[0]+move[0]][self.location[1]+move[1]] != 'X'):
                move_list.append(move)

        return move_list

    def directions(self):
        """Return a list of possible directions given the current agent location."""
        direction_list = []
        for direction in Maze.possible_directions:
            move = Maze.dirs_to_moves[direction]
            if (0 <= (self.location[0]+move[0]) < len(self.grid) and
                    0 <= (self.location[1]+move[1]) < len(self.grid[0]) and
                    self.grid[self.location[0]+move[0]][self.location[1]+move[1]] != 'X'):
                direction_list.append(direction)

        return direction_list

    def neighbor(self, move):
        """
        Return another Maze instance with a move made.
        The move is represented as a direction (N, S, E, W).
        """
        move_coord = Maze.dirs_to_moves[move]
        return Maze(self.grid, (self.location[0]+move_coord[0], self.location[1]+move_coord[1]))


class QAgent():
    """Solves a maze with Q-learning."""

    def __init__(self, maze):
        """Create the Q table based on the dimensions and borders of the grid."""
        self.alpha = 0.1  # learning rate.
        self.gamma = 0.8  # discount factor.
        self.epsilon = 1  # randomness factor. e=0 makes the agent greedy.
        self.maze = copy.deepcopy(maze)
        self.grid = copy.deepcopy(maze.grid)
        self.reward_table = [
            [None for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]
        self.qtable = [
            [0 for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]

        # Initialize the reward table.
        for i in range(len(self.reward_table)):
            for j in range(len(self.reward_table[0])):
                if self.grid[i][j] == ' ':
                    self.reward_table[i][j] = -0.04  # -0.01
                elif self.grid[i][j] == 'G':
                    self.reward_table[i][j] = 1
                elif self.grid[i][j] == 'E':
                    self.reward_table[i][j] = -1

    def r(self, s, a):
        """
        A reward function R(s,a) that gives the agent a reward for taking
        action a in state s.
        """
        return self.reward_table[s[0]+a[0]][s[1]+a[1]]

    def q(self, s, a):
        """The Q function Q(s,a) gives the quality of taking action a in state s."""
        # The Q value of the current state is based on the max Q value of the next state.
        next_state_max_q = max([self.qtable[s[0]+x][s[1]+y] for (x,y) in self.maze.moves()])
        self.qtable[s[0]+a[0]][s[1]+a[1]] = (self.qtable[s[0]+a[0]][s[1]+a[1]]
            + self.alpha * (self.r(s,a) + self.gamma * next_state_max_q
                - self.qtable[s[0]+a[0]][s[1]+a[1]]))

        return self.qtable[s[0]+a[0]][s[1]+a[1]]

    def qlearn(self, num_simulations):
        """Used to make the Q agent learn about its environment."""
        initial_maze_loc = self.maze.location
        for i in range(num_simulations):
            curr_coord = self.maze.location
            new_epsilon = round(1 - (i+1)/num_simulations, 2)
            self.epsilon = new_epsilon if new_epsilon > 0 else self.epsilon

            while (self.grid[curr_coord[0]][curr_coord[1]] != 'G' and
                self.grid[curr_coord[0]][curr_coord[1]] != 'E'):
                rand_num = round(random.random(), 2)

                move = (0,0)
                if rand_num < self.epsilon:  # exploration
                    move = random.choice(self.maze.moves())
                else:                        # exploitation
                    possible_moves = self.maze.moves()
                    best_next_move_q = 0
                    for pmove in possible_moves:
                        if (self.qtable[curr_coord[0]+pmove[0]][curr_coord[1]+pmove[1]] >=
                            best_next_move_q):
                            move = pmove
                            best_next_move_q = (
                                self.qtable[curr_coord[0]+pmove[0]][curr_coord[1]+pmove[1]])

                self.q(curr_coord, move)
                curr_coord = (curr_coord[0]+move[0], curr_coord[1]+move[1])
                self.maze.location = curr_coord
            self.maze.location = initial_maze_loc
            #print(f"Simulation {i+1} of {num_simulations} complete.")

    def solve_maze(self):
        """
        Return an ordered list of moves the agent takes to solve the maze.
        This method assumes that the Q agent has finished learning about its environment.
        """
        initial_maze_loc = self.maze.location
        curr_coord = initial_maze_loc
        solution_path_directions = []
        #print("in solve_maze:")

        # The agent always chooses the next location with the highest Q value.
        # With this strategy, the agent aims to reach the goal using the
        # most optimal path possible.
        while (self.grid[curr_coord[0]][curr_coord[1]] != 'G' and
            self.grid[curr_coord[0]][curr_coord[1]] != 'E'):
            possible_moves = self.maze.moves()

            # Find the next best move.
            best_next_move = (0,0)
            best_next_move_q = float('-inf')
            for move in possible_moves:
                if self.qtable[curr_coord[0]+move[0]][curr_coord[1]+move[1]] >= best_next_move_q:
                    best_next_move = move
                    best_next_move_q = self.qtable[curr_coord[0]+move[0]][curr_coord[1]+move[1]]

            direction = self.maze.moves_to_dirs[best_next_move]
            solution_path_directions.append(direction)
            curr_coord = (curr_coord[0]+best_next_move[0], curr_coord[1]+best_next_move[1])
            self.maze.location = curr_coord
        self.maze.location = initial_maze_loc  # reset maze location to initial coord.

        return solution_path_directions

    def reset_q_table(self):
        """Reset all Q values in the Q table to 0."""
        for i in range(len(self.qtable)):
            for j in range(len(self.qtable[0])):
                self.qtable[i][j] = 0
        self.epsilon = 1  # Reset espilon as well.

    def get_qtable_str(self):
        """Used to represent the Q table as a readable string."""
        output = "[\n"
        for row in self.qtable:
            output += "\t" + str([round(x,2) for x in row]) + ",\n"
        output += "]\n"

        return output


def main():
    """Create a maze, solve it with Q-learning, and console-animate."""

    # The X's represent the boundaries of the maze.
    # Reaching state G results in a reward of +1.
    # Reaching state E results in a reward of -1.
    grid = [  # 4 x 3 maze.
        "XXXXXX",
        "X   GX",
        "X X EX",
        "X    X",
        "XXXXXX"
    ]

    grid2 = [  # 10 x 8 maze.
        "XXXXXXXXXXXX",
        "X X        X",
        "X X XXXXXX X",
        "X         EX",
        "XX XXXXXX  X",
        "X  X X     X",
        "X XX XGXX  X",
        "X    XX X  X",
        "XXXX      XX",
        "XXXXXXXXXXXX"
    ]

    maze = Maze(grid, (2, 1))
    maze.display()

    agent = QAgent(maze)
    agent.qlearn(250)
    path = agent.solve_maze()

    while path:
        move = path.pop(0)
        maze = maze.neighbor(move)
        time.sleep(0.50)
        maze.display()

    print("path: " + str(path))
    print("Q table:")
    print(agent.get_qtable_str())

if __name__ == '__main__':
    main()
