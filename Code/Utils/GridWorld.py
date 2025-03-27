from Utils.Parameters import *
import numpy as np


class GridWorld:
    """GridWorld 4x4 Environment."""

    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        """Resets the agent to the starting position."""
        self.agent_position = (0, 0)
        self.goal_position = (3, 3)
        self.obstacle_position = (1, 1)
        return self.get_state()

    def get_state(self):
        state = np.zeros((GRID_SIZE, GRID_SIZE))
        state[self.agent_position] = 1
        return state.flatten()

    def step(self, action: int):
        x, y = self.agent_position
        dx, dy = MOVES[action]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            self.agent_position = (new_x, new_y)

        if self.agent_position == self.goal_position:
            return self.get_state(), 10, True

        elif self.agent_position == self.obstacle_position:
            return self.get_state(), -5, False

        else:
            return self.get_state(), -1, False
