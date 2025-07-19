# This defines the basic Windy Gridworld class

# What do I need for an agent to do randomized walk?
import numpy as np
from gymnasium import Env
import pygame

class GridWorld(Env):

    def __init__(self, mapping, sto = False):
        grid = np.zeros((7,10))
        upwards = [0,0,0,1,1,1,2,2,1,0]

        start = (3,0)
        goal = (3,7)
        cur_loc = start

        self._action_to_direction = mapping

    def _get_obs(self):
        return {"agent location": self.cur_loc}
    
    def _get_info(self):
        return None
    
    def reset(self):
        cur_loc = self.start

    def step(self, action):
        pass
    
