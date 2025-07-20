# This defines the basic Windy Gridworld class

# What do I need for an agent to do randomized walk?
import numpy as np
from gymnasium import Env
import pygame

class GridWorld(Env):

    def __init__(self, mapping, sto = False):

        self.size = 30  # Size of each grid cell in pixels
        self.window_size = (10 * self.size, 7 * self.size)  # 10 columns, 7 rows
        self.window = None
        self.clock = None
        
        self.grid = np.zeros((7,10))
        self.nS = (7,10)
        self.nA = (len(mapping))
        self.upward_dir = [[0,0],[0,0],[0,0],[-1,0],[-1,0],[-1,0],[-2,0],[-2,0],[-1,0],[0,0]]

        self.start = (3,0)
        self.goal = (3,7)
        self.cur_loc = self.start

        self._action_to_direction = mapping

    def _get_obs(self):
        return tuple(self.cur_loc)
    
    def _get_info(self):
        return None
    
    def reset(self):
        self.cur_loc = self.start
        return self._get_obs(), self._get_info()

    # def step(self, action):
    #     direction = self._action_to_direction[action]
    #     up = self.upward_dir[self.cur_loc[1]]
    #     self.cur_loc += direction
    #     self.cur_loc = np.clip(self.cur_loc + up, [0, 0], [6, 9])

    #     terminated = True if all(self.cur_loc == self.goal) else False
    #     reward = 0 if terminated else -1
    #     return self._get_obs(), reward, terminated, False, self._get_info()

    def step(self, action):
        direction = self._action_to_direction[action]
        new_loc = np.clip(self.cur_loc + direction, [0, 0], [6, 9])
        # Apply wind from the NEW column
        new_col = new_loc[1]
        up = self.upward_dir[new_col]  # Now using destination column
        new_loc = np.clip(new_loc + up, [0, 0], [6, 9])
        self.cur_loc = new_loc
        # Termination check remains the same
        terminated = all(self.cur_loc == self.goal)
        reward = 0 if terminated else -1
        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption('Windy Gridworld')
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Fill background
        self.window.fill((240, 240, 240))
        
        # Draw grid cells
        for row in range(7):
            for col in range(10):
                rect = pygame.Rect(col * self.size, row * self.size, self.size, self.size)
                
                # Color start position
                if (row, col) == self.start:
                    pygame.draw.rect(self.window, (100, 200, 100), rect)  # Green
                # Color goal position
                elif (row, col) == self.goal:
                    pygame.draw.rect(self.window, (200, 100, 100), rect)  # Red
                # Regular cell
                else:
                    pygame.draw.rect(self.window, (255, 255, 255), rect)
                
                # Draw grid borders
                pygame.draw.rect(self.window, (180, 180, 180), rect, 1)
        
        # Draw agent (as a blue circle)
        agent_x = self.cur_loc[1] * self.size + self.size // 2
        agent_y = self.cur_loc[0] * self.size + self.size // 2
        pygame.draw.circle(self.window, (50, 50, 200), (agent_x, agent_y), self.size // 3)
        
        pygame.display.flip()
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.window = None
                return
        self.clock.tick(120)  # Control rendering speed


if __name__ == "__main__":
    mapping = {
        0: np.array([1, 0]),   # Down
        1: np.array([0, 1]),   # Right
        2: np.array([-1, 0]),  # Up
        3: np.array([0, -1])   # Left
    }
    
    agent = GridWorld(mapping)
    observation, info = agent.reset()
    
    # policy: take random actions
    terminated = False
    while not terminated:
        action = np.random.randint(0, 4)
        words = ['Down', 'Right', 'Up', 'Left']
        observation, reward, terminated, truncated, info = agent.step(action)
        print("Turned: ", words[action], ", now at: ", observation["agent location"])
        agent.render()
        pygame.time.delay(10)  # Slow down for visualization
        
    pygame.quit()

