# Defines the basic Windy Gridworld class
import numpy as np
from gymnasium import Env
import pygame

class MazeWorld(Env):

    def __init__(self, mapping, sto = False):
        
        # pygame setup
        self.size = 30  # Size of each grid cell in pixels
        self.window_size = (9 * self.size, 6 * self.size)  # 10 columns, 7 rows
        self.window = None
        self.clock = None
        
        self.grid = np.zeros((6,9))
        self.nS = (6,9)
        self.nA = (len(mapping))

        self.start = (5,3)
        self.goal = (0,8)
        self.cur_loc = self.start
        self.blocks = [(3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7)]

        self._action_to_direction = mapping

    def _get_obs(self):
        return tuple(self.cur_loc)
    
    def _get_info(self):
        return None
    
    def reset(self):
        self.cur_loc = self.start
        return self._get_obs(), self._get_info()

    def step(self, action, change=False):
        if change:
            self.blocks = [(3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8)]
        direction = self._action_to_direction[action]
        if tuple(self.cur_loc + direction) not in self.blocks:
            self.cur_loc += direction
        self.cur_loc = np.clip(self.cur_loc, [0, 0], [5, 8])

        terminated = True if all(self.cur_loc == self.goal) else False
        reward = 1 if terminated else -1
        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption('Maze')
        
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
                elif (row, col) in self.blocks:
                    pygame.draw.rect(self.window, (150, 150, 150), rect)  # Grey
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
    
    agent = MazeWorld(mapping)
    observation, info = agent.reset()
    
    # policy: take random actions
    terminated = False
    while not terminated:
        action = np.random.randint(0, 4)
        words = ['Down', 'Right', 'Up', 'Left']
        observation, reward, terminated, truncated, info = agent.step(action)
        print("Turned: ", words[action], ", now at: ", observation)
        agent.render()
        pygame.time.delay(10)  # Slow down for visualization
        
    pygame.quit()

