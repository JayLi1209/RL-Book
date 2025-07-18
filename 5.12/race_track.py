import numpy as np
from gymnasium import Env
import pygame


class RaceTrack(Env):

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}

    def __init__(self, track_map:str, render_mode:str=None, size:int=20):
        self.size = size

        assert track_map in ['a', 'b']
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        filename = 'track_a.npy' if track_map == 'a' else 'track_b.npy'
        with open('tracks/' + filename, 'rb') as f:
            self.track_map = np.load(f)

        # some pygame parameters
        self.window_size = self.track_map.shape
        self.window_size = (self.window_size[1] * self.size, self.window_size[0] * self.size)
        self.window = None # window for pygame rendering
        self.clock = None # clock for pygame ticks
        self.truncated = False

        self.start_states = np.dstack(np.where(self.track_map == 0))[0]

        # Observation space
        self.nS = (*self.track_map.shape, 5, 5)
        self.nA = 9
        self.state = None
        self.speed = None

        self._action_to_accleration = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (1, -1),
            7: (1, 0),
            8: (1, 1)
        }

    def _get_obs(self):
        return (*self.state, *self.speed)
    
    def _get_info(self):
        return None
    
    # Assumed finish line is only one col
    def _check_finish(self):
        finish_states = np.where(self.track_map == 2)
        rows = finish_states[0]
        col = finish_states[1][0]
        if self.state[0] in rows and self.state[1] >= col:
            return True
        return False
    
    # Bresenham's line algorithm to get all points between two positions
    def _get_line_points(self, start, end):
        x0, y0 = start.astype(int)
        x1, y1 = end.astype(int)
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points
    
        
    # Check if it runs out of track
    def _check_out_of_track(self, start, end):
        """Check if any point along the path is off-track"""
        points = self._get_line_points(start, end)
        for (r, c) in points:
            # Check if position is outside track boundaries
            if r < 0 or r >= self.track_map.shape[0] or c < 0 or c >= self.track_map.shape[1]:
                return True
            # Check if position is gravel (off-track)
            if self.track_map[r, c] == -1:
                return True
        return False
    
    def reset(self):
        start_idx = np.random.choice(self.start_states.shape[0])
        self.state = self.start_states[start_idx]
        self.speed = (0, 0)

        if self.render_mode == 'human':
            self.render(self.render_mode)
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        new_state = np.copy(self.state)
        y_act, x_act = self._action_to_accleration[action]

        tmp_y_acc = self.speed[0] + y_act
        tmp_x_acc = self.speed[1] + x_act

        if tmp_y_acc < -4: tmp_y_acc = -4
        if tmp_y_acc > 0: tmp_y_acc = 0
        if tmp_x_acc < -4: tmp_x_acc = -4
        if tmp_x_acc > 4: tmp_x_acc = 4

        new_state[0] += tmp_y_acc
        new_state[1] += tmp_x_acc

        terminated = False
        if self._check_finish():
            terminated = True
        elif self._check_out_of_track(self.state, new_state):
            self.reset()
        else:
            self.state = new_state
            self.speed = (tmp_y_acc, tmp_x_acc)

        if self.render_mode == 'human':
            self.render(self.render_mode)

        return self._get_obs(), -1, terminated, self.truncated

    def render(self, mode):

        if self.window is None:
            pygame.init()
            pygame.display.set_caption('Race Track')
            if mode == 'human':
                self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        rows, cols = self.track_map.shape
        self.window.fill((255, 255, 255))
        
        # Draw the map
        for row in range(rows):
            for col in range(cols):
                cell_val = self.track_map[row, col]
                # Draw finishing cells
                if cell_val == 2:
                    fill = (235, 52, 52)
                    pygame.draw.rect(self.window, fill, (col * self.size, row * self.size, self.size, self.size), 0)
                # Draw starting cells
                elif cell_val == 0:
                    fill = (61, 227, 144)            
                    pygame.draw.rect(self.window, fill, (col * self.size, row * self.size, self.size, self.size), 0)

                color =(120, 120, 120)
                # Draw gravels
                if cell_val == -1:
                    color = (255, 255, 255)
                # Draw race track
                elif cell_val == 1:
                    color = (160, 160, 160)
                
                pygame.draw.rect(self.window, color, (col * self.size, row * self.size, self.size, self.size), 1)
        
        # Draw the car
        pygame.draw.rect(self.window, (86, 61, 227), (self.state[1] * self.size, self.state[0] * self.size, self.size, self.size), 0)

        if mode == "human":
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.window = None
                    pygame.quit()
                    self.truncated = True
            self.clock.tick(self.metadata['render_fps'])



if __name__ == '__main__':

    render_mode = 'human'
    # render_mode = None
    env = RaceTrack('a', render_mode=render_mode, size=20)
    env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = np.random.choice(env.nA)
        observation, reward, terminated, truncated = env.step(action)
        total_reward += reward
        if terminated: print(observation, reward, terminated, total_reward)