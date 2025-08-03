import numpy as np
from gymnasium import Env
import matplotlib.pyplot as plt
import pygame


TD_ALPHAS = [0.05, 0.1, 0.15]
MC_ALPHAS = [0.01, 0.02, 0.03, 0.04]
GAMMA = 1
EPISODE = 100_000
class RandomWalk(Env):

    def __init__(self):
        self.start = 3
        self.termination = [0, 6]

        self.nA = 2
        self.nS = 7

        self.cur_loc = None
        self._action_to_direction = {
            0: -1,
            1: 1
        }

        # Rendering setup
        self.window_size = (800, 400)
        self.cell_size = 80
        self.window = None
        self.clock = None
        self.font = None
        self.values = None
        self.episode_count = 0

    def _get_obs(self):
        return self.cur_loc
    
    def _get_info(self):
        return None
    
    def reset(self):
        self.cur_loc = self.start
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        dir = self._action_to_direction[action]
        self.cur_loc += dir

        if self.cur_loc <= self.termination[0]:
            return self._get_obs(), 0, True, False, self._get_info()
        if self.cur_loc >= self.termination[1]:
            return self._get_obs(), 1, True, False, self._get_info()
        
        return self._get_obs(), 0, False, False, self._get_info()
    
    # Action selection: random
    def pick_action(self):
        num = np.random.rand()
        if num > 0.5:
            return 0
        return 1

    def render(self, values=None):        
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption('Random Walk Environment')
            self.font = pygame.font.SysFont('Arial', 24)
            self.title_font = pygame.font.SysFont('Arial', 32, bold=True)
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Fill background
        self.window.fill((240, 240, 240))
        
        # Draw title
        title = self.title_font.render(f'Random Walk - Episode {self.episode_count+1}', True, (50, 50, 50))
        self.window.blit(title, (self.window_size[0]//2 - title.get_width()//2, 10))
        
        # Draw states
        state_labels = ['0 (Term)', 'A', 'B', 'C', 'D', 'E', '6 (Term)']
        for i in range(7):
            # Draw state cell
            x_pos = 100 + i * 100
            y_pos = 100
            
            # Set color based on state type
            if i == 0 or i == 6:
                color = (220, 150, 150)  # Terminal states
            else:
                color = (200, 220, 200)  # Non-terminal states
                
            pygame.draw.rect(self.window, color, (x_pos-30, y_pos-30, 60, 60))
            pygame.draw.rect(self.window, (100, 100, 100), (x_pos-30, y_pos-30, 60, 60), 2)
            
            # Draw state label
            label = self.font.render(state_labels[i], True, (50, 50, 50))
            self.window.blit(label, (x_pos - label.get_width()//2, y_pos - label.get_height()//2))
            
            # Draw value if available
            if values is not None and i < len(values):
                val_text = self.font.render(f"{values[i]:.3f}", True, (30, 30, 120))
                self.window.blit(val_text, (x_pos - val_text.get_width()//2, y_pos + 50))
        
        # Draw agent
        if self.cur_loc is not None:
            agent_x = 100 + self.cur_loc * 100
            agent_y = 100
            pygame.draw.circle(self.window, (50, 50, 200), (agent_x, agent_y), 20)
            pygame.draw.circle(self.window, (200, 200, 255), (agent_x, agent_y), 15)
        
        # Draw value plot if available
        if values is not None and self.episode_count > 0:
            # Only show non-terminal states (A-E)
            plot_values = values[1:6]
            
            # Draw plot title
            plot_title = self.font.render('Current Value Estimates', True, (50, 50, 50))
            self.window.blit(plot_title, (100, 200))
            
            # Draw value bars
            bar_width = 60
            for i, val in enumerate(plot_values):
                x_pos = 150 + i * 100
                height = int(val * 100)
                pygame.draw.rect(self.window, (100, 150, 220), 
                               (x_pos - bar_width//2, 350 - height, bar_width, height))
                pygame.draw.rect(self.window, (50, 50, 50), 
                               (x_pos - bar_width//2, 350 - height, bar_width, height), 1)
                
                # Draw value text
                val_text = self.font.render(f"{val:.3f}", True, (30, 30, 120))
                self.window.blit(val_text, (x_pos - val_text.get_width()//2, 360))
                
                # Draw state label
                state_label = self.font.render(chr(65+i), True, (50, 50, 50))
                self.window.blit(state_label, (x_pos - state_label.get_width()//2, 320))
        
        # Draw ground truth reference
        if self.episode_count > 0:
            truth_label = self.font.render("Ground Truth: A=1/6, B=2/6, C=3/6, D=4/6, E=5/6", True, (120, 80, 40))
            self.window.blit(truth_label, (100, 380))
        
        pygame.display.flip()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.window = None
                return
        
        # Control rendering speed
        self.clock.tick(100)

def plot(data):
    groups = ['A', 'B', 'C', 'D', 'E']  # Group labels
    x = np.arange(len(groups))  # Create x-axis positions

    plt.figure(figsize=(10, 6))
    episode_num = [1, 5, 10, 100]

    # Plot each probability set with distinct colors
    for i, probs in enumerate(data):
        if i == 0:
            plt.plot(x, probs, 
                marker='o', 
                linestyle='-', 
                linewidth=2.5,
                markersize=8,
                label=f'ground truth')
        else:
            plt.plot(x, probs, 
                marker='o', 
                linestyle='-', 
                linewidth=2.5,
                markersize=8,
                label=f'Episode {episode_num[i-1]}')

    # Formatting the plot
    plt.xticks(x, groups)
    plt.ylim(0, 1.05)  # Add some headroom at top
    plt.xlabel('Groups', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Probability Distribution using TD(0)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    agent = RandomWalk()

    values = np.full(agent.nS, 0.5)
    values[0] = 0
    values[6] = 0

    data = [[1/6, 2/6, 3/6, 4/6, 5/6]]

    for i in range(EPISODE):
        s, _ = agent.reset()
        terminated = False

        print("Episode", i)
        while not terminated:
            agent.render(values)
            action = agent.pick_action()
            next_s, reward, terminated, _, _ = agent.step(action)
            # translation = ["bad_ter", "A", "B", "C", "D", "E", "ter"]
            # dir = ["l", "r"]
            values[s] = values[s] + TD_ALPHAS[1] * (reward + GAMMA * values[next_s] - values[s])
            # print("Picked action:", dir[action], " Next state: ", translation[next_s], "Value: ", values[s])
            s = next_s
        
        if i in [1-1, 5-1, 10-1, 100-1]:
            data.append([values[1], values[2], values[3], values[4], values[5]])
        
        print("Values:", values[1], values[2], values[3], values[4], values[5])
    
    plot(data)