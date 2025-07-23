import numpy as np
from gymnasium import Env
import matplotlib.pyplot as plt


TD_ALPHAS = [0.05, 0.1, 0.15]
MC_ALPHAS = [0.01, 0.02, 0.03, 0.04]
GAMMA = 1
EPISODE = 100
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
    

        
