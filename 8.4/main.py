import numpy as np
import maze_world as MazeWorld
import matplotlib.pyplot as plt
import random
import math

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.95
EPISODES = 2000 # Alternative way to control runtime
TS_TRUNCATE = 3000 
CHANGE_TS = 1000 # Where to switch course
total_ts = 0
N = 25 # Number of planning steps
cults_counts = []
reward_counts = []
c_r = 0 # culmultive reward for plotting
k = 0.1 # The Dyna-Q+ parameter


mapping = {
    0: np.array([1, 0]),   # Down
    1: np.array([0, 1]),   # Right
    2: np.array([-1, 0]),  # Up
    3: np.array([0, -1])   # Left
}

def get_greedy_action(Q, cur_loc):
    return np.argmax(Q[cur_loc])

def get_max_action(Q, next_s):
    max_num = -math.inf
    max_a = 0
    for i in range(len(Q[next_s])):
        if Q[next_s][i] > max_num:
            max_num = Q[next_s][i]
            max_a = i

    return max_num, max_a

def get_action(agent, Q, cur_loc):
    chance = np.random.random()
    true_action = np.argmax(Q[cur_loc])
    if chance > EPSILON:
        return true_action
    
    random_action = np.random.randint(agent.nA)
    while random_action == true_action:
        random_action = np.random.randint(agent.nA)
    return random_action

# It's basically a dict, a lookup table.
class DynaQModel:
    def __init__(self):
        self.model = {}

    def update(self, state, action, reward, next_state): 
        self.model[(state, action)] = (reward, next_state)

    def sample(self):
        if not self.model:
            return None, None
        return random.choice(list(self.model.keys()))

    def predict(self, state, action):
        return self.model.get((state, action), None)

def initialize_history(agent):
    history = {}
    for i in range(agent.nS[0]):
        for j in range(agent.nS[1]):
            history[(i,j)] = []
    return history

def get_random_state_action(agent, history):
    while True:
        row = random.randint(0, agent.nS[0]-1)
        col = random.randint(0, agent.nS[1]-1)
        length = len(history[(row, col)])
        if length != 0:
            action_index = random.randint(0, length-1)
            return (row, col), history[(row, col)][action_index]

def get_reward_ts_plot(time_steps, rewards, reward_dyna_q_plus):
    # Plot cumulative reward vs. time steps
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, rewards, label="Dyna-Q", color="blue", linewidth=2)
    plt.plot(time_steps, reward_dyna_q_plus, label="Dyna-Q+", color="red", linewidth=2)


    # Customize the plot
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.title("Cumulative Reward Over Time Steps", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save or display
    plt.savefig("8.4/cumulative_reward.png", dpi=300)  # Save as PNG
    plt.show()



if __name__ == "__main__":

    agent = MazeWorld.MazeWorld(mapping,sto = False)
    Q = np.zeros((*agent.nS, agent.nA)) # action space
    model = DynaQModel()
    history = initialize_history(agent)
    ts_history = np.zeros((*agent.nS, agent.nA)) # For Dyna-Q+: how often has it been visited?

    change = False # switch maze?
    for i in range(EPISODES):
        if total_ts >= TS_TRUNCATE:
            break

        if total_ts >= CHANGE_TS:
            change = True
        print("Staring Episode: ", i)
        obs, _ = agent.reset()
        cur_a = get_action(agent, Q, obs)
        terminated = False
        cur_s = obs
        # print(history)
        # print(cur_s)
        tmp_ts = 0

        while not terminated:
            # agent.render()
            next_s, reward, terminated, truncated, _ = agent.step(cur_a, change)
            next_a_val, next_a = get_max_action(Q, next_s)
            Q[cur_s][cur_a] = Q[cur_s][cur_a] + ALPHA * (reward + GAMMA * next_a_val - Q[cur_s][cur_a])
            history[tuple(cur_s)].append(cur_a)
            reward_q_plus = reward + k * math.sqrt(total_ts - ts_history[tuple(cur_s)][cur_a])
            ts_history[tuple(cur_s)][cur_a] = total_ts
            model.update(cur_s, cur_a, reward, next_s)

            # print("reward", reward, cur_s)
            for j in range(N):
                rand_s, rand_a = get_random_state_action(agent, history)
                pred_r, pred_s = model.predict(rand_s, rand_a)
                next_pred_a_val, _ = get_max_action(Q, pred_s)
                Q[rand_s][rand_a] = Q[rand_s][rand_a] + ALPHA * (reward + GAMMA * next_pred_a_val - Q[rand_s][rand_a])

            c_r += reward
            cur_s = next_s
            cur_a = next_a
            total_ts += 1
            tmp_ts += 1            
        
        c_r += tmp_ts
        c_r -= 1
        print("Total timestep consumed: ", total_ts, "reward: ", c_r)
        cults_counts.append(total_ts)
        reward_counts.append(c_r)

    # print(total_ts)
    # print(c_r)
    get_reward_ts_plot(cults_counts, reward_counts)


