import numpy as np
import grid_world as GridWorld
import matplotlib.pyplot as plt
import matplotlib.patches as patches


EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1
EPISODES = 170
TS_TRUNCATE = 8000
total_ts = 0
cults_counts = []

mapping = {
    0: np.array([1, 0]),   # Down
    1: np.array([0, 1]),   # Right
    2: np.array([-1, 0]),  # Up
    3: np.array([0, -1])   # Left
}

king_mapping = {
    0: np.array([1, 0]),   # Down
    1: np.array([0, 1]),   # Right
    2: np.array([-1, 0]),  # Up
    3: np.array([0, -1]),   # Left
    4: np.array([1, 1]),   # DownRight
    5: np.array([1, -1]),   # DownLeft
    6: np.array([-1, 1]),   # UpRight
    7: np.array([-1, -1]),   # UpLeft
}

king_mapping_no_moves = {
    0: np.array([1, 0]),   # Down
    1: np.array([0, 1]),   # Right
    2: np.array([-1, 0]),  # Up
    3: np.array([0, -1]),   # Left
    4: np.array([1, 1]),   # DownRight
    5: np.array([1, -1]),   # DownLeft
    6: np.array([-1, 1]),   # UpRight
    7: np.array([-1, -1]),   # UpLeft
    8: np.array([0, 0]) # No move
}

agent = GridWorld.GridWorld(mapping, sto = False)

Q = np.zeros((*agent.nS, agent.nA))

def get_greedy_action(Q, cur_loc):
    return np.argmax(Q[cur_loc])

def get_action(agent, Q, cur_loc):
    chance = np.random.random()
    true_action = np.argmax(Q[cur_loc])
    if chance > EPSILON:
        return true_action
    
    random_action = np.random.randint(agent.nA)
    while random_action == true_action:
        random_action = np.random.randint(agent.nA)
    return random_action


def plot_trajectory(trajectory):
    # Grid dimensions
    rows, cols = 7, 10
    start, goal = (3, 0), (3, 7)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Set background to light gray (like Pygame)
    fig.set_facecolor((240/255, 240/255, 240/255))
    ax.set_facecolor((240/255, 240/255, 240/255))
    
    # Draw grid cells
    for r in range(rows):
        for c in range(cols):
            facecolor = (
                (100/255, 200/255, 100/255) if (r, c) == start else  # Start: green
                (200/255, 100/255, 100/255) if (r, c) == goal else   # Goal: red
                (1, 1, 1)                                            # Regular: white
            )
            rect = patches.Rectangle(
                (c, r), 1, 1, 
                facecolor=facecolor,
                edgecolor=(180/255, 180/255, 180/255),
                linewidth=1
            )
            ax.add_patch(rect)
    
    # Plot trajectory path (blue line connecting centers)
    if trajectory:
        x = [c + 0.5 for r, c in trajectory]
        y = [r + 0.5 for r, c in trajectory]
        ax.plot(x, y, 'b-', linewidth=2)
        
        # Draw agent positions as blue circles (like Pygame)
        for r, c in trajectory:
            circle = patches.Circle(
                (c + 0.5, r + 0.5),
                radius=0.33,
                facecolor=(50/255, 50/255, 200/255),
                edgecolor='none'
            )
            ax.add_patch(circle)
    
    # Set limits and invert y-axis (row 0 at top)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_ts_vs_episode():
    plt.figure(figsize=(8, 5))
    plt.plot(cults_counts, range(len(cults_counts)), marker='o', linestyle='-')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode')
    plt.title('Episode vs. Timesteps')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_optimal_trajectory(env, Q, max_steps=100):
    state, _ = env.reset()
    trajectory = [state]  # Start with initial state
    terminated = False
    step_count = 0
    
    while not terminated and step_count < max_steps:
        # Choose action with highest Q-value
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        trajectory.append(next_state)
        state = next_state
        step_count += 1
        print("Optimal Trajectory: ", state)
    
    return trajectory

if __name__ == "__main__":
    for i in range(EPISODES):
        print("Staring Episode: ", i)
        obs, _ = agent.reset()
        cur_a = get_action(agent, Q, obs)
        terminated = False
        cur_s = obs

        while not terminated and total_ts <= TS_TRUNCATE:
            agent.render()
            next_s, reward, terminated, truncated, _ = agent.step(cur_a)
            next_a = get_action(agent, Q, next_s)
            Q[cur_s][cur_a] = Q[cur_s][cur_a] + ALPHA * (reward + GAMMA * Q[next_s][next_a] - Q[cur_s][cur_a])
            cur_s = next_s
            cur_a = next_a
            total_ts += 1
        
        print("Total timestep consumed: ", total_ts)
        cults_counts.append(total_ts)
    
    # Plot the optimal graph plot:
    print("Start optimal trajectory: ")
    trajectory = get_optimal_trajectory(agent, Q)
    
    plot_trajectory(trajectory)
    plot_ts_vs_episode()