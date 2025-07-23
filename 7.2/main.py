# Exercise 7.2: n-step TD for random walk
import numpy as np
from random_walk import RandomWalk
from gymnasium import Env


N = 2
GAMMA = 1
EPISODE = 100
ALPHA = 0.1

if __name__ == "__main__":
    agent = RandomWalk()

    values = np.full(agent.nS, 0.5)
    values[0] = 0
    values[6] = 0

    for i in range(EPISODE):
        states_vals = []
        reward_vals = []
        T = 100_000
        t = 0
        s, _ = agent.reset()
        terminated = False
        states_vals.append(s)

        print("Episode", i)
        
        while not terminated:
            agent.render(values)
            if t < T:
                action = agent.pick_action()
                next_s, reward, terminated, _, _ = agent.step(action)
                if terminated:
                    T = t + 1
                
                states_vals.append(next_s)
                reward_vals.append(reward) # Reward and State indices are off by 1

            tau = t - N + 1
            if tau >= 0:
                G = 0
                for i in range(tau+1, min(tau+N, T)):
                    G += GAMMA ** (i-tau-1) * reward_vals[i]
                if tau + N < T:
                    G += GAMMA ** N * values[states_vals[tau + N]]
                values[states_vals[tau]] += ALPHA * (G - values[states_vals[tau]])
            
            if tau >= T - 1:
                break
            t += 1
            
            # action = agent.pick_action()
            # next_s, reward, terminated, _, _ = agent.step(action)
            # # translation = ["bad_ter", "A", "B", "C", "D", "E", "ter"]
            # # dir = ["l", "r"]
            # values[s] = values[s] + ALPHA * (reward + GAMMA * values[next_s] - values[s])
            # # print("Picked action:", dir[action], " Next state: ", translation[next_s], "Value: ", values[s])
            # s = next_s
        
        print("Values:", values[1], values[2], values[3], values[4], values[5])