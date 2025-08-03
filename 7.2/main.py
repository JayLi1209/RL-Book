# Exercise 7.2: n-step TD for random walk
import numpy as np
from random_walk import RandomWalk
import math

 # If N is small (N=2), it means we only consider the last two steps for each episode.
 # Since our policy is random, it would have much noise.
 # If N is too big, then we waste many timesteps by not taking their V (state-values) into account.
N = 5
GAMMA = 1
EPISODE = 100_000
ALPHA = 0.08

# target policy is random
if __name__ == "__main__":
    agent = RandomWalk()

    values = np.full(agent.nS, 0.5)
    values[0] = 0
    values[6] = 0

    for i in range(EPISODE):
        states_vals = []
        reward_vals = []
        T = math.inf
        t = 0
        s, _ = agent.reset()
        terminated = False
        states_vals.append(s)
        reward_vals.append(0)

        print("Episode", i)
        
        while True:
            agent.render(values)
            if t < T:
                action = agent.pick_action()
                next_s, reward, terminated, _, _ = agent.step(action)
                if terminated:
                    T = t + 1
                
                states_vals.append(next_s)
                reward_vals.append(reward) # (R,S) come in pairs. Need to add reward as well.

            tau = t - N + 1
            if tau >= 0:
                # translation = ["", "A", "B", "C", "D", "E", ""]
                G = 0
                for j in range(tau+1, min(tau+N, T)+1):
                    G += GAMMA ** (j-tau-1) * reward_vals[j]
                if tau + N < T:
                    G += GAMMA ** N * values[states_vals[tau + N]]
                # print("Updating", translation[states_vals[tau]], "with value ", values[states_vals[tau]], "and", G - values[states_vals[tau]])
                # print("reward_vals",reward_vals, "states_vals", states_vals)
                values[states_vals[tau]] += ALPHA * (G - values[states_vals[tau]])
            
            if tau == T - 1:
                break
            t += 1
        
        print("Values:", values[1], values[2], values[3], values[4], values[5])