# import gymnasium as gym
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle


# def run(episodes, render = False):
#     env = gym.make("FrozenLake-v1", map_name = "8x8", is_slippery = False, render_mode = 'human' if render else None)

#     q = np.zeros((env.observation_space.n, env.action_space.n))  # init a 64X4 array

#     learning_rate_a = 0.9
#     discount_factor_g = 0.9


#     #epsilon-greedy policy
#     epsilon = 1                   # 1 = 100% random actions
#     epsilon_decay_rate = 0.0001   # 1/0.0001 = 10,000
#     rng = np.random.default_rng() # random number generator

#     rewards_per_episode = np.zeros(episodes) # tracking -> how training progress
    
#     for i in range(episodes):

#         start = env.reset()[0]        # states: 0 to 63, 0=top left corner, 63 = bottom right corner
#         terminated = False            # True when fall in hole or reached goal 
#         truncated = False             # True when actions>200

   

#         while(not terminated and not truncated):

#             if rng.random() < epsilon:
#                 action = env.action_space.sample() # action: 0->left, 1->down, 2->right, 3->up
#             else:
#                 action = np.argmax(q[state, :])    

#             new_state, reward, terminated, truncated, _ = env.step(action)

#             q[state, action] = q[state, action] + learning_rate_a*(reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])

#             state = new_state

#         epsilon = max(epsilon - epsilon_decay_rate, 0)    

#         if (epsilon == 0):
#             learning_rate_a = 0.0001

#     env.close()

#     sum_rewards = np.zeros(episodes)
#     for t in range(episodes):
#         sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
#     plt.plot(sum_rewards)
#     plt.savefig('frozen_lake.png')

#     f = open('frozen_lake.png', 'wb')
#     pickle.dump(q, f)
#     f.close()

# if __name__ == "__main__":
#     run(15000)        



import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(episodes, render=False):
    env = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=False,
        render_mode="human" if render else None,
    )

    q = np.zeros((env.observation_space.n, env.action_space.n))  # Initialize Q-table

    learning_rate_a = 0.9
    discount_factor_g = 0.9

    # epsilon-greedy policy
    epsilon = 1.0  # 100% random actions initially
    epsilon_decay_rate = 0.0001  # Decay rate
    rng = np.random.default_rng()  # Random number generator

    rewards_per_episode = np.zeros(episodes)  # Tracking rewards per episode

    for i in range(episodes):
        state = env.reset()[0]  # Start state
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            # Select action
            if rng.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q[state, :])  # Exploit

            # Perform action
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Update Q-table
            q[state, action] += learning_rate_a * (
                reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
            )

            state = new_state
            total_reward += reward

        # Update epsilon with decay
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Update rewards
        rewards_per_episode[i] = total_reward

        # Reduce learning rate if epsilon reaches zero
        if epsilon == 0:
            learning_rate_a = 0.0001

    env.close()

    # Plotting rewards
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(sum_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Sum of Rewards (last 100 episodes)")
    plt.title("Training Progress - Frozen Lake")
    plt.savefig("frozen_lake_rewards.png")
    plt.close()

    # Save Q-table
    with open("frozen_lake_q_table.pkl", "wb") as f:
        pickle.dump(q, f)


if __name__ == "__main__":
    run(15000)
