import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import numpy as np

# Parameters for the Grid World
grid_size = (8, 8)  # Grid size
goal_state = (7, 7)  # Goal position (zero-based index)
start_state = (0, 0)  # Starting position
#obstacle_state = (4, 4) # Obstacle position
max_episodes = 1000  # Number of episodes
max_steps = 50  # Max steps per episode
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.8  # Exploration rate
cumulative_reward = 0

# Initialize Q-table
Q = np.zeros((*grid_size, 4))  # 4 actions: up, down, left, right

# Actions: (row change, col change)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def extract_policy(Q):
    policy = np.full(grid_size, -1, dtype=int) # Initialize policy with -1
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if (i, j) == goal_state:
                continue  # No action needed at the goal state
            policy[i, j] = np.argmax(Q[i, j, :])
    return policy
# Training Loop (Q-learning)
for episode in range(max_episodes):
    state = start_state
    for step in range(max_steps):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.randint(4)  # Random action (explore)
        else:
            action = np.argmax(Q[state[0], state[1], :])  # Best action (exploit)
        # Get next state based on action
        next_state = (max(0, min(grid_size[0] - 1, state[0] + actions[action][0])),
                      max(0, min(grid_size[1] - 1, state[1] + actions[action][1])))
        
        # Calculate reward
        reward = 1 if next_state == goal_state else -0.1
        cumulative_reward += reward
        # Q-learning update
        Q[state[0], state[1], action] += alpha * (
            reward + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action]
        )
        
        state = next_state
        if state == goal_state:
            break

# Extract policy
policy = extract_policy(Q)
scipy.io.savemat("policy.mat", {"policy": policy})
print("Learned Policy:")
print(policy)

# Visualization
# fig, ax = plt.subplots()
# ax.set_xticks(np.arange(grid_size[1] + 1) - 0.5, minor=True)
# ax.set_yticks(np.arange(grid_size[0] + 1) - 0.5, minor=True)
# ax.grid(which="minor", color='k', linestyle='solid', linewidth=0.5)
# ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

# # Mark start and goal positions
# ax.plot(start_state[1], start_state[0], 'go', markersize=10, label='Start')
# #ax.plot(obstacle_state[1], obstacle_state[0], 'ro', markersize=10, label='Obstacle')
# ax.plot(goal_state[1], goal_state[0], 'ro', markersize=10, label='Goal')

# Animate the learned policy
state = start_state
for _ in range(max_steps):
    action = np.argmax(Q[state[0], state[1], :])
    next_state = (max(0, min(grid_size[0] - 1, state[0] + actions[action][0])),
                  max(0, min(grid_size[1] - 1, state[1] + actions[action][1])))
    
    # ax.plot(state[1], state[0], 'yo', markersize=8)
    # plt.pause(0.5)
    state = next_state
    if state == goal_state:
        print("Goal reached!")
        break

# plt.legend()
# plt.show()
# print("Cumulative Reward: ", cumulative_reward)
