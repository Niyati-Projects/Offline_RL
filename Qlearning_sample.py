import numpy as np
import random
import matplotlib.pyplot as plt

class GridEnvironment:
    def __init__(self, grid_size, goal_state, obstacle_states):
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.obstacle_states = obstacle_states
        self.actions = ['up', 'down', 'left', 'right']
        self.action_mapping = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

    def is_valid_state(self, state):
        x, y = state
        return (
            0 <= x < self.grid_size[0] and
            0 <= y < self.grid_size[1] and
            state not in self.obstacle_states
        )

# Q-Learning Agent
class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((env.grid_size[0] * env.grid_size[1], len(env.actions)))

    def state_to_index(self, state):
        return state[0] * self.env.grid_size[1] + state[1]

    def index_to_state(self, index):
        return divmod(index, self.env.grid_size[1])

    def choose_action(self, state_index):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(self.env.actions) - 1)  # Explore
        else:
            return np.argmax(self.q_table[state_index, :])  # Exploit

    def update_q_value(self, state_index, action, reward, next_state_index):
        td_target = reward + self.gamma * np.max(self.q_table[next_state_index, :])
        td_error = td_target - self.q_table[state_index, action]
        self.q_table[state_index, action] += self.alpha * td_error

# Training Loop
def train_agent(agent, num_episodes=1000, max_steps_per_episode=100):
    total_rewards = []

    for episode in range(num_episodes):  # Number of Episodes
        state = (0, 0)  # Start state
        total_reward = 0

        for _ in range(max_steps_per_episode):  # Steps per episode
            state_index = agent.state_to_index(state)
            action = agent.choose_action(state_index)
            action_move = agent.env.action_mapping[agent.env.actions[action]]
            next_state = (state[0] + action_move[0], state[1] + action_move[1])

            if not agent.env.is_valid_state(next_state):
                next_state = state

            next_state_index = agent.state_to_index(next_state)
            if next_state == agent.env.goal_state:
                reward = 100
                done = True
            elif next_state == state:
                reward = -10
            else:
                reward = -1

            agent.update_q_value(state_index, action, reward, next_state_index)
            state = next_state
            total_reward += reward

            if next_state == agent.env.goal_state:
                break

        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
        total_rewards.append(total_reward)

    return total_rewards

# Parameter Sensitivity Analysis
def parameter_analysis(env, param_name, param_values, num_episodes=1000):
    results = {}
    for param_value in param_values:
        agent_params = {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 1.0}
        agent_params[param_name] = param_value
        agent = QLearning(env, **agent_params)
        rewards = train_agent(agent, num_episodes)
        results[param_value] = rewards

    # Plot results
    plt.figure()
    for param_value, rewards in results.items():
        plt.plot(range(num_episodes), rewards, label=f"{param_name}={param_value}")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Discounted Reward")
    plt.title(f"Effect of {param_name} on Learning")
    plt.legend()
    plt.show()

def learn(env, runs, num_episodes=1000):
    all_rewards = []

    for r in range(runs):
        agent = QLearning(env)
        rewards = train_agent(agent, num_episodes)
        print(f"Run {r + 1}: Learned Q-table:")
        print(agent.q_table)
        all_rewards.append(rewards)
        plt.plot(range(len(rewards)), rewards, label=f"Run {r + 1}")

    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Learning Performance Across Runs")
    plt.legend()
    plt.show()

    return all_rewards

# Main Execution
if __name__ == "__main__":
    # Environment setup
    grid_size = (8, 8)
    goal_state = (5, 5)
    obstacle_states = [(1, 2), (2, 2), (1, 3)]
    env = GridEnvironment(grid_size, goal_state, obstacle_states)

    # Multiple runs
    trials = learn(env, 3)

    # Parameter sensitivity analysis
    parameter_analysis(env, 'alpha', [0.1, 0.5, 0.9])
    parameter_analysis(env, 'gamma', [0.5, 0.7, 0.9])
    parameter_analysis(env, 'epsilon', [1.0, 0.5, 0.1])
