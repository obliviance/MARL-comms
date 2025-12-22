import random
import numpy as np
import torch
from ppo_agent import Agent

def run_random(observation, action_space, reward):
    return random.choice(range(action_space.n))

def speaker_policy(obs: np.ndarray, action_space):
    """Heuristic speaker policy from train_idqn.py."""
    try:
        n = action_space.n
        goal_idx = int(np.argmax(obs[:n]))
        if 0 <= goal_idx < n:
            return goal_idx
    except Exception:
        pass
    return action_space.sample()

# Simple SARSA policy
def train_base_sarsa(env, num_episodes, epsilon=0.1, alpha=0.5, gamma=0.9):
    # Initialize a Q-table for each agent in the env
    q_table = {agent: {} for agent in env.agents}

    # Helper function to get state-action pairs
    def get_q_value(agent, obs):
        state = tuple(np.round(obs, decimals=1))  # rounding for better hashing
        if state not in q_table[agent]:
            q_table[agent][state] = np.zeros(env.action_space(agent).n)
        return q_table[agent][state]

    # Training loop
    while num_episodes > 0:
        # Initialize state and action variables
        observations, _ = env.reset()
        states = observations
        actions = {}

        # Choose initial actions using epsilon-greedy policy
        for agent in env.agents:
            state = states[agent]
            action_space = env.action_space(agent)
            if random.random() < epsilon:
                actions[agent] = random.choice(range(action_space.n))
            else:
                actions[agent] = np.argmax(get_q_value(agent, state))

        # Episode loop
        while True:
            # Take actions and observe next state and reward
            observations, rewards, terminations, truncations, _ = env.step(actions)
            # Choose next actions using epsilon-greedy policy
            next_actions = {}
            for agent in env.agents:
                next_state = observations[agent]
                next_action_space = env.action_space(agent)
                if random.random() < epsilon:
                    next_actions[agent] = random.choice(range(next_action_space.n))
                else:
                    next_actions[agent] = np.argmax(get_q_value(agent, next_state))
            
                # Setting current and next states and actions
                state = tuple(np.round(states[agent], decimals=1))
                next_state = tuple(np.round(observations[agent], decimals=1))
                action = actions[agent]
                next_action = next_actions[agent]

                # SARSA update
                if state not in q_table[agent]:
                    q_table[agent][state] = np.zeros(env.action_space(agent).n)
                q_table[agent][state][action] += alpha * (rewards[agent] + gamma * get_q_value(agent, next_state)[next_action] - get_q_value(agent, state)[action])

            # Setting current state and action to next for the next step
            states = observations
            actions = next_actions

            # Check for episode termination
            if all(terminations.values()) or all(truncations.values()):
                break

        num_episodes -= 1
    
    env.close()

    return q_table

def run_base_sarsa(q_table, observation, action_space, agent_name):
    observation_rounded = tuple(np.round(observation, decimals=1))
    if observation_rounded not in q_table[agent_name]:
        q_table[agent_name][observation_rounded] = np.zeros(action_space.n)
    return int(np.argmax(q_table[agent_name][observation_rounded]))
    
def train_base_q_learning(env, num_episodes, epsilon=0.1, alpha=0.5, gamma=0.9):
    # Initialize a Q-table for each agent in the env
    q_table = {agent: {} for agent in env.agents}

    # Helper function to get state-action pairs
    def get_q_value(agent, obs):
        state = tuple(np.round(obs, decimals=1))  # rounding for better hashing
        if state not in q_table[agent]:
            q_table[agent][state] = np.zeros(env.action_space(agent).n)
        return q_table[agent][state]

    # Training loop
    while num_episodes > 0:
        # Initialize state and action variables
        observations, _ = env.reset()
        states = observations

        # Episode loop
        while True:
            # Choose initial actions using epsilon-greedy policy
            actions = {}
            for agent in env.agents:
                state = states[agent]
                action_space = env.action_space(agent)
                if random.random() < epsilon:
                    actions[agent] = random.choice(range(action_space.n))
                else:
                    actions[agent] = np.argmax(get_q_value(agent, state))
            # Take actions and observe next state and reward
            observations, rewards, terminations, truncations, _ = env.step(actions)
            # Choose next actions using epsilon-greedy policy
            for agent in env.agents:            
                # Setting current and next states and actions
                state = tuple(np.round(states[agent], decimals=1))
                next_state = tuple(np.round(observations[agent], decimals=1))
                action = actions[agent]
                if state not in q_table[agent]:
                    q_table[agent][state] = np.zeros(env.action_space(agent).n)

                # Q-learning update
                q_table[agent][state][action] += alpha * (rewards[agent] + gamma * np.max(get_q_value(agent, next_state)) - get_q_value(agent, state)[action])

            # Setting current state to next for the next step
            states = observations

            # Check for episode termination
            if all(terminations.values()) or all(truncations.values()):
                break

        num_episodes -= 1

    env.close()

    return q_table

def run_base_q_learning(q_table, observation, action_space, agent_name):
    observation_rounded = tuple(np.round(observation, decimals=1))
    if observation_rounded not in q_table[agent_name]:
        q_table[agent_name][observation_rounded] = np.zeros(action_space.n)
    return int(np.argmax(q_table[agent_name][observation_rounded]))

def run_ppo(agents, observation, action_space, agent_name):
    if agent_name == "speaker_0":
        return speaker_policy(observation, action_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_tensor = torch.tensor(observation, dtype=torch.float32).to(device).unsqueeze(0)
    with torch.no_grad():
        action, _, _, _ = agents[agent_name].get_action_and_value(obs_tensor)
    return int(action.item())

def load_ppo_model(path, env, device):
    agents = {
        agent: Agent(obs_size=env.observation_space(agent).shape[0], 
                     num_actions=env.action_space(agent).n).to(device)
        for agent in env.agents if agent != "speaker_0"
    }
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    for agent_name in agents:
        agents[agent_name].load_state_dict(checkpoint[agent_name])
        agents[agent_name].eval()
    return agents

# Other policy implementations ...
