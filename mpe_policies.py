import random
import numpy as np

def run_random(observation, action_space, reward):
    return random.choice(range(action_space.n))

# Simple SARSA policy
def train_sarsa(env, num_episodes, epsilon=0.1, alpha=0.5, gamma=0.9):
    # Initialize a Q-table for each agent in the env
    q_table = {agent: {} for agent in env.agents}

    # Helper function to get state-action pairs
    def get_q_value(agent, obs):
        state = tuple(np.round(obs, decimals=4))  # rounding for better hashing
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
                state = tuple(np.round(states[agent], decimals=4))
                next_state = tuple(np.round(observations[agent], decimals=4))
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

def run_sarsa(q_table, observation, action_space, agent_name):
    observation_rounded = tuple(np.round(observation, decimals=4))
    if observation_rounded not in q_table[agent_name]:
        q_table[agent_name][observation_rounded] = np.zeros(action_space.n)
    return int(np.argmax(q_table[agent_name][observation_rounded]))
    
def train_q_learning(env, epsilon=0.1, alpha=0.5, gamma=0.9, num_episodes=100):
    # Initialize a Q-table for each agent in the env
    env.reset()
    q_table = {agent: {} for agent in env.agents}

    # Helper function to get state-action pairs
    def get_q_value(agent, obs):
        state = tuple(np.round(obs, decimals=4))  # rounding for better hashing
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
                state = tuple(np.round(states[agent], decimals=4))
                next_state = tuple(np.round(observations[agent], decimals=4))
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

    return q_table

def run_q_learning(q_table, observation, action_space, agent_name):
    observation_rounded = tuple(np.round(observation, decimals=4))
    if observation_rounded not in q_table[agent_name]:
        q_table[agent_name][observation_rounded] = np.zeros(action_space.n)
    return int(np.argmax(q_table[agent_name][observation_rounded]))

# Other policy implementations ...
