

def random_policy(obs, agents, action_space):
    actions = {}
    for agent in agents:
        actions[agent] = action_space(agent).sample()
    return actions