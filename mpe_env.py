from pettingzoo.mpe import simple_speaker_listener_v4
import random
import numpy as np
from mpe_model import random_policy

# noise config
NOISE_PROB = 0.3    # temp drop prob p; TODO later we'll sweep [0.0, 0.1, 0.3, 0.5]
COMM_DIM = 3        # last 3 entries of listener obs = communication portion

# if the current agent is the listener, randomly drop the comm part of its observation
def apply_comm_dropout(agent_name, obs, drop_prob=NOISE_PROB, comm_dim=COMM_DIM):
    if obs is None:
        return obs
    
    if agent_name.startswith("listener"):
        if random.random() < drop_prob:
            obs = obs.copy()
            obs[-comm_dim:] = 0.0
    return obs

env = simple_speaker_listener_v4.env(render_mode='human')


env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    observation = apply_comm_dropout(agent, observation, drop_prob=NOISE_PROB, comm_dim=COMM_DIM)
    
    if termination or truncation:
        action = None
    else:
        action = random_policy(observation, env.action_space(agent), reward)
        #action = env.action_space(agent).sample() # this is where you would insert your policy

    env.step(action)
env.close()