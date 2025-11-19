from pettingzoo.mpe import simple_speaker_listener_v4

from mpe_model import random_policy

env = simple_speaker_listener_v4.env(render_mode='human')


env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = random_policy(observation, reward, env.action_space(agent))
        #action = env.action_space(agent).sample() # this is where you would insert your policy

    env.step(action)
env.close()