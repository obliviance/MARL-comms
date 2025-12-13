from pettingzoo.mpe import simple_speaker_listener_v4
import random
import numpy as np
from mpe_model import random_policy
import time

# noise config
NOISE_PROB = 0.3    # temp drop prob p; TODO later we'll sweep [0.0, 0.1, 0.3, 0.5]
COMM_DIM = 3        # last 3 entries of listener obs = communication portion

# if the current agent is the listener, randomly drop the comm part of its observation
# def apply_comm_dropout(agent_name, obs, drop_prob=NOISE_PROB, comm_dim=COMM_DIM):
#     if obs is None:
#         return obs
    
#     if agent_name.startswith("listener"):
#         if random.random() < drop_prob:
#             obs = obs.copy()
#             obs[-comm_dim:] = 0.0
#     return obs

# apply different types of noise to the comm part of listener's observations
# mode options:
# - none: no noise
# - dropout: zero out comm vector
# - gaussian: add N(0, sigma^2) noise to comm vector
# - flip: flip one hot comm to another random index
def apply_comm_noise(
    agent_name: str,
    obs: np.ndarray,
    mode: str,
    noise_prob: float = 0.3,
    comm_dim: int = COMM_DIM
):
    if obs is None:
        return obs
    
    if "listener" not in agent_name:
        return obs
    
    obs = obs.copy()
    comm_slice = slice(-comm_dim, None)
    
    if mode == "none":
        return obs
    
    if mode == "dropout":
        if random.random() < noise_prob:
            obs[comm_slice] = 0.0
            
    elif mode == "gaussian":
        if noise_prob > 0.0:
            noise = np.random.normal(0.0, noise_prob, size=comm_dim)
            obs[comm_slice] = obs[comm_slice] + noise
        
    elif mode == "flip":
        vec = obs[comm_slice]
        current_idx = int(np.argmax(vec))
        if random.random() < noise_prob:
            # choose a diff index
            candidates = [i for i in range(comm_dim) if i != current_idx]
            new_idx = random.choice(candidates)
        else:
            new_idx = current_idx
        new_vec = np.zeros(comm_dim, dtype=obs.dtype)
        new_vec[new_idx] = 1.0
        obs[comm_slice] = new_vec
    
    return obs

# zero out landmark relative position for listener
def mask_landmarks(obs: np.ndarray):
    # obs = [velocity_x, velocity_y, 6 landmark dims, 3 comm dims]
    masked = obs.copy()
    masked[2:8] = 0.0
    return masked

def run_episode():
    env = simple_speaker_listener_v4.env(render_mode='human', max_cycles=100, continuous_actions=False,)

    env.reset()
    step_count = 0
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        observation = apply_comm_noise(agent, observation, drop_prob=NOISE_PROB, comm_dim=COMM_DIM)
        
        if termination or truncation:
            action = None
        else:
            action = random_policy(observation, env.action_space(agent), reward)
            #action = env.action_space(agent).sample() # this is where you would insert your policy

        env.step(action)
        env.render()
        step_count += 1
        
        # slow things down to see movement
        time.sleep(0.03)

    print(f"Episode finished after {step_count} steps.")
    # keep window around until user presses Enter
    input("Press Enter to close the window...")
    env.close()

if __name__ == "__main__":
    run_episode()