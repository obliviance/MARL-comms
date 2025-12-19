from pettingzoo.mpe import simple_speaker_listener_v4
import random
import numpy as np

SPEAKER_NAME = "speaker_0"
LISTENER_NAME = "listener_0"
COMM_DIM = 3

def apply_comm_noise(
    obs: np.ndarray,
    mode: str,
    noise_prob: float,
):
    if obs is None:
        return obs
    
    obs = obs.copy()
    comm_slice = slice(-COMM_DIM, None)
    
    if mode == "none":
        return obs
    
    if mode == "dropout":
        if random.random() < noise_prob:
            obs[LISTENER_NAME][comm_slice] = 0.0
            
    elif mode == "gaussian":
        if noise_prob > 0.0:
            noise = np.random.normal(0.0, noise_prob, size=COMM_DIM)
            obs[LISTENER_NAME][comm_slice] = obs[LISTENER_NAME][comm_slice] + noise
        
    elif mode == "flip":
        vec = obs[LISTENER_NAME][comm_slice]
        current_idx = int(np.argmax(vec))
        if random.random() < noise_prob:
            # choose a diff index
            candidates = [i for i in range(COMM_DIM) if i != current_idx]
            new_idx = random.choice(candidates)
        else:
            new_idx = current_idx
        new_vec = np.zeros(COMM_DIM, dtype=obs.dtype)
        new_vec[new_idx] = 1.0
        obs[LISTENER_NAME][comm_slice] = new_vec
    
    return obs

# zero out landmark relative position for listener
def mask_listener_landmarks(obs: np.ndarray):
    # obs = [velocity_x, velocity_y, 6 landmark dims, 3 comm dims]
    masked = obs.copy()
    masked[LISTENER_NAME][2:8] = 0.0
    return masked

class MPEEnv:
    def __init__(self, render_mode, max_cycles, continuous_actions, noise_mode, noise_prob):
        self.noise_mode = noise_mode
        self.noise_prob = noise_prob
        self.env = simple_speaker_listener_v4.parallel_env(render_mode=render_mode, max_cycles=max_cycles, continuous_actions=continuous_actions)
        self.agents = [SPEAKER_NAME, LISTENER_NAME]
    
    def reset(self):
        observations, infos = self.env.reset()
        observations = apply_comm_noise(
            obs=observations,
            mode=self.noise_mode,
            noise_prob=self.noise_prob
        )
        return observations, infos
    
    def step(self, action):
        observations, rewards, terminations, truncations, infos = self.env.step(action)
        observations = apply_comm_noise(
            obs=observations,
            mode=self.noise_mode,
            noise_prob=self.noise_prob
        )
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()

    def action_space(self, agent):
        return self.env.action_space(agent)
    
    def observation_space(self, agent):
        return self.env.observation_space(agent)