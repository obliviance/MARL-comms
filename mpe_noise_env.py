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

        # Dictionary to hold step rewards for each episode, for each agent
        self.rewards_dict = {agent: [] for agent in self.agents}
        # Step rewards for the current episode for each agent
        self.episode_rewards = {agent: [] for agent in self.agents}
        # Sum of the rewards in each episode for each agent
        self.cumulative_rewards = {agent: [] for agent in self.agents}
        # Average reward over each episode for each agent
        self.average_rewards = {agent: [] for agent in self.agents}
        # Record of each episode's success (1 for success, 0 for failure)
        self.successes = []
        # Length of each episode 
        self.episode_lengths = []
        self.episode_length = 0
    
    def reset(self):
        observations, infos = self.env.reset()
        observations = apply_comm_noise(
            obs=observations,
            mode=self.noise_mode,
            noise_prob=self.noise_prob
        )

        # Update metrics
        for agent in self.agents:
            self.rewards_dict[agent].append(self.episode_rewards[agent])
            self.cumulative_rewards[agent].append(sum(self.episode_rewards[agent]))
            self.average_rewards[agent].append(
                np.mean(self.episode_rewards[agent])
            )
            self.episode_rewards[agent] = []
        self.episode_lengths.append(self.episode_length)
        self.episode_length = 0

        return observations, infos
    
    def step(self, action):
        # Takes a step in the environment with noise applied to observations
        observations, rewards, terminations, truncations, infos = self.env.step(action)
        observations = apply_comm_noise(
            obs=observations,
            mode=self.noise_mode,
            noise_prob=self.noise_prob
        )

        # Update metrics
        for agent in self.agents:
            self.episode_rewards[agent].append(rewards[agent])
        self.episode_length += 1
        if all(terminations.values()):
            self.successes.append(1)
        else:
            self.successes.append(0)
        
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        self.env.render()
    
    def close(self):
        policy_name = "SARSA"
        with open(policy_name + "_results.txt", "a") as f:
            f.write(f"Results for Policy: {policy_name}, Noise Mode: {self.noise_mode}, Noise Probability: {self.noise_prob}\n\n")
            f.write("Successes over Episodes:\n")
            f.write(f"{self.successes}\n\n")
            f.write("Episode Lengths:\n")
            f.write(f"{self.episode_lengths}\n\n")
            for agent in self.agents:
                f.write(f"Agent: {agent}\n")
                for episode in range(len(self.rewards_dict[agent])):
                    f.write(f"Episode {episode + 1} Values:\n")
                    f.write(f"Cumulative Reward: {self.cumulative_rewards[agent][episode]}\n")
                    f.write(f"Average Reward: {self.average_rewards[agent][episode]}\n")
        self.env.close()

    def action_space(self, agent):
        return self.env.action_space(agent)
    
    def observation_space(self, agent):
        return self.env.observation_space(agent)