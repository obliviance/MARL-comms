from pettingzoo.mpe import simple_speaker_listener_v4
import random
import numpy as np
from pathlib import Path
import json

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
        new_vec = np.zeros(COMM_DIM, dtype=obs[LISTENER_NAME].dtype)
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
    def __init__(self, render_mode, max_cycles, continuous_actions, noise_mode, noise_prob, policy_name, run_mode):
        self.noise_mode = noise_mode
        self.noise_prob = noise_prob
        self.env = simple_speaker_listener_v4.parallel_env(render_mode=render_mode, max_cycles=max_cycles, continuous_actions=continuous_actions)
        self.agents = [SPEAKER_NAME, LISTENER_NAME]
        self.policy_name = policy_name
        self.run_mode = run_mode

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
        self.episode_done = False
        self.episode_success = None

    def _finalize_episode(self):
        """Record metrics for the episode that just ended."""
        if self.episode_length == 0:
            return  # nothing to record (first reset before any steps)
        self.successes.append(int(bool(self.episode_success)))
        self.episode_lengths.append(self.episode_length)
        for agent in self.agents:
            ep_rewards = list(self.episode_rewards[agent])
            self.rewards_dict[agent].append(ep_rewards)
            self.cumulative_rewards[agent].append(float(sum(ep_rewards)))
            self.average_rewards[agent].append(float(np.mean(ep_rewards)) if ep_rewards else 0.0)
            self.episode_rewards[agent].clear()
        self.episode_length = 0
        self.episode_done = False
        self.episode_success = None
    
    def reset(self):
        self._finalize_episode()
        # Calling the original env reset
        observations, infos = self.env.reset()

        # Applying communication noise and masking landmarks for listener
        observations = apply_comm_noise(
            obs=observations,
            mode=self.noise_mode,
            noise_prob=self.noise_prob
        )
        observations = mask_listener_landmarks(observations)

        print(f"Episode {len(self.episode_lengths)} reset. Metrics updated.")

        return observations, infos
    
    def step(self, action):
        # Calling the original env step
        observations, rewards, terminations, truncations, infos = self.env.step(action)
        
        # Apply communication noise and mask landmarks for listener
        observations = apply_comm_noise(
            obs=observations,
            mode=self.noise_mode,
            noise_prob=self.noise_prob
        )
        observations = mask_listener_landmarks(observations)

        # Update reward and episode length metrics
        for agent in self.agents:
            self.episode_rewards[agent].append(rewards[agent])
        self.episode_length += 1
        # Mark episode done/success when any agent terminates or truncates
        if not self.episode_done and (any(terminations.values()) or any(truncations.values())):
            self.episode_done = True
            self.episode_success = all(terminations.values())
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        self.env.render()
    
    def close(self):
        self._finalize_episode()

        # Save results to a JSON file
        results = {
            "policy": self.policy_name,
            "run_mode": self.run_mode,
            "noise": {
                "mode": self.noise_mode,
                "prob": self.noise_prob
            },
            "successes": self.successes,
            "episode_lengths": self.episode_lengths,
            "agents": {
                agent: {
                    "episode_rewards": self.rewards_dict[agent],
                    "cumulative_rewards": self.cumulative_rewards[agent],
                    "average_rewards": self.average_rewards[agent]
                } for agent in self.agents
            }
        }

        # Write results to a JSON file in the eval_data directory
        results_filepath = Path(f"eval_data/{self.policy_name}_{self.run_mode}_{self.noise_mode}_{self.noise_prob}_results.json")
        results_filepath.write_text(json.dumps(results, indent=2))

        # Closing the original env
        self.env.close()

    def action_space(self, agent):
        return self.env.action_space(agent)
    
    def observation_space(self, agent):
        return self.env.observation_space(agent)